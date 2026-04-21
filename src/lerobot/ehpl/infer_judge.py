#!/usr/bin/env python
"""
Infer/score skill-level preference pairs with a trained AdvantageJudge.

This script loads a checkpoint produced by `train_judge.py`, runs inference on
an input parquet, and writes a new parquet with score columns added.

Supported input formats:
  A) Preference pairs (recommended):
     - context: 1D vector [context_dim]
     - action_w: winner chunk (nested [T,A] or flattened [T*A])
     - action_l: loser  chunk (nested [T,A] or flattened [T*A])
     - (optional) success: 0/1

Outputs added:
  - score_w: A(c, action_w)
  - score_l: A(c, action_l)
  - margin:  score_w - score_l
  - pred_wins: bool (score_w > score_l)
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.ehpl.modeling_ehpl import AdvantageJudge, AdvantageJudgeConfig
from lerobot.ehpl.processor_ehpl import SkillPreferenceDataset, skill_preference_collate_fn


def _load_model(ckpt_path: str | Path, device: torch.device) -> AdvantageJudge:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model_cfg_dict = ckpt.get("model_config")
    if not isinstance(model_cfg_dict, dict):
        raise ValueError("Checkpoint missing 'model_config' dict. Was it produced by train_judge.py?")
    cfg = AdvantageJudgeConfig(**model_cfg_dict)
    model = AdvantageJudge(cfg).to(device)
    sd = ckpt.get("model_state_dict")
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint missing 'model_state_dict'.")
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Path to judge checkpoint .pt")
    p.add_argument("--parquet", type=str, required=True, help="Input parquet (pairs).")
    p.add_argument("--out_parquet", type=str, required=True, help="Output parquet with scores added.")

    p.add_argument("--context_key", type=str, default="context")
    p.add_argument("--action_w_key", type=str, default="action_w")
    p.add_argument("--action_l_key", type=str, default="action_l")
    p.add_argument("--success_key", type=str, default=None)

    p.add_argument("--T", type=int, default=None, help="Fixed chunk length if actions are flattened.")
    p.add_argument("--action_dim", type=int, default=None, help="Action dim if actions are flattened.")
    p.add_argument("--max_rows", type=int, default=None)

    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    ds = SkillPreferenceDataset(
        args.parquet,
        context_key=args.context_key,
        action_w_key=args.action_w_key,
        action_l_key=args.action_l_key,
        success_key=args.success_key,
        action_dim=args.action_dim,
        fixed_T=args.T,
        max_rows=args.max_rows,
    )

    model = _load_model(args.ckpt, device=device)
    if model.cfg.context_dim != int(ds.context_dim):
        raise ValueError(f"context_dim mismatch: model={model.cfg.context_dim} vs data={ds.context_dim}")
    if ds.action_dim is None:
        raise ValueError("Dataset action_dim is None; please pass --action_dim (and optionally --T).")
    if model.cfg.action_dim != int(ds.action_dim):
        raise ValueError(f"action_dim mismatch: model={model.cfg.action_dim} vs data={ds.action_dim}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=skill_preference_collate_fn,
        drop_last=False,
    )

    scores_w: list[np.ndarray] = []
    scores_l: list[np.ndarray] = []
    for batch in loader:
        context = batch["context"].to(device)
        aw = batch["action_w"].to(device)
        al = batch["action_l"].to(device)

        out_w = model(context, aw)
        out_l = model(context, al)
        a_w = out_w["A"].detach().to("cpu", dtype=torch.float32).numpy()
        a_l = out_l["A"].detach().to("cpu", dtype=torch.float32).numpy()
        scores_w.append(a_w)
        scores_l.append(a_l)

    score_w = np.concatenate(scores_w, axis=0)
    score_l = np.concatenate(scores_l, axis=0)
    if score_w.shape[0] != len(ds) or score_l.shape[0] != len(ds):
        raise RuntimeError("Score length mismatch; unexpected batching bug.")

    margin = score_w - score_l
    pred_wins = score_w > score_l

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(args.parquet)
    if args.max_rows is not None:
        table = table.slice(0, int(args.max_rows))

    table = table.append_column("score_w", pa.array(score_w.astype(np.float32)))
    table = table.append_column("score_l", pa.array(score_l.astype(np.float32)))
    table = table.append_column("margin", pa.array(margin.astype(np.float32)))
    table = table.append_column("pred_wins", pa.array(pred_wins))

    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out_path))

    # Also write a tiny sidecar summary for quick sanity-checking
    summary = {
        "ckpt": str(args.ckpt),
        "parquet": str(args.parquet),
        "out_parquet": str(out_path),
        "n": int(table.num_rows),
        "score_w": {
            "min": float(np.min(score_w)),
            "max": float(np.max(score_w)),
            "mean": float(np.mean(score_w)),
        },
        "score_l": {
            "min": float(np.min(score_l)),
            "max": float(np.max(score_l)),
            "mean": float(np.mean(score_l)),
        },
        "margin": {
            "min": float(np.min(margin)),
            "max": float(np.max(margin)),
            "mean": float(np.mean(margin)),
        },
        "pred_wins_rate": float(np.mean(pred_wins.astype(np.float32))),
        "model_config": asdict(model.cfg),
    }
    (out_path.parent / (out_path.stem + "_summary.json")).write_text(
        __import__("json").dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

