#!/usr/bin/env python
"""
Select (winner, loser) preference pairs from same-start candidates using a trained judge.

Input:
  - candidates parquet produced by `build_pairs_p2.py`
  - dataset_root to load the segment-start observation (image + task)
  - judge checkpoint produced by `train_judge.py --mode success`

Output:
  - P2 pairs parquet with columns:
      segment_start_index, episode_index, frame_index, task_index, h_seg, act_dim,
      action_w, action_l, action_mask,
      score_w, score_l, margin
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.ehpl.candidates_p2_dataset import EhplCandidatesP2Dataset
from lerobot.ehpl.configuration_ehpl import EhplScoringConfig
from lerobot.ehpl.modeling_ehpl import AdvantageJudge, AdvantageJudgeConfig, EhplFrozenContextEncoder


def _load_model(ckpt_path: str | Path, device: torch.device) -> tuple[AdvantageJudge, dict[str, Any]]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model_cfg_dict = ckpt.get("model_config")
    if not isinstance(model_cfg_dict, dict):
        raise ValueError("Checkpoint missing 'model_config' dict.")
    cfg = AdvantageJudgeConfig(**model_cfg_dict)
    model = AdvantageJudge(cfg).to(device)
    sd = ckpt.get("model_state_dict")
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint missing 'model_state_dict'.")
    model.load_state_dict(sd, strict=True)
    model.eval()
    scoring_cfg = ckpt.get("scoring_config")
    return model, {"scoring_config": scoring_cfg, "ckpt": ckpt}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Path to judge checkpoint .pt")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--candidates_parquet", type=str, required=True)
    p.add_argument("--out_parquet", type=str, required=True)
    p.add_argument("--min_margin", type=float, default=None, help="Optional margin threshold τ; filters pairs.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    ds = EhplCandidatesP2Dataset(dataset_root=args.dataset_root, candidates_parquet=args.candidates_parquet)

    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k in batch[0].keys():
            if k == "task":
                out[k] = [b[k] for b in batch]
            elif torch.is_tensor(batch[0][k]):
                out[k] = torch.stack([b[k] for b in batch], dim=0)
            else:
                out[k] = [b[k] for b in batch]
        return out

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate,
        drop_last=False,
    )

    model, meta = _load_model(args.ckpt, device=device)

    scoring_cfg_dict = meta.get("scoring_config")
    scoring_cfg = EhplScoringConfig(**scoring_cfg_dict) if isinstance(scoring_cfg_dict, dict) else EhplScoringConfig()
    context_encoder = EhplFrozenContextEncoder(scoring_cfg)

    # Accumulate per-row scores and grouping keys
    scores: list[np.ndarray] = []
    keys: list[tuple[int, int, int, int, int]] = []  # (episode, seg_start, frame, task, cand_idx)
    flat_actions: list[list[float]] = []
    flat_masks: list[list[int]] = []
    h_seg = ds.h_seg
    act_dim = ds.act_dim

    with torch.no_grad():
        for batch in tqdm(loader, desc="scoring candidates"):
            img = batch.get("observation.images.image")
            if img is None:
                raise ValueError("Missing `observation.images.image` in dataset.")
            images = img.to(device, non_blocking=True).unsqueeze(1)  # [B,1,C,H,W]
            image_attention_mask = torch.ones(images.shape[0], 1, device=images.device, dtype=torch.bool)
            text = batch["task"]
            context = context_encoder(images=images, image_attention_mask=image_attention_mask, text=text).to(device)

            action = batch["action"].to(device, non_blocking=True)
            out = model(context, action)
            a = out["A"].detach().to("cpu", dtype=torch.float32).numpy()
            scores.append(a)

            ep = batch["episode_index"].detach().to("cpu").numpy().astype(np.int64)
            seg = batch["index"].detach().to("cpu").numpy().astype(np.int64)
            frm = batch["frame_index"].detach().to("cpu").numpy().astype(np.int64)
            task = batch["task_index"].detach().to("cpu").numpy().astype(np.int64)
            cand = batch["candidate_index"].detach().to("cpu").numpy().astype(np.int64)
            for i in range(len(a)):
                keys.append((int(ep[i]), int(seg[i]), int(frm[i]), int(task[i]), int(cand[i])))
            # persist original flattened action/mask from parquet via dataset tensors
            # (to avoid numeric drift, we re-flatten from tensors)
            act_np = action.detach().to("cpu", dtype=torch.float32).numpy()
            m_np = batch["action_mask"].detach().to("cpu").numpy().astype(np.uint8)
            for i in range(act_np.shape[0]):
                flat_actions.append(act_np[i].reshape(-1).astype(np.float32).tolist())
                flat_masks.append(m_np[i].astype(np.uint8).tolist())

    score_all = np.concatenate(scores, axis=0)
    if score_all.shape[0] != len(keys):
        raise RuntimeError("Score/key length mismatch.")

    # Group by same-start key (episode_index, segment_start_index)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row_idx, (ep, seg, _frm, _task, _cand) in enumerate(keys):
        groups[(ep, seg)].append(row_idx)

    out_rows: list[dict[str, Any]] = []
    for (ep, seg), idxs in groups.items():
        s = score_all[idxs]
        w_local = int(np.argmax(s))
        l_local = int(np.argmin(s))
        w_i = idxs[w_local]
        l_i = idxs[l_local]

        _, _, frame_w, task_w, _ = keys[w_i]
        # frame/task should match within group
        margin = float(score_all[w_i] - score_all[l_i])
        if args.min_margin is not None and margin < float(args.min_margin):
            continue

        out_rows.append(
            {
                "segment_start_index": int(seg),
                "episode_index": int(ep),
                "frame_index": int(frame_w),
                "task_index": int(task_w),
                "h_seg": int(h_seg),
                "act_dim": int(act_dim),
                "action_w": flat_actions[w_i],
                "action_l": flat_actions[l_i],
                "action_mask": flat_masks[w_i],
                "score_w": float(score_all[w_i]),
                "score_l": float(score_all[l_i]),
                "margin": float(margin),
            }
        )

    if not out_rows:
        raise RuntimeError("No pairs produced. Check min_margin or input candidates.")

    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(out_rows)
    pq.write_table(table, str(out_path))

    summary = {
        "ckpt": str(args.ckpt),
        "dataset_root": str(args.dataset_root),
        "candidates_parquet": str(args.candidates_parquet),
        "out_parquet": str(out_path),
        "n_pairs": int(table.num_rows),
        "min_margin": args.min_margin,
        "model_config": asdict(model.cfg),
        "scoring_config": asdict(scoring_cfg),
    }
    (out_path.parent / (out_path.stem + "_summary.json")).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

