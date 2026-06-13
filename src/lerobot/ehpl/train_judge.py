#!/usr/bin/env python
"""
Train a Segment-level Advantage Judge for EHPL (paper-aligned).

Inputs:
  - candidates parquet produced by `build_pairs_p2.py`
  - `dataset_root` to load (image, task) at the segment start
  - label is episode-level `episode_success` ('success'/'failure') -> y in {0,1}

Objective:
  A(c,u) = Q(c,u) - V(c)

Loss:
  BCEWithLogits(A(c,u), y)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from lerobot.ehpl.candidates_p2_dataset import EhplCandidatesP2Dataset
from lerobot.ehpl.configuration_ehpl import EhplScoringConfig
from lerobot.ehpl.modeling_ehpl import AdvantageJudge, AdvantageJudgeConfig, EhplFrozenContextEncoder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--parquet", type=str, required=True, help="Candidates parquet from build_pairs_p2.py")

    p.add_argument("--action_encoder", type=str, default="mlp_pool", choices=["mlp_pool", "transformer"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_rows", type=int, default=None)

    p.add_argument("--save_dir", type=str, default="outputs/judge_train")
    p.add_argument("--save_every", type=int, default=1)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ds = EhplCandidatesP2Dataset(dataset_root=args.dataset_root, candidates_parquet=args.parquet)
    context_encoder = EhplFrozenContextEncoder(EhplScoringConfig())
    scoring_cfg = context_encoder.cfg

    # Split train/val
    n_val = int(round(len(ds) * float(args.val_frac)))
    n_val = min(max(n_val, 0), len(ds) - 1) if len(ds) > 1 else 0
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate,
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=_collate,
            drop_last=False,
        )
        if n_val > 0
        else None
    )

    cfg = AdvantageJudgeConfig(
        context_dim=scoring_cfg.context_dim,
        action_dim=int(getattr(ds, "act_dim", 7)),
        action_encoder=args.action_encoder,  # type: ignore[arg-type]
    )
    model = AdvantageJudge(cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Save training config for reproducibility
    (save_dir / "train_config.json").write_text(
        json.dumps(
            {
                "parquet": args.parquet,
                "dataset_root": args.dataset_root,
                "dataset": {
                    "context_dim": getattr(ds, "context_dim", None),
                    "fixed_T": getattr(ds, "fixed_T", None),
                    "action_dim": getattr(ds, "action_dim", None),
                    "n": len(ds),
                },
                "model": asdict(cfg),
                "scoring": asdict(scoring_cfg) if scoring_cfg is not None else None,
                "train": {
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "val_frac": args.val_frac,
                    "seed": args.seed,
                },
            },
            indent=2,
        )
    )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=True)
        for batch in pbar:
            # Images: use the main camera key used by EhplFrozenContextEncoder (first valid cam)
            img = batch.get("observation.images.image")
            if img is None:
                raise ValueError(
                    "Missing `observation.images.image` in dataset. "
                    "Ensure your dataset has videos (v2.1) or inline images (v3.0)."
                )
            images = img.to(device, non_blocking=True).unsqueeze(1)  # [B,1,C,H,W]
            image_attention_mask = torch.ones(images.shape[0], 1, device=images.device, dtype=torch.bool)
            text = batch["task"]
            with torch.no_grad():
                context = context_encoder(images=images, image_attention_mask=image_attention_mask, text=text)
            context = context.to(device)

            action = batch["action"].to(device, non_blocking=True)
            out = model(context, action)
            a = out["A"]

            if "success" not in batch:
                continue
            y = batch["success"].to(device, non_blocking=True)
            mask = torch.isfinite(y)
            if not bool(mask.any()):
                continue
            loss = F.binary_cross_entropy_with_logits(a[mask], y[mask])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            pbar.set_postfix(
                {
                    "loss": f"{loss.detach().item():.4f}",
                }
            )

        # Validation
        val_metrics = None
        if val_loader is not None:
            model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_loader:
                    img = batch.get("observation.images.image")
                    if img is None:
                        continue
                    images = img.to(device, non_blocking=True).unsqueeze(1)
                    image_attention_mask = torch.ones(images.shape[0], 1, device=images.device, dtype=torch.bool)
                    text = batch["task"]
                    context = context_encoder(images=images, image_attention_mask=image_attention_mask, text=text).to(device)
                    action = batch["action"].to(device, non_blocking=True)
                    out = model(context, action)
                    a = out["A"]
                    if "success" not in batch:
                        continue
                    y = batch["success"].to(device, non_blocking=True)
                    m = torch.isfinite(y)
                    if not bool(m.any()):
                        continue
                    losses.append(F.binary_cross_entropy_with_logits(a[m], y[m]).detach().item())
            val_metrics = {"loss": float(np.mean(losses)) if losses else float("nan")}
            print(f"[val] epoch={epoch} loss={val_metrics['loss']:.4f}")

        # Save checkpoint
        if epoch % int(args.save_every) == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": asdict(cfg),
                "scoring_config": asdict(scoring_cfg) if scoring_cfg is not None else None,
                "dataset": {
                    "context_dim": getattr(ds, "context_dim", None),
                    "fixed_T": getattr(ds, "fixed_T", None),
                    "action_dim": getattr(ds, "action_dim", None),
                },
                "val_metrics": val_metrics,
            }
            ckpt_path = save_dir / f"judge_epoch_{epoch:03d}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()

