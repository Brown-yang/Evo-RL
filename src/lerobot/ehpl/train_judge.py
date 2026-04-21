#!/usr/bin/env python
"""
Train a Segment-level Advantage Judge on skill-level preference pairs.

Data format (parquet):
  - context: 1D vector, shape [context_dim]
  - action_w: winner action chunk, shape [T, action_dim] (or flattened [T*action_dim])
  - action_l: loser action chunk, shape [T, action_dim] (or flattened [T*action_dim])
  - success (optional): 0/1 (episode-level or segment-level)

Objective:
  A(c,u) = Q(c,u) - V(c)

Loss:
  1) Pairwise ranking loss:
       L_rank = -logsigmoid(A_w - A_l)
  2) Value loss (optional if success exists):
       L_value = (V(c) - success)^2
  3) Total:
       L = λ1 * L_rank + λ2 * L_value

Usage example:
  python -m lerobot.ehpl.train_judge \\
    --parquet /path/to/pairs.parquet \\
    --context_key context \\
    --action_w_key action_w \\
    --action_l_key action_l \\
    --success_key success \\
    --batch_size 256 --lr 3e-4 --epochs 5 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from lerobot.ehpl.modeling_ehpl import AdvantageJudge, AdvantageJudgeConfig
from lerobot.ehpl.processor_ehpl import SkillPreferenceDataset, skill_preference_collate_fn


@torch.no_grad()
def evaluate(
    model: AdvantageJudge,
    loader: DataLoader,
    device: torch.device,
    *,
    lambda_rank: float,
    lambda_value: float,
    use_value_loss: bool,
) -> dict[str, float]:
    model.eval()
    losses = []
    rank_losses = []
    value_losses = []
    correct = 0
    total = 0
    for batch in loader:
        context = batch["context"].to(device)
        aw = batch["action_w"].to(device)
        al = batch["action_l"].to(device)

        out_w = model(context, aw)
        out_l = model(context, al)

        a_w = out_w["A"]
        a_l = out_l["A"]

        rank_loss = -F.logsigmoid(a_w - a_l).mean()
        loss = lambda_rank * rank_loss

        value_loss = torch.tensor(0.0, device=device)
        if use_value_loss and "success" in batch:
            success = batch["success"].to(device)
            mask = torch.isfinite(success)
            if mask.any():
                # V is same for w/l because only depends on context; take from out_w
                v = out_w["V"]
                value_loss = F.mse_loss(v[mask], success[mask])
                loss = loss + lambda_value * value_loss

        losses.append(loss.detach().item())
        rank_losses.append(rank_loss.detach().item())
        value_losses.append(value_loss.detach().item())

        correct += int(((a_w - a_l) > 0).sum().item())
        total += int(a_w.numel())

    return {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "rank_loss": float(np.mean(rank_losses)) if rank_losses else math.nan,
        "value_loss": float(np.mean(value_losses)) if value_losses else math.nan,
        "pair_acc": float(correct / max(total, 1)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=str, required=True, help="Path to parquet file.")
    p.add_argument("--context_key", type=str, default="context")
    p.add_argument("--action_w_key", type=str, default="action_w")
    p.add_argument("--action_l_key", type=str, default="action_l")
    p.add_argument("--success_key", type=str, default=None, help="Optional success field (0/1).")

    p.add_argument("--context_dim", type=int, default=None, help="If None, infer from data.")
    p.add_argument("--T", type=int, default=None, help="Action chunk length. Required if actions are flattened.")
    p.add_argument("--action_dim", type=int, default=None, help="Action dimension. Required if actions are flattened.")

    p.add_argument("--action_encoder", type=str, default="mlp_pool", choices=["mlp_pool", "transformer"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--lambda_rank", type=float, default=1.0)
    p.add_argument("--lambda_value", type=float, default=0.5)
    p.add_argument("--no_value_loss", action="store_true", help="Disable value loss even if success exists.")

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
    if args.context_dim is not None and int(args.context_dim) != int(ds.context_dim):
        raise ValueError(f"context_dim mismatch: expected {args.context_dim}, got {ds.context_dim}")

    # Split train/val
    n_val = int(round(len(ds) * float(args.val_frac)))
    n_val = min(max(n_val, 0), len(ds) - 1) if len(ds) > 1 else 0
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=skill_preference_collate_fn,
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=skill_preference_collate_fn,
            drop_last=False,
        )
        if n_val > 0
        else None
    )

    cfg = AdvantageJudgeConfig(
        context_dim=ds.context_dim,
        action_dim=ds.action_dim,
        action_encoder=args.action_encoder,  # type: ignore[arg-type]
    )
    model = AdvantageJudge(cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    use_value_loss = (not args.no_value_loss) and (args.success_key is not None)

    # Save training config for reproducibility
    (save_dir / "train_config.json").write_text(
        json.dumps(
            {
                "parquet": args.parquet,
                "keys": {
                    "context": args.context_key,
                    "action_w": args.action_w_key,
                    "action_l": args.action_l_key,
                    "success": args.success_key,
                },
                "dataset": {
                    "context_dim": ds.context_dim,
                    "fixed_T": getattr(ds, "fixed_T", None),
                    "action_dim": getattr(ds, "action_dim", None),
                    "n": len(ds),
                },
                "model": asdict(cfg),
                "train": {
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "lambda_rank": args.lambda_rank,
                    "lambda_value": args.lambda_value,
                    "use_value_loss": use_value_loss,
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
            context = batch["context"].to(device, non_blocking=True)
            aw = batch["action_w"].to(device, non_blocking=True)
            al = batch["action_l"].to(device, non_blocking=True)

            out_w = model(context, aw)
            out_l = model(context, al)
            a_w = out_w["A"]
            a_l = out_l["A"]

            rank_loss = -F.logsigmoid(a_w - a_l).mean()
            loss = float(args.lambda_rank) * rank_loss

            value_loss = torch.tensor(0.0, device=device)
            if use_value_loss and "success" in batch:
                success = batch["success"].to(device, non_blocking=True)
                mask = torch.isfinite(success)
                if mask.any():
                    v = out_w["V"]  # [B]
                    value_loss = F.mse_loss(v[mask], success[mask])
                    loss = loss + float(args.lambda_value) * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            pbar.set_postfix(
                {
                    "loss": f"{loss.detach().item():.4f}",
                    "rank": f"{rank_loss.detach().item():.4f}",
                    "v": f"{value_loss.detach().item():.4f}",
                }
            )

        # Validation
        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                lambda_rank=float(args.lambda_rank),
                lambda_value=float(args.lambda_value),
                use_value_loss=use_value_loss,
            )
            print(
                f"[val] epoch={epoch} loss={val_metrics['loss']:.4f} "
                f"rank={val_metrics['rank_loss']:.4f} value={val_metrics['value_loss']:.4f} "
                f"pair_acc={val_metrics['pair_acc']*100:.1f}%"
            )

        # Save checkpoint
        if epoch % int(args.save_every) == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": asdict(cfg),
                "dataset": {
                    "context_dim": ds.context_dim,
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

