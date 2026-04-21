#!/usr/bin/env python

"""Dataset/processing utilities for EHPL model-learning components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

@dataclass(frozen=True)
class SkillPreferenceParquetConfig:
    parquet_path: str
    context_key: str = "context"
    action_w_key: str = "action_w"
    action_l_key: str = "action_l"
    success_key: str | None = None

    action_dim: int | None = None
    fixed_T: int | None = None
    max_rows: int | None = None


def _to_numpy_float32(x: Any) -> np.ndarray:
    """Convert parquet cell (pyarrow scalar/list/np) into float32 numpy array."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if hasattr(x, "as_py"):  # pyarrow scalar
        x = x.as_py()
    return np.asarray(x, dtype=np.float32)


def _reshape_action(action_cell: Any, *, action_dim: int | None, fixed_T: int | None) -> np.ndarray:
    """
    Normalize an action cell into shape [T, action_dim].

    Supports:
      - nested list: [[...], ...] -> ndarray [T, action_dim]
      - flat list:   [...]        -> reshape via action_dim (+ optional fixed_T)
    """
    arr = _to_numpy_float32(action_cell)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 1:
        raise ValueError(f"Unsupported action cell ndim={arr.ndim}, shape={arr.shape}")

    if action_dim is None:
        raise ValueError("Flattened action found; please provide action_dim in config.")
    d = int(action_dim)

    if fixed_T is not None:
        t = int(fixed_T)
        expected = t * d
        if arr.shape[0] != expected:
            raise ValueError(f"Expected flattened action length {expected}, got {arr.shape[0]}")
        return arr.reshape(t, d)

    if arr.shape[0] % d != 0:
        raise ValueError(
            f"Flattened action length {arr.shape[0]} is not divisible by action_dim={d}. Provide fixed_T."
        )
    t = int(arr.shape[0] // d)
    return arr.reshape(t, d)


class SkillPreferenceDataset(Dataset):
    """Load skill-level preference samples from a parquet file."""

    def __init__(self, parquet_path: str | Path, **kwargs) -> None:
        super().__init__()
        cfg = SkillPreferenceParquetConfig(parquet_path=str(parquet_path), **kwargs)
        self.cfg = cfg

        import pyarrow.parquet as pq

        cols = [cfg.context_key, cfg.action_w_key, cfg.action_l_key]
        if cfg.success_key is not None:
            cols.append(cfg.success_key)

        table = pq.read_table(cfg.parquet_path, columns=cols)
        if cfg.max_rows is not None:
            table = table.slice(0, int(cfg.max_rows))
        if table.num_rows == 0:
            raise ValueError(f"Parquet has 0 rows: {cfg.parquet_path}")

        self._table = table
        self._n = table.num_rows

        first_context = _to_numpy_float32(table[cfg.context_key][0])
        if first_context.ndim != 1:
            raise ValueError(f"context must be 1D, got {first_context.shape}")
        self.context_dim = int(first_context.shape[0])

        first_aw = _to_numpy_float32(table[cfg.action_w_key][0])
        if first_aw.ndim == 2:
            self.action_dim = int(first_aw.shape[1]) if cfg.action_dim is None else int(cfg.action_dim)
            self.fixed_T = int(first_aw.shape[0]) if cfg.fixed_T is None else int(cfg.fixed_T)
        else:
            self.action_dim = cfg.action_dim
            self.fixed_T = cfg.fixed_T

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        row = {name: self._table[name][idx] for name in self._table.column_names}

        context = _to_numpy_float32(row[self.cfg.context_key])
        aw = _reshape_action(row[self.cfg.action_w_key], action_dim=self.action_dim, fixed_T=self.fixed_T)
        al = _reshape_action(row[self.cfg.action_l_key], action_dim=self.action_dim, fixed_T=self.fixed_T)

        out: dict[str, Tensor] = {
            "context": torch.from_numpy(context).to(dtype=torch.float32),
            "action_w": torch.from_numpy(aw).to(dtype=torch.float32),
            "action_l": torch.from_numpy(al).to(dtype=torch.float32),
        }

        if self.cfg.success_key is not None and self.cfg.success_key in row:
            s = row[self.cfg.success_key]
            if hasattr(s, "as_py"):
                s = s.as_py()
            out["success"] = torch.tensor(float("nan") if s is None else float(s), dtype=torch.float32)

        return out


def skill_preference_collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Pad variable-length action chunks to max T in batch and stack."""
    if not batch:
        raise ValueError("Empty batch")

    context = torch.stack([b["context"] for b in batch], dim=0).to(dtype=torch.float32)
    aw_list = [b["action_w"].to(dtype=torch.float32) for b in batch]
    al_list = [b["action_l"].to(dtype=torch.float32) for b in batch]

    t_max = max(int(x.shape[0]) for x in aw_list + al_list)
    action_dim = int(aw_list[0].shape[1])
    for x in aw_list + al_list:
        if x.ndim != 2:
            raise ValueError(f"action must be [T,A], got {tuple(x.shape)}")
        if int(x.shape[1]) != action_dim:
            raise ValueError("action_dim mismatch in batch")

    def _pad(seq: Tensor) -> tuple[Tensor, Tensor]:
        t = int(seq.shape[0])
        if t == t_max:
            return seq, torch.ones(t_max, dtype=torch.bool)
        pad = torch.zeros(t_max - t, action_dim, dtype=seq.dtype)
        padded = torch.cat([seq, pad], dim=0)
        mask = torch.zeros(t_max, dtype=torch.bool)
        mask[:t] = True
        return padded, mask

    aw_pad, aw_mask = zip(*[_pad(x) for x in aw_list], strict=False)
    al_pad, al_mask = zip(*[_pad(x) for x in al_list], strict=False)
    action_mask = torch.stack([m1 | m2 for m1, m2 in zip(aw_mask, al_mask, strict=False)], dim=0)

    out: dict[str, Tensor] = {
        "context": context,
        "action_w": torch.stack(list(aw_pad), dim=0),
        "action_l": torch.stack(list(al_pad), dim=0),
        "action_mask": action_mask,
    }

    if "success" in batch[0]:
        out["success"] = torch.stack([b["success"] for b in batch], dim=0).to(dtype=torch.float32)

    return out

