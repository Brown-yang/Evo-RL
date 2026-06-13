#!/usr/bin/env python
"""
EHPL Candidates P2 Dataset.

This mirrors `EhplPairsP2Dataset` but for *same-start candidate chunks*.

Each item returns the observation at the segment start (single-frame) plus:
  - action: candidate action chunk [H, act_dim]
  - action_mask: uint8 mask [H]
  - candidate_index: int
  - (optional) success: float in {0,1} derived from episode-level `episode_success`

Expected columns in candidates parquet:
  - segment_start_index, episode_index, frame_index, task_index
  - h_seg, act_dim
  - candidate_index
  - action (flattened H*act_dim list)
  - action_mask (length H)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from lerobot.ehpl.pairs_p2_dataset import EhplPairsP2Dataset


def _load_episode_success_map(dataset_root: str) -> dict[int, float]:
    """
    Load an episode_index -> success(0/1) map from `meta/episodes/*.parquet` if present.

    The recorder writes `episode_success` as 'success'/'failure' (or None).
    """
    root = Path(dataset_root)
    ep_dir = root / "meta" / "episodes"
    if not ep_dir.exists():
        return {}

    mapping: dict[int, float] = {}
    for pf in sorted(ep_dir.rglob("*.parquet")):
        try:
            df = pd.read_parquet(str(pf), engine="pyarrow", columns=["episode_index", "episode_success"])
        except Exception:
            continue
        for _, row in df.iterrows():
            ep = int(row["episode_index"])
            lab = row.get("episode_success")
            if isinstance(lab, str):
                l = lab.strip().lower()
                if l == "success":
                    mapping[ep] = 1.0
                elif l == "failure":
                    mapping[ep] = 0.0
    return mapping


class EhplCandidatesP2Dataset(Dataset):
    def __init__(
        self,
        *,
        dataset_root: str,
        candidates_parquet: str,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.dataset_root = str(dataset_root)
        self.candidates_parquet = str(candidates_parquet)
        self.device = device

        info_path = Path(dataset_root) / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError("Expected dataset meta at " + str(info_path))
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        self.codebase_version = self.info.get("codebase_version", "v2.1")

        self._pairs_loader = EhplPairsP2Dataset(
            dataset_root=dataset_root,
            pairs_parquet=_dummy_pairs_parquet_for_obs_only(dataset_root),
            device=device,
        )
        # Replace the pairs table with candidates table; keep obs-loading helpers from EhplPairsP2Dataset.
        self._is_v3 = self._pairs_loader._is_v3  # noqa: SLF001

        self.candidates = pd.read_parquet(self.candidates_parquet, engine="pyarrow")
        required = (
            "segment_start_index",
            "episode_index",
            "frame_index",
            "task_index",
            "h_seg",
            "act_dim",
            "candidate_index",
            "action",
            "action_mask",
        )
        missing = [c for c in required if c not in self.candidates.columns]
        if missing:
            raise ValueError("candidates parquet missing columns: " + str(missing))

        self.h_seg = int(self.candidates.iloc[0]["h_seg"])
        self.act_dim = int(self.candidates.iloc[0]["act_dim"])
        if not (self.candidates["h_seg"].astype(int) == self.h_seg).all():
            raise ValueError("candidates parquet has non-constant h_seg; keep fixed per run.")
        if not (self.candidates["act_dim"].astype(int) == self.act_dim).all():
            raise ValueError("candidates parquet has non-constant act_dim; keep fixed per run.")

        self._ep_success = _load_episode_success_map(dataset_root)

    def __len__(self) -> int:
        return int(len(self.candidates))

    def _reshape_flat(self, flat: Any, name: str) -> torch.Tensor:
        arr = np.asarray(flat, dtype=np.float32)
        exp = self.h_seg * self.act_dim
        if arr.size != exp:
            raise ValueError(f"{name} expected length {exp} (=H*D), got {arr.size}")
        return torch.from_numpy(arr).reshape(self.h_seg, self.act_dim)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.candidates.iloc[i]
        episode_index = int(row["episode_index"])
        frame_index = int(row["frame_index"])
        task_index = int(row["task_index"])
        segment_start_index = int(row["segment_start_index"])

        if self._is_v3:
            item = self._pairs_loader._load_obs_v3(episode_index, frame_index, task_index, segment_start_index)  # noqa: SLF001
        else:
            item = self._pairs_loader._load_obs_v21(episode_index, frame_index, task_index, segment_start_index)  # noqa: SLF001

        # task instruction string if available
        if self._pairs_loader.task_index_to_instruction is not None:  # noqa: SLF001
            item["task"] = self._pairs_loader.task_index_to_instruction.get(task_index, f"Task {task_index}")  # noqa: SLF001
        else:
            item["task"] = f"Task {task_index}"

        action = self._reshape_flat(row["action"], name="action")
        action_mask = torch.from_numpy(np.asarray(row["action_mask"], dtype=np.uint8))
        if action_mask.numel() != self.h_seg:
            raise ValueError(f"action_mask expected length {self.h_seg}, got {action_mask.numel()}")

        item["action"] = action
        item["action_mask"] = action_mask
        item["candidate_index"] = torch.tensor(int(row["candidate_index"]), dtype=torch.int64)

        if episode_index in self._ep_success:
            item["success"] = torch.tensor(float(self._ep_success[episode_index]), dtype=torch.float32)

        if self.device is not None:
            for k, v in list(item.items()):
                if torch.is_tensor(v):
                    item[k] = v.to(self.device)
        return item


def _dummy_pairs_parquet_for_obs_only(dataset_root: str) -> str:
    """
    EhplPairsP2Dataset requires a pairs parquet at init time. We generate a tiny
    in-memory-like parquet on disk under dataset_root/.ehpl_tmp the first time.
    """
    tmp_dir = Path(dataset_root) / ".ehpl_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / "dummy_pairs_for_obs_only.parquet"
    if path.exists():
        return str(path)

    # Minimal 1-row table; values will not be used beyond init.
    import pyarrow as pa
    import pyarrow.parquet as pq

    dummy = pa.Table.from_pylist(
        [
            {
                "segment_start_index": 0,
                "episode_index": 0,
                "frame_index": 0,
                "task_index": 0,
                "h_seg": 1,
                "act_dim": 7,
                "action_w": [0.0] * 7,
                "action_l": [0.0] * 7,
                "action_mask": [1],
            }
        ]
    )
    pq.write_table(dummy, str(path))
    (path.with_suffix(".meta.json")).write_text(json.dumps({"note": "auto-generated"}, indent=2) + "\n")
    return str(path)


__all__ = ["EhplCandidatesP2Dataset"]

