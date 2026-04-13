#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EHPL Pairs P2 Dataset for v2.1 `*_lerobot` layouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from lerobot.utils.constants import ACTION


@dataclass(frozen=True)
class EhplP2Sample:
    """One EHPL preference sample at a segment start."""

    segment_start_index: int
    episode_index: int
    frame_index: int
    task_index: int
    action_w: torch.Tensor
    action_l: torch.Tensor
    action_mask: torch.Tensor


class EhplPairsP2Dataset(Dataset):
    """Preference pairs dataset (P2 layout) for v2.1 `*_lerobot` datasets (parquet-per-episode + mp4-per-episode).

    This dataset format is what we observed in `libero_10_no_noops_1.0.0_lerobot`:
    - parquet: `data/chunk-000/episode_{episode_index:06d}.parquet`
    - videos: `videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4`

    Each item returns the observation at the segment start (single-frame):
    - `observation.images.image`, `observation.images.wrist_image`: float32 CHW in [0, 1]
    - `observation.state`: float32 (8,)
    - `index`, `episode_index`, `frame_index`, `task_index`
    - `task` (optional): instruction string if `task_index_to_instruction` is provided
    plus EHPL fields:
    - `action_w`, `action_l`: preferred/rejected action sequences [H, act_dim]
    - `action_mask`: uint8 mask [H]
    """

    def __init__(
        self,
        dataset_root: str,
        pairs_parquet: str,
        task_index_to_instruction: dict[int, str] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.dataset_root = str(dataset_root)
        self.pairs_parquet = str(pairs_parquet)
        self.task_index_to_instruction = task_index_to_instruction
        self.device = device

        info_path = Path(dataset_root) / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError("Expected v2.1 dataset meta at " + str(info_path))

        self.info = json_load(info_path)
        self.fps = int(self.info.get("fps", 20))

        self.data_path_tpl = str(Path(dataset_root) / "data" / "chunk-000" / "episode_{episode_index:06d}.parquet")
        self.video_path_tpl = str(Path(dataset_root) / "videos" / "chunk-000" / "{video_key}" / "episode_{episode_index:06d}.mp4")

        # Load pairs parquet
        self.pairs = pd.read_parquet(pairs_parquet, engine="pyarrow")

        # Validate columns
        required = (
            "segment_start_index",
            "episode_index",
            "frame_index",
            "task_index",
            "h_seg",
            "act_dim",
            "action_w",
            "action_l",
            "action_mask",
        )
        missing = [c for c in required if c not in self.pairs.columns]
        if missing:
            raise ValueError("pairs parquet missing columns: " + str(missing))

        # Extract h_seg and act_dim (should be constant across all rows)
        self.h_seg = int(self.pairs.iloc[0]["h_seg"])
        self.act_dim = int(self.pairs.iloc[0]["act_dim"])

        if not (self.pairs["h_seg"].astype(int) == self.h_seg).all():
            raise ValueError("pairs parquet has non-constant h_seg; keep fixed per run.")
        if not (self.pairs["act_dim"].astype(int) == self.act_dim).all():
            raise ValueError("pairs parquet has non-constant act_dim; keep fixed per run.")

    def __len__(self) -> int:
        return int(len(self.pairs))

    def _reshape_flat(self, flat: Any, name: str) -> torch.Tensor:
        arr = np.asarray(flat, dtype=np.float32)
        exp = self.h_seg * self.act_dim
        if arr.size != exp:
            raise ValueError(f"{name} expected length {exp} (=H*D), got {arr.size}")
        return torch.from_numpy(arr).reshape(self.h_seg, self.act_dim)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.pairs.iloc[i]
        episode_index = int(row["episode_index"])
        frame_index = int(row["frame_index"])
        task_index = int(row["task_index"])
        segment_start_index = int(row["segment_start_index"])

        # Load episode parquet for observation data
        ep_parquet = Path(self.dataset_root) / self.data_path_tpl.format(
            episode_chunk=0, episode_index=episode_index
        )
        df = pd.read_parquet(
            str(ep_parquet),
            engine="pyarrow",
            columns=(
                "observation.state",
                "timestamp",
                "index",
                "episode_index",
                "frame_index",
                "task_index",
            ),
        )

        if frame_index >= len(df):
            raise IndexError(f"frame_index={frame_index} out of range for episode {episode_index}")

        # State
        state = np.asarray(df.iloc[frame_index]["observation.state"], dtype=np.float32)
        if state.shape != (8,):
            raise ValueError(
                f"Unexpected state shape {state.shape} at ep={episode_index} t={frame_index}"
            )

        # Decode images from video
        image = decode_mp4_frame(
            Path(self.video_path_tpl.format(
                episode_chunk=0,
                video_key="observation.images.image",
                episode_index=episode_index,
            )),
            frame_index,
        )
        wrist = decode_mp4_frame(
            Path(self.video_path_tpl.format(
                episode_chunk=0,
                video_key="observation.images.wrist_image",
                episode_index=episode_index,
            )),
            frame_index,
        )

        item = {
            "observation.state": torch.from_numpy(state),
            "observation.images.image": image,
            "observation.images.wrist_image": wrist,
            "index": torch.tensor(segment_start_index, dtype=torch.int64),
            "episode_index": torch.tensor(episode_index, dtype=torch.int64),
            "frame_index": torch.tensor(frame_index, dtype=torch.int64),
            "task_index": torch.tensor(task_index, dtype=torch.int64),
        }

        # Optional task instruction
        if self.task_index_to_instruction is not None:
            item["task"] = self.task_index_to_instruction.get(task_index, f"Task {task_index}")

        # EHPL action fields
        action_w = self._reshape_flat(row["action_w"], name="action_w")
        action_l = self._reshape_flat(row["action_l"], name="action_l")
        action_mask = torch.from_numpy(np.asarray(row["action_mask"], dtype=np.uint8))
        if action_mask.numel() != self.h_seg:
            raise ValueError(
                f"action_mask expected length {self.h_seg}, got {action_mask.numel()}"
            )

        item[ACTION] = action_w  # default action is the preferred one
        item["action_w"] = action_w
        item["action_l"] = action_l
        item["action_mask"] = action_mask

        # Move to device if specified
        if self.device is not None:
            for k, v in list(item.items()):
                if torch.is_tensor(v):
                    item[k] = v.to(self.device)

        return item


def json_load(path: str) -> dict[str, Any]:
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def decode_mp4_frame(path: Path, frame_index: int) -> torch.Tensor:
    """Decode one frame from an mp4 and return float32 CHW in [0, 1]."""
    if not path.exists():
        raise FileNotFoundError(str(path))

    import imageio.v2 as imageio
    reader = imageio.get_reader(str(path))
    frame = reader.get_data(frame_index)
    reader.close()

    # frame shape should be (H, W, 3)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Unexpected frame shape {frame.shape} in {path}")

    # Convert to float32 CHW in [0, 1]
    t = torch.from_numpy(frame).permute(2, 0, 1).contiguous().to(dtype=torch.float32) / 255.0
    return t
