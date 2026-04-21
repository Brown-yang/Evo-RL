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

"""EHPL Pairs P2 Dataset — supports both v2.1 and v3.0 lerobot layouts."""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from lerobot.utils.constants import ACTION

logger = logging.getLogger(__name__)


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
    """Preference pairs dataset (P2 layout) supporting v2.1 and v3.0 lerobot datasets.

    v2.1 layout (parquet-per-episode + mp4-per-episode):
    - parquet: ``data/chunk-000/episode_{episode_index:06d}.parquet``
    - videos:  ``videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4``

    v3.0 layout (shared parquet files + inline images):
    - parquet: ``data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet``
    - images stored as ``{'bytes': ..., 'path': ...}`` dicts inside the parquet
    - ``video_path`` in info.json is ``null``

    Each item returns the observation at the segment start (single-frame):
    - ``observation.images.image``, ``observation.images.wrist_image``: float32 CHW in [0, 1]
    - ``observation.state``: float32 (8,)
    - ``index``, ``episode_index``, ``frame_index``, ``task_index``
    - ``task`` (optional): instruction string if ``task_index_to_instruction`` is provided
    plus EHPL fields:
    - ``action_w``, ``action_l``: preferred/rejected action sequences [H, act_dim]
    - ``action_mask``: uint8 mask [H]
    """

    def __init__(
        self,
        dataset_root: str,
        pairs_parquet: str,
        task_index_to_instruction: dict[int, str] | None = None,
        device: str | None = None,
        min_pair_margin: float | None = None,
    ) -> None:
        super().__init__()
        self.dataset_root = str(dataset_root)
        self.pairs_parquet = str(pairs_parquet)
        self.device = device
        self.min_pair_margin = min_pair_margin

        info_path = Path(dataset_root) / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError("Expected dataset meta at " + str(info_path))

        self.info = json_load(info_path)
        self.fps = int(self.info.get("fps", 20))
        self.codebase_version = self.info.get("codebase_version", "v2.1")

        # Load task instructions from tasks.parquet if available
        if task_index_to_instruction is None:
            tasks_path = Path(dataset_root) / "meta" / "tasks.parquet"
            if tasks_path.exists():
                tasks_df = pd.read_parquet(str(tasks_path), engine="pyarrow")
                # tasks.parquet has task text as index and task_index as column
                self.task_index_to_instruction = {}
                for task_text, row in tasks_df.iterrows():
                    self.task_index_to_instruction[int(row["task_index"])] = str(task_text)
                logger.info(f"Auto-loaded {len(self.task_index_to_instruction)} tasks from {tasks_path}")
            else:
                self.task_index_to_instruction = None
        else:
            self.task_index_to_instruction = task_index_to_instruction

        # Detect layout version
        self._is_v3 = self.codebase_version.startswith("v3")
        if self._is_v3:
            logger.info("Detected v3.0 dataset layout (inline images in parquet)")
            self._init_v3()
        else:
            logger.info("Detected v2.1 dataset layout (per-episode parquet + mp4)")
            self._init_v21()

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

        if self.min_pair_margin is not None:
            if "margin" not in self.pairs.columns:
                raise ValueError(
                    "min_pair_margin is set but pairs parquet has no 'margin' column. "
                    "Run `python -m lerobot.ehpl.infer_judge ...` to add score_w/score_l/margin first."
                )
            n0 = int(len(self.pairs))
            self.pairs = self.pairs[self.pairs["margin"].astype(float) >= float(self.min_pair_margin)].reset_index(
                drop=True
            )
            logger.info(
                "Filtered pairs by margin>=%s: %d -> %d rows",
                self.min_pair_margin,
                n0,
                int(len(self.pairs)),
            )

    # ------------------------------------------------------------------ #
    # v2.1 init — per-episode parquet + mp4 videos
    # ------------------------------------------------------------------ #
    def _init_v21(self) -> None:
        self.data_path_tpl = str(
            Path(self.dataset_root) / "data" / "chunk-000" / "episode_{episode_index:06d}.parquet"
        )
        self.video_path_tpl = str(
            Path(self.dataset_root) / "videos" / "chunk-000" / "{video_key}" / "episode_{episode_index:06d}.mp4"
        )

    # ------------------------------------------------------------------ #
    # v3.0 init — shared file-*.parquet with inline images
    # ------------------------------------------------------------------ #
    def _init_v3(self) -> None:
        """Build an episode→file mapping by scanning parquet files."""
        data_dir = Path(self.dataset_root) / "data" / "chunk-000"
        self._ep_to_file: dict[int, Path] = {}
        for pf in sorted(data_dir.glob("file-*.parquet")):
            df_index = pd.read_parquet(pf, engine="pyarrow", columns=["episode_index"])
            for ep in df_index["episode_index"].unique():
                self._ep_to_file[int(ep)] = pf
        logger.info(f"v3.0 episode→file mapping: {len(self._ep_to_file)} episodes across {len(set(self._ep_to_file.values()))} files")

        # Detect image keys from info.json features
        self._image_keys = [
            k for k, v in self.info.get("features", {}).items()
            if v.get("dtype") == "image"
        ]
        logger.info(f"v3.0 image keys: {self._image_keys}")

    # ------------------------------------------------------------------ #
    # __len__ / reshape helper
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return int(len(self.pairs))

    def _reshape_flat(self, flat: Any, name: str) -> torch.Tensor:
        arr = np.asarray(flat, dtype=np.float32)
        exp = self.h_seg * self.act_dim
        if arr.size != exp:
            raise ValueError(f"{name} expected length {exp} (=H*D), got {arr.size}")
        return torch.from_numpy(arr).reshape(self.h_seg, self.act_dim)

    # ------------------------------------------------------------------ #
    # __getitem__
    # ------------------------------------------------------------------ #
    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.pairs.iloc[i]
        episode_index = int(row["episode_index"])
        frame_index = int(row["frame_index"])
        task_index = int(row["task_index"])
        segment_start_index = int(row["segment_start_index"])

        if self._is_v3:
            item = self._load_obs_v3(episode_index, frame_index, task_index, segment_start_index)
        else:
            item = self._load_obs_v21(episode_index, frame_index, task_index, segment_start_index)

        # Task instruction (required by pi0.5 preprocessor)
        if self.task_index_to_instruction is not None:
            item["task"] = self.task_index_to_instruction.get(task_index, f"Task {task_index}")
        else:
            item["task"] = f"Task {task_index}"

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

    # ------------------------------------------------------------------ #
    # v2.1 observation loading
    # ------------------------------------------------------------------ #
    def _load_obs_v21(
        self, episode_index: int, frame_index: int, task_index: int, segment_start_index: int
    ) -> dict[str, Any]:
        ep_parquet = self.data_path_tpl.format(episode_chunk=0, episode_index=episode_index)
        df = pd.read_parquet(
            ep_parquet,
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

        state = np.asarray(df.iloc[frame_index]["observation.state"], dtype=np.float32)

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

        return {
            "observation.state": torch.from_numpy(state),
            "observation.images.image": image,
            "observation.images.wrist_image": wrist,
            "index": torch.tensor(segment_start_index, dtype=torch.int64),
            "episode_index": torch.tensor(episode_index, dtype=torch.int64),
            "frame_index": torch.tensor(frame_index, dtype=torch.int64),
            "task_index": torch.tensor(task_index, dtype=torch.int64),
        }

    # ------------------------------------------------------------------ #
    # v3.0 observation loading
    # ------------------------------------------------------------------ #
    def _load_obs_v3(
        self, episode_index: int, frame_index: int, task_index: int, segment_start_index: int
    ) -> dict[str, Any]:
        pf = self._ep_to_file.get(episode_index)
        if pf is None:
            raise FileNotFoundError(
                f"No file found for episode_index={episode_index}. "
                f"Available: {sorted(self._ep_to_file.keys())}"
            )

        # Read the relevant columns
        cols = ["observation.state", "episode_index", "frame_index"] + self._image_keys
        df = pd.read_parquet(str(pf), engine="pyarrow", columns=cols)

        # Filter to the target episode and frame
        ep_df = df[df["episode_index"] == episode_index].reset_index(drop=True)
        if frame_index >= len(ep_df):
            raise IndexError(
                f"frame_index={frame_index} out of range for episode {episode_index} "
                f"(file has {len(ep_df)} frames for this episode)"
            )

        frame_row = ep_df.iloc[frame_index]

        state = np.array(frame_row["observation.state"], dtype=np.float32, copy=True)

        item: dict[str, Any] = {
            "observation.state": torch.from_numpy(state),
            "index": torch.tensor(segment_start_index, dtype=torch.int64),
            "episode_index": torch.tensor(episode_index, dtype=torch.int64),
            "frame_index": torch.tensor(frame_index, dtype=torch.int64),
            "task_index": torch.tensor(task_index, dtype=torch.int64),
        }

        # Decode inline images
        for img_key in self._image_keys:
            img_dict = frame_row[img_key]
            if isinstance(img_dict, dict) and "bytes" in img_dict:
                img_bytes = img_dict["bytes"]
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                arr = np.asarray(img, dtype=np.float32) / 255.0
                # HWC -> CHW
                item[img_key] = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            else:
                raise ValueError(
                    f"Expected image dict with 'bytes' key for {img_key}, "
                    f"got {type(img_dict)}"
                )

        return item


# ------------------------------------------------------------------ #
# Utilities
# ------------------------------------------------------------------ #

def json_load(path: str | Path) -> dict[str, Any]:
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
