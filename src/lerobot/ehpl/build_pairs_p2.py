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

"""Build EHPL same-start candidates (P2) from a lerobot dataset (v2.1 or v3.0).

Reads episode data, segments trajectories at skill boundaries
(gripper events + change-point detection), then constructs preference pairs
by comparing expert segments against perturbed candidates.

Paper-aligned pipeline:
  same-start generate K candidates -> train judge from episode success ->
  score candidates -> select winner/loser + margin filter.

This script implements only the first step: **candidate generation**.
Use `train_judge.py` and `select_pairs_from_candidates.py` for the rest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class P2Config:
    h_seg: int = 32
    act_dim: int = 7
    seed: int = 0
    max_segment_starts_per_episode: int = 24
    min_segment_len: int = 12
    change_point_c: float = 4.0
    gripper_event_threshold: float = 0.01
    noise_sigma_arm: float = 0.03
    noise_sigma_rot: float = 0.02
    gripper_jitter_prob: float = 0.15
    n_candidates: int = 6


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation."""
    med = np.median(x)
    return float(np.median(np.abs(x - med))) + 1e-08


def _load_episode_from_df(df: pd.DataFrame) -> dict[str, Any]:
    """Convert a single-episode DataFrame into state, action, and metadata arrays."""
    state = np.stack([np.asarray(v, dtype=np.float32) for v in df["observation.state"].to_list()], axis=0)
    action = np.stack([np.asarray(v, dtype=np.float32) for v in df["action"].to_list()], axis=0)

    # Validate shapes
    if state.ndim != 2 or state.shape[1] != 8:
        raise ValueError(f"Unexpected state shape {state.shape}")
    if action.ndim != 2 or action.shape[1] != 7:
        raise ValueError(f"Unexpected action shape {action.shape}")

    out = {
        "state": state,
        "action": action,
        "timestamp": df["timestamp"].to_numpy(copy=False).astype(np.float32),
        "frame_index": df["frame_index"].to_numpy(copy=False).astype(np.int64),
        "episode_index": int(df["episode_index"].iloc[0]),
        "index": df["index"].to_numpy(copy=False).astype(np.int64),
        "task_index": int(df["task_index"].iloc[0]),
    }
    return out


def _load_all_episodes(dataset_root: str, episode_start: int, episode_end: int) -> list[dict[str, Any]]:
    """Load episodes from a lerobot dataset (v2.1 or v3.0)."""
    root = Path(dataset_root)
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)
    else:
        info = {}

    chunk_dir = root / "data" / "chunk-000"
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Expected {chunk_dir} to exist")

    parquet_files = sorted(chunk_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {chunk_dir}")

    # Detect format: v3.0 uses file-*.parquet, v2.1 uses episode_*.parquet
    first_name = parquet_files[0].name
    is_v3 = first_name.startswith("file-")

    cols = ("observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index")

    episodes: list[dict[str, Any]] = []

    if is_v3:
        # v3.0: read all file-*.parquet, split by episode_index
        print(f"Detected v3.0 layout, reading {len(parquet_files)} files...")
        all_dfs = []
        for pf in parquet_files:
            df = pd.read_parquet(str(pf), engine="pyarrow", columns=list(cols))
            all_dfs.append(df)
        big_df = pd.concat(all_dfs, ignore_index=True)

        for ep_idx in sorted(big_df["episode_index"].unique()):
            ep_idx = int(ep_idx)
            if ep_idx < episode_start or ep_idx >= episode_end:
                continue
            ep_df = big_df[big_df["episode_index"] == ep_idx].reset_index(drop=True)
            episodes.append(_load_episode_from_df(ep_df))
    else:
        # v2.1: each file is one episode
        print(f"Detected v2.1 layout, reading {len(parquet_files)} episode files...")
        for p in parquet_files[episode_start:episode_end]:
            df = pd.read_parquet(str(p), engine="pyarrow", columns=list(cols))
            episodes.append(_load_episode_from_df(df))

    return episodes


def _compute_boundaries(state: np.ndarray, action: np.ndarray, cfg: P2Config) -> np.ndarray:
    """Return sorted unique boundary indices in [0, T). Always includes 0."""
    T = int(state.shape[0])

    # Gripper events: detect large changes in gripper dimension (last dim of state)
    g = np.array(state[:, 6:8].mean(axis=1), dtype=np.float32)
    dg = np.abs(np.diff(g))
    event_idx = np.where(dg > cfg.gripper_event_threshold)[0]

    # Change-point detection on arm features
    feat = state[:, :6].astype(np.float32)
    d = np.linalg.norm(np.diff(feat, axis=0, prepend=feat[:1]), axis=-1)
    thr = float(cfg.change_point_c) * _mad(d)
    cp_idx = np.where(d > thr)[0]

    # Merge, always include 0
    b = np.unique(np.concatenate([np.array([0], dtype=np.int32), event_idx, cp_idx]))
    b = np.sort(b)

    # Filter out boundaries too close together
    kept = [b[0]]
    for idx in b[1:]:
        if idx - kept[-1] >= cfg.min_segment_len:
            kept.append(idx)

    return np.asarray(kept, dtype=np.int32)


def _slice_segment_actions(action: np.ndarray, t0: int, h_seg: int) -> tuple[np.ndarray, np.ndarray]:
    """Slice a segment of actions starting at t0, with padding and mask."""
    T = int(action.shape[0])
    t1 = min(t0 + h_seg, T)
    seg = np.zeros((h_seg, action.shape[1]), dtype=np.float32)
    mask = np.zeros(h_seg, dtype=np.uint8)
    seg[: t1 - t0] = action[t0:t1]
    mask[: t1 - t0] = 1
    return seg, mask


def _perturb_actions(
    u: np.ndarray, mask: np.ndarray, cfg: P2Config, rng: np.random.Generator
) -> np.ndarray:
    """Return a perturbed copy of u (shape [H,7])."""
    v = u.copy()
    H = v.shape[0]
    valid = mask.astype(bool)
    L = valid.sum()

    if L == 0:
        return v

    # Add noise to arm (first 3 dims) and rotation (dims 3:6)
    v[valid, :3] += rng.normal(0.0, size=(L, 3)).astype(np.float32) * cfg.noise_sigma_arm
    v[valid, 3:6] += rng.normal(0.0, size=(L, 3)).astype(np.float32) * cfg.noise_sigma_rot

    # Gripper jitter
    if rng.random() < cfg.gripper_jitter_prob:
        start = int(rng.integers(0, max(L, 1)))
        end = min(start + int(rng.integers(1, max(4, 1))), H)
        v[start:end, 6] = np.clip(1.0 - v[start:end, 6], -1.0, 1.0)

    return v


def _flatten_actions(u: np.ndarray) -> list[float]:
    """Flatten action array to a list of floats."""
    return u.astype(np.float32).reshape(-1).tolist()


def _maybe_append_candidate_rows(
    *,
    rows: list[dict[str, Any]],
    candidates: list[np.ndarray],
    mask: np.ndarray,
    cfg: P2Config,
    episode_index: int,
    frame_index: int,
    task_index: int,
    segment_start_index_abs: int,
) -> None:
    for i, u in enumerate(candidates):
        rows.append(
            {
                "segment_start_index": int(segment_start_index_abs),
                "episode_index": int(episode_index),
                "frame_index": int(frame_index),
                "task_index": int(task_index),
                "h_seg": int(cfg.h_seg),
                "act_dim": int(cfg.act_dim),
                "candidate_index": int(i),
                "action": _flatten_actions(u),
                "action_mask": mask.astype(np.uint8).tolist(),
            }
        )


def build_pairs_for_dataset(
    dataset_root: str,
    out_parquet: str,
    cfg: P2Config,
    episode_start: int = 0,
    episode_end_exclusive: int | None = None,
) -> None:
    """Build same-start candidates from a lerobot dataset (v2.1 or v3.0)."""
    rng = np.random.default_rng(cfg.seed)

    # Determine episode range
    info_path = Path(dataset_root) / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)
        total_eps = info.get("total_episodes", 9999)
    else:
        total_eps = 9999
    ep_end = min(total_eps, episode_end_exclusive) if episode_end_exclusive is not None else total_eps

    episodes = _load_all_episodes(dataset_root, episode_start, ep_end)
    print(f"Loaded {len(episodes)} episodes (range [{episode_start}, {ep_end}))")

    rows = []
    for ep in episodes:
        action = ep["action"]
        idx_abs = int(ep["index"][0])
        T = int(action.shape[0])

        boundaries = _compute_boundaries(ep["state"], action, cfg)

        # Select segment starts (up to max_segment_starts_per_episode)
        if boundaries.size > cfg.max_segment_starts_per_episode:
            sel = rng.choice(boundaries.tolist(), size=cfg.max_segment_starts_per_episode, replace=False)
        else:
            sel = boundaries.tolist()

        for t0 in sel:
            t0 = int(t0)
            u_expert, mask = _slice_segment_actions(action, t0, cfg.h_seg)

            # Generate candidates
            candidates = [u_expert]
            for _ in range(max(cfg.n_candidates - 1, 1)):
                candidates.append(_perturb_actions(u_expert, mask, cfg, rng))

            segment_start_index_abs = idx_abs + t0
            _maybe_append_candidate_rows(
                rows=rows,
                candidates=candidates,
                mask=mask,
                cfg=cfg,
                episode_index=int(ep["episode_index"]),
                frame_index=int(ep["frame_index"][t0]),
                task_index=int(ep["task_index"]),
                segment_start_index_abs=int(segment_start_index_abs),
            )

    if not rows:
        raise RuntimeError("No preference rows produced. Check thresholds or dataset paths.")

    # Write output
    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(out_path))

    # Write metadata
    meta = {
        "dataset_root": dataset_root,
        "num_rows": len(rows),
        "config": cfg.__dict__,
        "episode_range": (episode_start, ep_end),
    }
    with open(out_path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(rows)} pairs to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--out_parquet", type=str, required=True)
    ap.add_argument("--h_seg", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode_start", type=int, default=0)
    ap.add_argument("--episode_end", type=int, default=None)
    ap.add_argument("--max_segment_starts_per_episode", type=int, default=24)
    ap.add_argument("--n_candidates", type=int, default=6)
    args = ap.parse_args()

    cfg = P2Config(
        h_seg=args.h_seg,
        seed=args.seed,
        max_segment_starts_per_episode=args.max_segment_starts_per_episode,
        n_candidates=args.n_candidates,
    )
    build_pairs_for_dataset(
        dataset_root=args.dataset_root,
        out_parquet=args.out_parquet,
        cfg=cfg,
        episode_start=args.episode_start,
        episode_end_exclusive=args.episode_end,
    )


if __name__ == "__main__":
    main()
