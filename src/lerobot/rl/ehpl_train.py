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

"""EHPL (skill-level preference) training helpers for `lerobot-train`.

Pairs with v2.1 `*_lerobot` layouts (parquet-per-episode + mp4) and a P2 `pairs.parquet`
produced by `lerobot.ehpl.build_pairs_p2`.
"""

from __future__ import annotations

import json
import os
import sys
import types
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.utils.data import Dataset

from lerobot.ehpl.pairs_p2_dataset import EhplPairsP2Dataset
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION
from lerobot.utils.logging_utils import MetricsTracker
from lerobot.utils.utils import has_method


def load_stats_v21(stats_json_path: str) -> dict[str, dict[str, Any]]:
    with open(stats_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_ehpl_hf_env(
    *,
    hf_home: str | None,
    offline: bool,
) -> None:
    """Configure Hugging Face cache / offline mode before loading tokenizers (PI0/PI05)."""
    if hf_home:
        hf_home_p = Path(hf_home)
        is_hub_cache_dir = (hf_home_p / "models--google--paligemma-3b-pt-224").exists() or any(
            p.name.startswith("models--") for p in hf_home_p.glob("models--*")
        )
        if is_hub_cache_dir:
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home_p)
            os.environ["HF_HUB_CACHE"] = str(hf_home_p)
            os.environ["TRANSFORMERS_CACHE"] = str(hf_home_p)
            os.environ.setdefault("HF_HOME", str(hf_home_p.parent))
        else:
            os.environ["HF_HOME"] = str(hf_home_p)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home_p / "hub"))
            os.environ.setdefault("HF_HUB_CACHE", str(hf_home_p / "hub"))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home_p / "hub"))
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def monkeypatch_siglip_check() -> None:
    try:
        import transformers.models.siglip as siglip_pkg
    except Exception:
        return

    modname = "transformers.models.siglip.check"
    if modname in sys.modules and hasattr(siglip_pkg, "check"):
        return

    check_mod = types.ModuleType(modname)

    def _ok() -> bool:
        return True

    check_mod.check_whether_transformers_replace_is_installed_correctly = _ok  # type: ignore[attr-defined]
    sys.modules[modname] = check_mod
    setattr(siglip_pkg, "check", check_mod)


def monkeypatch_gemma_rmsnorm_cond() -> None:
    try:
        from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
    except Exception:
        return

    code = getattr(GemmaRMSNorm.forward, "__code__", None)
    if code is not None and "cond" in code.co_varnames:
        return

    orig_forward = GemmaRMSNorm.forward

    def forward(self, hidden_states, cond=None):  # noqa: ANN001
        return orig_forward(self, hidden_states), None

    GemmaRMSNorm.forward = forward  # type: ignore[assignment]


def monkeypatch_gemma_gated_residual() -> None:
    """Provide OpenPI internal helper `_gated_residual` if missing.

    The OpenPI PI05 port calls `transformers.models.gemma.modeling_gemma._gated_residual`.
    Some `transformers` builds (non-OpenPI-patched) don't ship it.
    """
    try:
        from transformers.models.gemma import modeling_gemma
    except Exception:
        return

    if hasattr(modeling_gemma, "_gated_residual"):
        return

    def _gated_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
        # Conservative fallback: if gate is None, use standard residual.
        if gate is None:
            return x + y
        # OpenPI gates can be broadcastable; sigmoid keeps values in [0,1].
        return x + torch.sigmoid(gate) * y

    setattr(modeling_gemma, "_gated_residual", _gated_residual)

def adapt_ehpl_image_keys_inplace(batch: dict[str, Any], expected_image_keys: list[str]) -> None:
    if not expected_image_keys:
        return
    src_base = "observation.images.image"
    src_wrist = "observation.images.wrist_image"
    if src_base not in batch and src_wrist not in batch:
        return
    for k in expected_image_keys:
        if k in batch:
            continue
        lk = k.lower()
        if "base" in lk or "agent" in lk or "front" in lk or "ego" in lk:
            batch[k] = batch.get(src_base, batch.get(src_wrist))
        elif "wrist" in lk or "hand" in lk or "eye_in_hand" in lk:
            batch[k] = batch.get(src_wrist, batch.get(src_base))
        else:
            batch[k] = batch.get(src_base, batch.get(src_wrist))


def _load_task_index_map(path: str | None) -> dict[int, str] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[int, str] = {}
    for k, v in raw.items():
        out[int(k)] = str(v)
    return out


class EhplPreferenceTrainDataset(Dataset):
    """Torch dataset + `.meta.stats` for `make_pre_post_processors` (v2.1 stats.json)."""

    def __init__(
        self,
        *,
        dataset_root: str,
        pairs_parquet: str,
        stats_json: str,
        task_index_to_instruction_json: str | None = None,
    ) -> None:
        super().__init__()
        self._inner = EhplPairsP2Dataset(
            dataset_root=dataset_root,
            pairs_parquet=pairs_parquet,
            task_index_to_instruction=_load_task_index_map(task_index_to_instruction_json),
        )
        # `make_policy` needs `dataset.meta.features` (v2.1 `info.json`); stats come from `meta/stats.json`.
        self.meta = SimpleNamespace(
            stats=load_stats_v21(stats_json),
            features=self._inner.info["features"],
        )

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._inner[index]

    @property
    def num_frames(self) -> int:
        return len(self._inner)

    @property
    def num_episodes(self) -> int:
        return int(self._inner.pairs["episode_index"].nunique())


def ehpl_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


def update_policy_ehpl(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    raw_batch: dict[str, Any],
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    expected_image_keys: list[str],
    beta: float,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict[str, Any]]:
    """Preference ranking step: softplus(beta * (loss_w - loss_l))."""
    from contextlib import nullcontext

    start_time = time.perf_counter()
    policy.train()

    def _move_to_device_inplace(batch: dict[str, Any]) -> dict[str, Any]:
        # Processor pipelines sometimes leave some tensors on CPU depending on step configs.
        # EHPL must guarantee everything is on accelerator.device for PI05 vision tower (bf16 kernels).
        dev = accelerator.device
        for k, v in list(batch.items()):
            if torch.is_tensor(v) and v.device != dev:
                batch[k] = v.to(dev, non_blocking=True)
        return batch

    # Determine the model's chunk_size so we can pad EHPL actions to match.
    _unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    chunk_size = getattr(_unwrapped.config, "chunk_size", None)

    def _one_side(actions_key: str) -> torch.Tensor:
        b = dict(raw_batch)
        actions = raw_batch[actions_key]
        # Pad action time dimension from h_seg to chunk_size if needed
        if chunk_size is not None and actions.shape[1] < chunk_size:
            pad_len = chunk_size - actions.shape[1]
            actions = F.pad(actions, (0, 0, 0, pad_len))  # pad last-but-one dim (time)
        b[ACTION] = actions
        adapt_ehpl_image_keys_inplace(b, expected_image_keys)
        b = preprocessor(b)
        b = _move_to_device_inplace(b)
        loss_vec, _ = policy.forward(b, reduction="none")
        return loss_vec

    # NOTE: Some SigLIP / LayerNorm kernels in the Paligemma vision tower can error under bf16 autocast
    # (expected Float but found BFloat16). For EHPL stability we run the preference forward in fp32.
    # This keeps training correct; you can re-enable autocast later once kernels are verified.
    with torch.autocast(device_type=accelerator.device.type, enabled=False):
        loss_w = _one_side("action_w")
        loss_l = _one_side("action_l")
        delta = loss_w - loss_l
        loss = F.softplus(beta * delta).mean()

    accelerator.backward(loss)

    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    with lock if lock is not None else nullcontext():
        optimizer.step()
    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time

    output_dict = {
        "ehpl_loss_w_mean": loss_w.mean().item(),
        "ehpl_loss_l_mean": loss_l.mean().item(),
        "ehpl_delta_mean": delta.mean().item(),
    }
    return train_metrics, output_dict


def make_ehpl_train_dataset(cfg: Any) -> EhplPreferenceTrainDataset:
    """Build EHPL dataset from `TrainPipelineConfig` (requires `dataset.root` + `ehpl.pairs_parquet`)."""
    root = cfg.dataset.root
    if not root:
        raise ValueError("ehpl.enable=true requires `dataset.root` pointing to a v2.1 lerobot dataset.")
    if not cfg.ehpl.pairs_parquet:
        raise ValueError("ehpl.enable=true requires `ehpl.pairs_parquet` (P2 pairs parquet path).")
    stats_json = cfg.ehpl.stats_json or str(Path(root) / "meta" / "stats.json")
    return EhplPreferenceTrainDataset(
        dataset_root=str(root),
        pairs_parquet=str(cfg.ehpl.pairs_parquet),
        stats_json=stats_json,
        task_index_to_instruction_json=cfg.ehpl.task_index_to_instruction_json,
    )
