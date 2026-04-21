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

"""Standalone EHPL (skill-level preference) post-training script.

Minimal self-contained loop for debugging/prototyping EHPL on pi0/pi0.5.
For production use, prefer the integrated `lerobot-train --ehpl.enable=true` path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader


def load_stats_v21(stats_json_path: str) -> dict[str, dict[str, Any]]:
    """Load v2.1 meta/stats.json into the dict expected by NormalizerProcessorStep."""
    with open(stats_json_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return stats


def _monkeypatch_siglip_check() -> None:
    """Allow PI0/PI05 to run without transformers-replace installation.

    Upstream OpenPI/LeRobot expects `from transformers.models.siglip import check` to exist and
    provide `check_whether_transformers_replace_is_installed_correctly()`.
    Some transformers builds do not ship this module. For EHPL offline post-training, we can
    safely bypass this guard as we are not modifying SigLIP internals here.
    """
    import sys
    import types

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

    check_mod.check_whether_transformers_replace_is_installed_correctly = _ok
    sys.modules[modname] = check_mod
    setattr(siglip_pkg, "check", check_mod)


def _monkeypatch_gemma_rmsnorm_cond() -> None:
    """Patch GemmaRMSNorm.forward to accept OpenPI's `cond` kwarg (ignored).

    Some OpenPI ports call `layernorm(x, cond=...)`. Vanilla transformers may not support
    this signature. For EHPL debugging runs, we can ignore `cond` to unblock execution.
    """
    try:
        from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
    except Exception:
        return

    code = getattr(GemmaRMSNorm.forward, "__code__", None)
    if code is not None and "cond" in code.co_varnames:
        return

    orig_forward = GemmaRMSNorm.forward

    def forward(self, hidden_states, cond=None):
        return orig_forward(self, hidden_states)

    GemmaRMSNorm.forward = forward


def _adapt_image_keys_inplace(batch: dict[str, Any], expected_image_keys: list[str]) -> None:
    """Best-effort mapping from common LeRobot dataset keys to PI0/PI05 expected image keys."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=str, choices=["pi0", "pi05"], required=True)
    ap.add_argument("--pretrained", type=str, required=True, help="SFT checkpoint path or HF repo id")
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--pairs_parquet", type=str, required=True)
    ap.add_argument(
        "--min_pair_margin",
        type=float,
        default=None,
        help="If pairs parquet has a `margin` column, keep only rows with margin >= this value.",
    )
    ap.add_argument("--stats_json", type=str, default=None, help="Path to meta/stats.json (v2.1)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2.5e-05)
    ap.add_argument("--beta", type=float, default=1.0, help="SMPO temperature beta (pairwise preference strength)")
    ap.add_argument("--save_dir", type=str, default="/mydata/Evo-RL/ehpl/checkpoints/ehpl_posttrain_debug")
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--skip_siglip_check", action="store_true", help="Bypass SigLIP transformers-replace guard")
    ap.add_argument("--hf_home", type=str, default=None, help="Set HF_HOME so tokenizers/models load from a local hub cache (e.g. /mydata/hub).")
    ap.add_argument("--offline", action="store_true", help="Enable offline mode (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1). Requires cache present.")
    ap.add_argument("--no_tokenizer", action="store_true", help="Bypass tokenizer pipeline and use dummy language tokens (debug only).")
    args = ap.parse_args()

    # --- HF env setup ---
    hf_home = args.hf_home
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
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # --- Monkeypatches ---
    if args.skip_siglip_check:
        _monkeypatch_siglip_check()
    _monkeypatch_gemma_rmsnorm_cond()

    # --- Imports that require HF env to be set ---
    from lerobot.ehpl.pairs_p2_dataset import EhplPairsP2Dataset
    from lerobot.policies.factory import get_policy_class
    from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    # --- Accelerator ---
    accelerator = Accelerator(cpu=args.device == "cpu")
    device = accelerator.device
    torch.manual_seed(args.seed)

    # --- Stats ---
    dataset_root = args.dataset_root
    stats_json = args.stats_json or str(Path(dataset_root) / "meta" / "stats.json")
    stats = load_stats_v21(stats_json)

    # --- Policy ---
    if args.skip_siglip_check:
        _monkeypatch_siglip_check()
    _monkeypatch_gemma_rmsnorm_cond()

    policy_cls = get_policy_class(args.policy)
    policy = policy_cls.from_pretrained(args.pretrained)
    policy.config.pretrained_path = Path(args.pretrained)

    # --- Preprocessor ---
    if args.policy == "pi05":
        preproc, _ = make_pi05_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=policy.config.pretrained_path,
            dataset_stats=stats,
        )
    else:
        preproc, _ = make_pi0_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=policy.config.pretrained_path,
            dataset_stats=stats,
        )

    # --- Dataset ---
    ds = EhplPairsP2Dataset(
        dataset_root=dataset_root,
        pairs_parquet=args.pairs_parquet,
        task_index_to_instruction=None,
        min_pair_margin=args.min_pair_margin,
    )

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        keys = batch[0].keys()
        for k in keys:
            vals = [b[k] for b in batch]
            if torch.is_tensor(vals[0]):
                out[k] = torch.stack(vals, dim=0)
            else:
                out[k] = vals
        return out

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )

    # --- Optimizer ---
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)

    # --- Prepare ---
    policy, opt, dl = accelerator.prepare(policy, opt, dl)
    policy.train()

    # --- Expected image keys ---
    expected_image_keys = list(
        getattr(accelerator.unwrap_model(policy).config, "image_features", {}).keys()
    )

    # --- Training loop ---
    step = 0
    for batch in dl:
        if step >= args.steps:
            break

        # --- Forward: preferred ---
        batch_w = dict(batch)
        batch_w[ACTION] = batch["action_w"]
        _adapt_image_keys_inplace(batch_w, expected_image_keys)

        # Handle no_tokenizer mode: inject dummy language tokens
        if args.no_tokenizer:
            bsz = batch_w[ACTION].shape[0]
            max_len = getattr(policy.config, "tokenizer_max_length", 64) if hasattr(policy, "config") else 64
            batch_w[OBS_LANGUAGE_TOKENS] = torch.zeros(bsz, max_len, dtype=torch.int64, device=device)
            batch_w[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(bsz, max_len, dtype=torch.bool, device=device)

        batch_w = preproc(batch_w)

        loss_w, _ = policy.forward(batch_w, reduction="none")

        # --- Forward: rejected ---
        batch_l = dict(batch)
        batch_l[ACTION] = batch["action_l"]
        _adapt_image_keys_inplace(batch_l, expected_image_keys)

        if args.no_tokenizer:
            batch_l[OBS_LANGUAGE_TOKENS] = batch_w.get(OBS_LANGUAGE_TOKENS, torch.zeros(bsz, max_len, dtype=torch.int64, device=device))
            batch_l[OBS_LANGUAGE_ATTENTION_MASK] = batch_w.get(OBS_LANGUAGE_ATTENTION_MASK, torch.ones(bsz, max_len, dtype=torch.bool, device=device))

        batch_l = preproc(batch_l)

        loss_l, _ = policy.forward(batch_l, reduction="none")

        # --- Preference loss ---
        delta = loss_w - loss_l
        pref_loss = F.softplus(args.beta * delta).mean()

        accelerator.backward(pref_loss)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if accelerator.is_main_process:
            print(
                f"step={step}"
                f" pref_loss={pref_loss.item():.4f}"
                f" loss_w={loss_w.mean().item():.4f}"
                f" loss_l={loss_l.mean().item():.4f}"
                f" delta={delta.mean().item():.4f}"
            )

        step += 1

        # --- Save checkpoint ---
        if args.save_every > 0 and step % args.save_every == 0:
            save_dir = Path(args.save_dir) / f"step_{step:06d}"
            save_dir.mkdir(parents=True, exist_ok=True)
            accelerator.unwrap_model(policy).save_pretrained(str(save_dir))

    # --- Final save ---
    save_dir = Path(args.save_dir) / "final"
    save_dir.mkdir(parents=True, exist_ok=True)
    accelerator.unwrap_model(policy).save_pretrained(str(save_dir))
    print(f"Saved final checkpoint to {save_dir}")


if __name__ == "__main__":
    main()
