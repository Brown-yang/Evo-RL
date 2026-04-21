#!/usr/bin/env python

"""EHPL scoring model configuration.

You only need **one** config to run the EHPL scorer end-to-end:

  (images, text, action_chunk) -> score(A)

Internally, the scorer is composed of:
  1) a frozen multimodal embedding model that produces `c`
  2) a trainable segment-level judge that maps (c, action_chunk) -> Q/V/A

This file intentionally exposes a single public config: `EhplScoringConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class EhplScoringConfig:
    """Config for the EHPL scorer (frozen encoder + trainable judge).

    Sections follow `pistar06` for readability.
    """

    # =========================
    # Backbone components (frozen embedding model)
    # =========================
    vl_embedding_repo_id: str = "Qwen/Qwen3-VL-Embedding-2B"
    vl_embedding_revision: str | None = None

    # =========================
    # Input fields / keys
    # =========================
    task_field: str = "task"
    camera_features: list[str] = field(default_factory=list)

    # (optional) if you build prompts that include discretized state
    include_state_in_prompt: bool = True
    state_feature: str = "observation.state"
    max_state_dim: int = 32
    state_discretization_bins: int = 256

    # =========================
    # Context encoder shape (embedding output)
    # =========================
    context_dim: int = 2048  # Qwen3-VL-Embedding-2B output dim

    # =========================
    # Judge model shape (trainable)
    # =========================
    action_dim: int = 7
    context_feat_dim: int = 128
    action_feat_dim: int = 128
    judge_hidden_dim: int = 256
    judge_layer_norm: bool = True

    action_encoder: Literal["mlp_pool", "transformer"] = "mlp_pool"
    # mlp_pool params
    action_pool: Literal["mean", "max"] = "mean"
    token_hidden_dims: tuple[int, ...] = (128,)
    # transformer params
    n_layers: int = 2
    n_heads: int = 4
    ff_dim: int | None = None
    transformer_pool: Literal["mean", "cls"] = "mean"

    keepdim: bool = False

    # =========================
    # Runtime (frozen embedding)
    # =========================
    dropout: float = 0.1
    dtype: str = "float32"
    offline: bool = True


__all__ = ["EhplScoringConfig"]

