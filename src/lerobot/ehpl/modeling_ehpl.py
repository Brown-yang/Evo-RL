#!/usr/bin/env python

"""Modeling code for EHPL model-learning components."""

from __future__ import annotations

from typing import Literal, Any

import os
from PIL import Image
import numpy as np
import torch
from torch import Tensor, nn

from dataclasses import dataclass

from lerobot.ehpl.configuration_ehpl import EhplScoringConfig


def _make_mlp(
    in_dim: int,
    hidden_dims: list[int],
    out_dim: int,
    *,
    activation: nn.Module | None = None,
    dropout: float = 0.0,
    layer_norm: bool = False,
) -> nn.Sequential:
    """Small helper to build an MLP."""
    if activation is None:
        activation = nn.GELU()
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ActionEncoderMLPPool(nn.Module):
    """Encode action chunk [B,T,D] -> [B,F] via per-step MLP + pooling."""

    def __init__(
        self,
        action_dim: int,
        feat_dim: int,
        *,
        token_hidden_dims: list[int] | None = None,
        pool: Literal["mean", "max"] = "mean",
        dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if token_hidden_dims is None:
            token_hidden_dims = [128]
        self.pool = pool
        self.token_mlp = _make_mlp(
            in_dim=action_dim,
            hidden_dims=token_hidden_dims,
            out_dim=feat_dim,
            dropout=dropout,
            layer_norm=layer_norm,
        )

    def forward(self, action: Tensor) -> Tensor:
        if action.ndim != 3:
            raise ValueError(f"Expected action shape [B,T,D], got {tuple(action.shape)}")
        b, t, d = action.shape
        tokens = self.token_mlp(action.reshape(b * t, d)).reshape(b, t, -1)
        if self.pool == "mean":
            return tokens.mean(dim=1)
        if self.pool == "max":
            return tokens.max(dim=1).values
        raise ValueError(f"Unsupported pool='{self.pool}'")


class ActionEncoderTransformer(nn.Module):
    """Encode action chunk [B,T,D] -> [B,F] via projection + Transformer + pooling."""

    def __init__(
        self,
        action_dim: int,
        feat_dim: int,
        *,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_dim: int | None = None,
        dropout: float = 0.0,
        pool: Literal["mean", "cls"] = "mean",
    ) -> None:
        super().__init__()
        if feat_dim % n_heads != 0:
            raise ValueError(f"feat_dim ({feat_dim}) must be divisible by n_heads ({n_heads}).")
        if ff_dim is None:
            ff_dim = feat_dim * 4
        self.pool = pool
        self.in_proj = nn.Linear(action_dim, feat_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim)) if pool == "cls" else None
        enc_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(feat_dim)

    def forward(self, action: Tensor) -> Tensor:
        if action.ndim != 3:
            raise ValueError(f"Expected action shape [B,T,D], got {tuple(action.shape)}")
        x = self.in_proj(action)  # [B,T,F]
        if self.pool == "cls":
            assert self.cls_token is not None
            x = torch.cat([self.cls_token.expand(x.shape[0], 1, -1), x], dim=1)
        x = self.encoder(x)
        x = self.out_norm(x)
        if self.pool == "cls":
            return x[:, 0, :]
        return x.mean(dim=1)


class AdvantageJudge(nn.Module):
    """
    Segment-level advantage judge.

    Computes:
      Q(c, u) from fused (context_feat, action_feat)
      V(c) from context_feat only
      A(c, u) = Q - V
    """

    def __init__(self, cfg: AdvantageJudgeConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.context_encoder = _make_mlp(
            in_dim=cfg.context_dim,
            hidden_dims=[cfg.fusion_hidden_dim],
            out_dim=cfg.context_feat_dim,
            dropout=cfg.dropout,
            layer_norm=cfg.layer_norm,
        )

        if cfg.action_encoder == "mlp_pool":
            self.action_encoder = ActionEncoderMLPPool(
                action_dim=cfg.action_dim,
                feat_dim=cfg.action_feat_dim,
                token_hidden_dims=list(cfg.token_hidden_dims),
                pool=cfg.action_pool,
                dropout=cfg.dropout,
                layer_norm=cfg.layer_norm,
            )
        elif cfg.action_encoder == "transformer":
            self.action_encoder = ActionEncoderTransformer(
                action_dim=cfg.action_dim,
                feat_dim=cfg.action_feat_dim,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                ff_dim=cfg.ff_dim,
                dropout=cfg.dropout,
                pool=cfg.transformer_pool,
            )
        else:
            raise ValueError(f"Unsupported action_encoder='{cfg.action_encoder}'")

        fused_dim = cfg.context_feat_dim + cfg.action_feat_dim

        self.shared_backbone = _make_mlp(
            in_dim=fused_dim,
            hidden_dims=[cfg.fusion_hidden_dim, cfg.fusion_hidden_dim],
            out_dim=cfg.fusion_hidden_dim,
            dropout=cfg.dropout,
            layer_norm=cfg.layer_norm,
        )

        self.q_head = nn.Linear(cfg.fusion_hidden_dim, 1)
        self.v_head = _make_mlp(
            in_dim=cfg.context_feat_dim,
            hidden_dims=[cfg.fusion_hidden_dim],
            out_dim=1,
            dropout=cfg.dropout,
            layer_norm=cfg.layer_norm,
        )

    def forward(self, context: Tensor, action: Tensor) -> dict[str, Tensor]:
        if context.ndim != 2:
            raise ValueError(f"Expected context shape [B,C], got {tuple(context.shape)}")
        if action.ndim != 3:
            raise ValueError(f"Expected action shape [B,T,A], got {tuple(action.shape)}")
        if context.shape[0] != action.shape[0]:
            raise ValueError(f"Batch mismatch: {context.shape[0]} vs {action.shape[0]}")
        if context.shape[1] != self.cfg.context_dim:
            raise ValueError(f"context_dim mismatch: expected {self.cfg.context_dim}, got {context.shape[1]}")
        if action.shape[2] != self.cfg.action_dim:
            raise ValueError(f"action_dim mismatch: expected {self.cfg.action_dim}, got {action.shape[2]}")

        context_feat = self.context_encoder(context)
        action_feat = self.action_encoder(action)
        fused = torch.cat([context_feat, action_feat], dim=-1)
        shared = self.shared_backbone(fused)

        q = self.q_head(shared)
        v = self.v_head(context_feat)
        a = q - v

        if not self.cfg.keepdim:
            q = q.squeeze(-1)
            v = v.squeeze(-1)
            a = a.squeeze(-1)

        return {"Q": q, "V": v, "A": a}


def _resolve_load_dtype(dtype: str) -> torch.dtype:
    if dtype in {"float32", "fp32"}:
        return torch.float32
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype='{dtype}'. Use float32/float16/bfloat16.")


def _extract_hidden_size(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("Expected HF model with `.config` to infer hidden size.")
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return int(cfg.text_config.hidden_size)
    raise ValueError("Could not infer hidden_size from model config.")


def _extract_vision_feature_size(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("Expected HF vision model with `.config` to infer hidden size.")
    if hasattr(cfg, "vision_config") and hasattr(cfg.vision_config, "hidden_size"):
        return int(cfg.vision_config.hidden_size)
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    raise ValueError("Could not infer vision feature size from model config.")


def _resolve_image_size(image_processor: Any) -> tuple[int, int]:
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        h = int(size.get("height") or size.get("shortest_edge") or 384)
        w = int(size.get("width") or size.get("shortest_edge") or 384)
        return h, w
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return int(size[0]), int(size[1])
    # reasonable default for siglip
    return 384, 384


def _resolve_norm_stats(image_processor: Any) -> tuple[list[float], list[float]]:
    mean = getattr(image_processor, "image_mean", None) or [0.5, 0.5, 0.5]
    std = getattr(image_processor, "image_std", None) or [0.5, 0.5, 0.5]
    return list(map(float, mean)), list(map(float, std))


@dataclass(frozen=True)
class AdvantageJudgeConfig:
    """Internal config for AdvantageJudge network shape."""

    context_dim: int
    action_dim: int
    context_feat_dim: int = 128
    action_feat_dim: int = 128
    fusion_hidden_dim: int = 256
    dropout: float = 0.0
    layer_norm: bool = True
    action_encoder: Literal["mlp_pool", "transformer"] = "mlp_pool"
    action_pool: Literal["mean", "max"] = "mean"
    token_hidden_dims: tuple[int, ...] = (128,)
    n_layers: int = 2
    n_heads: int = 4
    ff_dim: int | None = None
    transformer_pool: Literal["mean", "cls"] = "mean"
    keepdim: bool = False


class EhplFrozenContextEncoder(nn.Module):
    """Frozen multimodal embedding using Qwen3-VL-Embedding (sentence-transformers).

    Produces a fixed-dim context embedding `c` from (image, text).
    """

    def __init__(self, cfg: EhplScoringConfig):
        super().__init__()
        self.cfg = cfg
        self._st_model = None  # lazy init

    def _lazy_load(self):
        if self._st_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for Qwen3-VL-Embedding. "
                "Install with `pip install sentence-transformers`."
            ) from e

        offline = bool(self.cfg.offline) or bool(int(os.environ.get("HF_HUB_OFFLINE", "0")))
        model_kwargs = {"local_files_only": True} if offline else {}
        self._st_model = SentenceTransformer(
            self.cfg.vl_embedding_repo_id,
            revision=self.cfg.vl_embedding_revision,
            device="cpu",  # we return torch tensor; caller can move to GPU if desired
            model_kwargs=model_kwargs,
        )

    @staticmethod
    def _to_pil(img: Tensor) -> Image.Image:
        # img: [C,H,W] or [H,W,C], uint8 or float
        if img.ndim != 3:
            raise ValueError(f"Expected image rank-3, got {tuple(img.shape)}")
        x = img.detach().to("cpu")
        if x.dtype != torch.uint8:
            x = x.to(torch.float32)
            if torch.max(x) <= 1.0:
                x = (x * 255.0).clamp(0, 255)
            x = x.to(torch.uint8)
        if x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
            x = x.permute(1, 2, 0)  # CHW -> HWC
        arr = x.numpy()
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return Image.fromarray(arr, mode="RGB")

    def forward(self, *, images: Tensor, image_attention_mask: Tensor, text: list[str] | tuple[str, ...] | str) -> Tensor:
        self._lazy_load()
        assert self._st_model is not None

        if isinstance(text, str):
            texts = [text] * int(images.shape[0])
        else:
            texts = list(text)
        if images.ndim != 5:
            raise ValueError(f"'images' must have shape [B,N,C,H,W], got {tuple(images.shape)}.")
        if image_attention_mask.ndim != 2:
            raise ValueError(f"'image_attention_mask' must have shape [B,N], got {tuple(image_attention_mask.shape)}.")
        b, n = images.shape[:2]
        if len(texts) != b:
            raise ValueError(f"text batch size mismatch: got {len(texts)} texts for B={b}")

        mask = image_attention_mask.to(dtype=torch.bool, device=images.device)
        docs = []
        for i in range(b):
            # pick first valid camera; fallback to camera 0
            cam_idx = int(torch.argmax(mask[i].to(torch.int64)).item()) if bool(mask[i].any()) else 0
            pil = self._to_pil(images[i, cam_idx])
            docs.append({"text": texts[i], "image": pil})

        emb = self._st_model.encode(docs, convert_to_numpy=True, normalize_embeddings=False)
        if emb.ndim != 2 or emb.shape[0] != b:
            raise ValueError(f"Unexpected embedding shape from sentence-transformers: {emb.shape}")
        if int(emb.shape[1]) != int(self.cfg.context_dim):
            raise ValueError(
                f"context_dim mismatch: cfg.context_dim={self.cfg.context_dim} vs embed_dim={emb.shape[1]}"
            )
        return torch.from_numpy(emb).to(dtype=torch.float32)


class EhplScoringModel(nn.Module):
    """End-to-end scoring interface with a scalar score.

    Default output is a single scalar per skill/segment: `A(c, u)`.
    For debugging, set `return_details=True` to get full Q/V/A dict.
    """

    def __init__(self, cfg: EhplScoringConfig):
        super().__init__()
        self.cfg = cfg
        self.context_encoder = EhplFrozenContextEncoder(cfg)
        judge_cfg = AdvantageJudgeConfig(
            context_dim=cfg.context_dim,
            action_dim=cfg.action_dim,
            context_feat_dim=cfg.context_feat_dim,
            action_feat_dim=cfg.action_feat_dim,
            fusion_hidden_dim=cfg.judge_hidden_dim,
            dropout=cfg.dropout,
            layer_norm=cfg.judge_layer_norm,
            action_encoder=cfg.action_encoder,
            action_pool=cfg.action_pool,
            token_hidden_dims=cfg.token_hidden_dims,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ff_dim=cfg.ff_dim,
            transformer_pool=cfg.transformer_pool,
            keepdim=cfg.keepdim,
        )
        self.judge = AdvantageJudge(judge_cfg)

    def forward(
        self,
        *,
        images: Tensor,
        image_attention_mask: Tensor,
        text: list[str] | tuple[str, ...] | str,
        action: Tensor,
        return_details: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        context = self.context_encoder(
            images=images,
            image_attention_mask=image_attention_mask,
            text=text,
        )
        out = self.judge(context=context, action=action)
        if return_details:
            return out
        return out["A"]

    def score(
        self,
        *,
        images: Tensor,
        image_attention_mask: Tensor,
        text: list[str] | tuple[str, ...] | str,
        action: Tensor,
    ) -> Tensor:
        """Convenience wrapper returning scalar `A(c,u)` only."""
        return self.forward(
            images=images,
            image_attention_mask=image_attention_mask,
            text=text,
            action=action,
            return_details=False,
        )


def test() -> None:
    torch.manual_seed(0)
    b, t = 4, 32
    cfg = AdvantageJudgeConfig(context_dim=64, action_dim=7, action_encoder="mlp_pool")
    m = AdvantageJudge(cfg)
    out = m(torch.randn(b, 64), torch.randn(b, t, 7))
    assert out["A"].shape == (b,)

