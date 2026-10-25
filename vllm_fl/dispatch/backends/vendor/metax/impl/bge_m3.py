# Copyright (c) 2026 BAAI. All rights reserved.
"""Explicit MetaX policy selection for BGE-M3's FP32 encoder adapter."""

from functools import partial

import torch
from vllm.config.vllm import get_current_vllm_config_or_none

import vllm_fl.envs as fl_envs
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


def attention_backend(use_mla=False, use_sparse=False):
    config = get_current_vllm_config_or_none()
    model = getattr(config, "model_config", None)
    if (
        not use_mla
        and not use_sparse
        and model is not None
        and model.dtype == torch.float32
        and "BgeM3EmbeddingModelFL"
        in (getattr(model.hf_config, "architectures", None) or [])
    ):
        if not fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM:
            raise RuntimeError("BGE-M3 FP32 MetaX policy requires FlagGems-vllm")
        return "vllm_fl.models.bge_m3.attention.BgeM3AttentionBackend"

    from vllm_fl.dispatch.backends.vendor.metax.metax import MacaBackend

    return MacaBackend().attention_backend(use_mla=use_mla, use_sparse=use_sparse)


def register_bge_m3_operators(registry, *, is_platform_available):
    selected_backend = partial(attention_backend)
    selected_backend._is_available = is_platform_available
    registry.register_many(
        [
            OpImpl(
                "attention_backend",
                "vendor.metax.bge_m3",
                BackendImplKind.VENDOR,
                selected_backend,
                vendor="metax",
                priority=BackendPriority.VENDOR,
            )
        ]
    )
