# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems implementations for mHC (Manifold-Constrained Hyper-Connections) operators.
"""

import torch

from vllm_fl.ops.deepseek_v4_metax.mhc import mhc_pre as _mhc_pre_graph
from flaggems_vllm import mhc_post as _fgv_mhc_post
from vllm_fl.ops.deepseek_v4_metax.hc_head import hc_head as hc_head_fused_kernel


def mhc_pre_flaggems(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _mhc_pre_graph(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
    )


def mhc_post_flaggems(
    x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
) -> torch.Tensor:
    return _fgv_mhc_post(x, residual, post, comb)


def hc_head_fused_kernel_flaggems(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    """FlagGems native implementation of hc_head_fused_kernel. Mutates `out` in-place."""
    hc_head_fused_kernel(
        hs_flat, fn, hc_scale, hc_base, out, hidden_size, rms_eps, hc_eps, hc_mult
    )
