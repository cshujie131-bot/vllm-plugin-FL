# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Graph-safe adapter for the MetaX mhc_pre implementation from FGV PR #755.

Kernel source stays in the FGV MetaX backend at upstream SHA
3a6ae4695cb63c8fe0a39774bf470e580fbbcd10. mhc_post is deliberately unchanged.
Shape dispatch is inside an opaque op, not frozen by Dynamo tracing.
"""

import torch
from flaggems_vllm.runtime.backend._metax.ops.mhc_pre import run as _run_pr755
from flaggems_vllm.ops.mhc.mhc_pre import mhc_pre as _legacy_mhc_pre


@torch.library.custom_op("vllm_fl::dsv4_mhc_pre", mutates_args=())
def _mhc_pre_metax(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eligible = (
        residual.ndim >= 3
        and residual.shape[-2:] == (4, 4096)
        and residual.dtype == torch.bfloat16
        and residual.is_cuda
        and residual.is_contiguous()
        and fn.shape == (24, 16384)
        and fn.dtype == torch.float32
        and fn.is_contiguous()
        and hc_scale.shape == (3,)
        and hc_base.shape == (24,)
        and hc_scale.dtype == hc_base.dtype == torch.float32
        and hc_scale.is_contiguous()
        and hc_base.is_contiguous()
        and fn.device == hc_scale.device == hc_base.device == residual.device
        and sinkhorn_repeat >= 1
    )
    if not eligible:
        return _legacy_mhc_pre(
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
    outer = residual.shape[:-2]
    flat = residual.reshape(-1, 4, 4096)
    if flat.shape[0] == 0:
        return (
            torch.empty((*outer, 4, 1), device=residual.device, dtype=torch.float32),
            torch.empty((*outer, 4, 4), device=residual.device, dtype=torch.float32),
            torch.empty((*outer, 4096), device=residual.device, dtype=torch.bfloat16),
        )
    post, comb, layer = _run_pr755(
        flat,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    return post.view(*outer, 4, 1), comb.view(*outer, 4, 4), layer.view(*outer, 4096)


@_mhc_pre_metax.register_fake
def _mhc_pre_metax_fake(
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
):
    outer, hc, h = residual.shape[:-2], residual.shape[-2], residual.shape[-1]
    return (
        torch.empty((*outer, hc, 1), device=residual.device, dtype=torch.float32),
        torch.empty((*outer, hc, hc), device=residual.device, dtype=torch.float32),
        torch.empty((*outer, h), device=residual.device, dtype=torch.bfloat16),
    )


def mhc_pre(
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
    return _mhc_pre_metax(
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
