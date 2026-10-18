"""General-shape MHC fallback. The default DS path is mhc_pre_metax.

Projection registration and fallback kernels are separate leaves of the import
graph. This module and the split tail never import each other backwards.
"""

import torch
from flaggems_vllm.ops import metax_tuning as _tuning
from . import mhc_prefill_split as _prefill_split
from . import mhc_pre_mm_op as _mhc_pre_mm_op  # noqa: F401
from .mhc_kernels import mhc_pre_fused_kernel_hc_mult_4, mhc_pre_generic_kernel


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
    """
    Optimized mHC pre block.

    - hc_mult == 4: specialized fused Triton kernel
    - hc_mult != 4: generic Triton kernel aligned to reference math
    """
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size

    assert fn.shape == (hc_mult3, hc_hidden_size)

    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size).contiguous()
    num_tokens = residual_flat.shape[0]
    device = residual.device
    if num_tokens <= 512:
        num_tokens_bucket = 1
    elif num_tokens <= 1024:
        num_tokens_bucket = 2
    elif num_tokens <= 2048:
        num_tokens_bucket = 3
    elif num_tokens <= 4096:
        num_tokens_bucket = 4
    else:
        num_tokens_bucket = 5

    # ── Step 1: GEMM via graph-visible custom op ──
    # Do not use a Python shape guard here: Dynamo/Inductor can fold it at capture
    # time.  Keep the projection as an opaque op and let its runtime implementation
    # dispatch on the real replay shape.  The op returns bf16 exactly like
    # torch.mm(x_flat, fn_bf16.t()); the original path then casts to fp32.
    x_flat = residual_flat.reshape(num_tokens, hc_hidden_size)
    fn_bf16 = fn.to(dtype=torch.bfloat16)
    gemm_out = torch.ops.dsv4_fgvllm_migration.mhc_pre_mm(x_flat, fn_bf16).float()

    # Runtime dispatch inside a graph-visible custom op: do not branch on
    # symbolic num_tokens here, since VLLM compilation may freeze that branch.
    if _tuning.current.mhc_split_tail and hc_mult == 4:
        post_mix, comb_mix, layer_input = _prefill_split.tail(
            gemm_out,
            hc_scale,
            hc_base,
            residual_flat,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
        return (
            post_mix.view(*outer_shape, hc_mult, 1),
            comb_mix.view(*outer_shape, hc_mult, hc_mult),
            layer_input.view(*outer_shape, hidden_size),
        )

    # ── Step 2: Fused sqrsum + norm + mix + sinkhorn + weighted sum ──
    post_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device)
    comb_mix = torch.empty(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )

    if hc_mult == 4:
        mhc_pre_fused_kernel_hc_mult_4[(num_tokens,)](
            gemm_out,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            num_tokens,
            num_tokens_bucket,
            residual_flat.stride(0),
            residual_flat.stride(1),
            residual_flat.stride(2),
            layer_input.stride(0),
            layer_input.stride(1),
            hidden_size,
            hc_hidden_size,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
            HC_MULT3=hc_mult3,
        )
    else:
        mhc_pre_generic_kernel[(num_tokens,)](
            gemm_out,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            num_tokens,
            num_tokens_bucket,
            residual_flat.stride(0),
            residual_flat.stride(1),
            residual_flat.stride(2),
            layer_input.stride(0),
            layer_input.stride(1),
            hidden_size,
            hc_hidden_size,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
            HC=hc_mult,
        )

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


# ───────────────────────── Reference implementations ─────────────────────────


def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_pre_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference."""
    hc_mult = residual.shape[-2]
    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    mixes = (
        residual_flat @ fn.T * (sqrsum.unsqueeze(-1) / fn.shape[-1] + rms_eps).rsqrt()
    )
    hc_scale_expanded = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ]
    )
    mixes = mixes * hc_scale_expanded + hc_base
    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
    post_mix = (
        mixes[:, hc_mult : 2 * hc_mult].sigmoid() * hc_post_mult_value
    ).unsqueeze(-1)
    res_mix = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)
    res_mix = sinkhorn_normalize_ref(
        res_mix, repeat=sinkhorn_repeat, eps=hc_sinkhorn_eps
    )
    layer_input = (residual * pre_mix).sum(-2).bfloat16()
    return post_mix, res_mix, layer_input
