# SPDX-License-Identifier: Apache-2.0
"""Explicit operator boundaries. No ATen/global vLLM registration replacement."""

import torch
from vllm import _custom_ops
from flag_gems.ops.dsv4_masked_fill import masked_fill_ as _mask_fast
from flaggems_vllm.ops.dsv4_int8_quant import quant as _quant
from flaggems_vllm.ops.dsv4_moe_align import align as selected_align
from flaggems_vllm.ops.silu_and_mul_with_clamp import silu_and_mul_with_clamp_kernel

_LIB = torch.library.Library("dsv4_fg_explicit", "FRAGMENT")


def _mask_impl(x, mask):
    eligible = (
        x.ndim == 2
        and x.shape[1] == 4096
        and x.is_contiguous()
        and x.dtype in (torch.bfloat16, torch.float32)
        and mask.dtype == torch.bool
        and mask.ndim == 2
        and mask.shape == (x.shape[0], 1)
        and mask.is_contiguous()
    )
    if eligible:
        _mask_fast(x, mask, 0)
    else:
        x.masked_fill_(mask, 0)


def _quant_impl(x):
    q, scale, _ = _quant(x)
    return q, scale


def _silu_impl(x, limit):
    d = x.shape[-1] // 2
    return silu_and_mul_with_clamp_kernel(x[..., :d], x[..., d:], limit)


_LIB.define("masked_fill_zero_(Tensor(a!) x, Tensor mask) -> ()")
_LIB.impl("masked_fill_zero_", _mask_impl, "CUDA")
_LIB.define("dynamic_int8(Tensor x) -> (Tensor, Tensor)")
_LIB.impl("dynamic_int8", _quant_impl, "CUDA")
_LIB.define("silu_clamp(Tensor x, Tensor limit) -> Tensor")
_LIB.impl("silu_clamp", _silu_impl, "CUDA")


@torch.library.register_fake("dsv4_fg_explicit::masked_fill_zero_")
def _mask_fake(x, mask):
    return None


@torch.library.register_fake("dsv4_fg_explicit::dynamic_int8")
def _quant_fake(x):
    return torch.empty_like(x, dtype=torch.int8), torch.empty(
        (x.shape[0], 1), device=x.device, dtype=torch.float32
    )


@torch.library.register_fake("dsv4_fg_explicit::silu_clamp")
def _silu_fake(x, limit):
    return torch.empty(
        (*x.shape[:-1], x.shape[-1] // 2), dtype=x.dtype, device=x.device
    )


def selected_mask(x, mask, value=0):
    if value != 0:
        raise ValueError("This embedding operator only supports zero fill")
    torch.ops.dsv4_fg_explicit.masked_fill_zero_(x, mask)
    return x


def scaled_int8_quant(x, scale=None, azp=None, symmetric=True):
    if (
        scale is None
        and azp is None
        and symmetric is True
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.is_contiguous()
        and x.shape[1] <= 16384
    ):
        q, s = torch.ops.dsv4_fg_explicit.dynamic_int8(x)
        return q, s, None
    return _custom_ops.scaled_int8_quant(x, scale, azp, symmetric)


def silu_clamp(x, limit):
    return torch.ops.dsv4_fg_explicit.silu_clamp(x, limit)
