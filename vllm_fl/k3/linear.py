"""K3 shape dispatch registered by the plugin; FlagGems remains independent."""

import logging
import torch
import triton
import flag_gems
from triton.runtime.jit import JITFunction
from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.ops.linear import linear_kernel
from .tp8_linear import plan, try_run

logger = logging.getLogger(__name__)


# Prefill fixed tilings reuse this canonical kernel and keep one registered entry.
def _k3_raw_prefill_linear():
    kernel = linear_kernel
    for _ in range(5):
        if isinstance(kernel, JITFunction):
            return kernel
        kernel = kernel.fn
    raise RuntimeError("Unexpected linear launch-wrapper depth")


_K3_PREFILL_LINEAR_KERNEL = _k3_raw_prefill_linear()
_K3_PREFILL_FIXED_TILES = {
    (3553, 384, 7168): (128, 128, 64, 8, 2),
    (3553, 7168, 192): (128, 128, 32, 8, 2),
    (4096, 7168, 192): (128, 128, 32, 8, 2),
    (3553, 1680, 7168): (128, 128, 64, 8, 2),
    (4096, 1680, 7168): (128, 128, 64, 8, 2),
}

# K3 TP32 BF16 decode shapes validated on real rank0/31 weight slices.
# Reuse the existing public FlagGems MM dispatch; no extra registration.
_K3_MM_BATCHES = (1, 2, 4, 8, 16, 29, 30, 31, 32, 33, 64)
_K3_MM_PROJECTIONS = frozenset(
    {
        (384, 7168),
        (576, 1536),
        (1680, 7168),
        (2112, 7168),
    }
)


def _k3_decode_mm_eligible(x, weight, bias, M, N, K):
    if (
        runtime.device.vendor_name != "metax"
        or M not in _K3_MM_BATCHES
        or bias is not None
        or not x.is_cuda
        or x.device != weight.device
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or weight.shape != (N, K)
    ):
        return False
    # Router scores take this path only for the byte-exact qualified M64 case.
    return (
        (M in (1, 2, 4, 8, 16, 32, 64) and (N, K) in _K3_MM_PROJECTIONS)
        or (M == 64 and (N, K) == (896, 7168))
        # Local-vocabulary projection: tiny near-zero drift accepted separately,
        # with original strict results, absolute errors and argmax checks retained.
        or (N, K) == (5120, 7168)
    )


# Prefill profiles contain these dominant total-token shapes. Retain the
# original dispatch for every unqualified shape, including router/KDA scores.
def _k3_prefill_mm_eligible(x, weight, bias, M, N, K):
    supported_layout = x.is_contiguous() or (
        (N, K) == (768, 512) and x.ndim == 2 and x.stride() == (576, 1)
    )
    if (
        runtime.device.vendor_name != "metax"
        or bias is not None
        or not x.is_cuda
        or x.device != weight.device
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not supported_layout
        or not weight.is_contiguous()
        or weight.shape != (N, K)
    ):
        return False
    return (
        ((N, K) == (768, 512) and 4097 <= M <= 65536)
        or ((N, K) == (384, 7168) and M in (529, 1024))
        or ((N, K) == (2112, 7168) and M in (3553, 4096))
        or ((N, K) == (576, 1536) and M == 529)
    )


def linear(input, weight, bias=None, *, tp8_compute=True):
    """
    Applies a linear transformation to the incoming data: y = xA^T + b

    Args:
        input: Input tensor of shape (*, in_features) where * means any number of
               additional dimensions, including none.
        weight: Weight tensor of shape (out_features, in_features)
        bias: Bias tensor of shape (out_features), optional

    Returns:
        Output tensor of shape (*, out_features)
    """
    logger.debug("GEMS LINEAR")

    input_dim = input.dim()
    if input_dim == 1:
        # Single 1D input: treat as (1, in_features)
        input = input.unsqueeze(0)
        single_1d = True
    else:
        single_1d = False

    # Flatten batch dimensions: (*, in_features) -> (batch, in_features)
    batch_dims = input.shape[:-1]
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    M = batch_size
    K = input.shape[-1]  # in_features
    N = weight.shape[0]  # out_features

    # Flatten input: (*, K) -> (M, K)
    input_flat = input.view(M, K)

    # Ensure weight is contiguous and properly shaped
    weight = weight.contiguous()

    # Allocate output
    output = torch.empty((M, N), device=input.device, dtype=input.dtype)

    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    fixed = _K3_PREFILL_FIXED_TILES.get((M, N, K))
    tp8_handled = False
    if tp8_compute and runtime.device.vendor_name == "metax" and input_flat.is_cuda:
        if plan(M, N, K) is not None:
            with torch_device_fn.device(input.device):
                tp8_handled = try_run(
                    input_flat, weight, bias, output, _K3_PREFILL_LINEAR_KERNEL
                )
    if tp8_handled:
        pass
    elif (
        fixed is not None
        and runtime.device.vendor_name == "metax"
        and bias is None
        and input_flat.is_cuda
        and input_flat.device == weight.device
        and input_flat.dtype == weight.dtype == torch.bfloat16
        and input_flat.is_contiguous()
        and weight.is_contiguous()
    ):
        bm, bn, bk, warps, stages = fixed
        with torch_device_fn.device(input.device):
            _K3_PREFILL_LINEAR_KERNEL[(triton.cdiv(M, bm), triton.cdiv(N, bn))](
                input_flat,
                weight,
                weight,
                output,
                M,
                N,
                K,
                input_flat.stride(0),
                input_flat.stride(1),
                weight.stride(0),
                weight.stride(1),
                output.stride(0),
                output.stride(1),
                0,
                BIAS=False,
                BLOCK_SIZE_M=bm,
                BLOCK_SIZE_N=bn,
                BLOCK_SIZE_K=bk,
                num_warps=warps,
                num_stages=stages,
            )
    elif _k3_decode_mm_eligible(
        input_flat, weight, bias, M, N, K
    ) or _k3_prefill_mm_eligible(input_flat, weight, bias, M, N, K):
        with torch_device_fn.device(input.device):
            flag_gems.mm_out(input_flat, weight.t(), out=output)
    else:
        with torch_device_fn.device(input.device):
            linear_kernel[grid](
                input_flat,
                weight,
                bias if bias is not None else weight,  # Pass dummy ptr if no bias
                output,
                M,
                N,
                K,
                input_flat.stride(0),
                input_flat.stride(1),
                weight.stride(0),
                weight.stride(1),
                output.stride(0),
                output.stride(1),
                bias.stride(0) if bias is not None else 0,
                BIAS=bias is not None,
            )

    # Reshape output: (M, N) -> (*, N)
    output = output.view(*batch_dims, N)

    # If original input was 1D, squeeze the batch dim
    if single_1d:
        output = output.squeeze(0)

    return output
