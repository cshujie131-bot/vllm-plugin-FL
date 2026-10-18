import torch
import triton
import triton.language as tl

from vllm_fl import dsv4_config as _dsv4


@triton.jit
def bf16_mqa_logits_rowshard_kernel(
    # Pointers to matrices
    q_ptr,
    kv_ptr,
    weights_ptr,
    cu_seq_len_k_start_ptr,
    cu_seq_len_k_end_ptr,
    output_ptr,
    # Matrix dimensions
    seq_len,
    seq_len_kv,
    num_heads,
    head_dim: tl.constexpr,
    # Strides
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kvn,
    stride_kvd,
    stride_wm,
    stride_wh,
    stride_om,
    stride_on,
    # Options
    apply_mask: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fully fused MQA logits kernel in Triton.
    Computes: logits[m,n] = sum_h( ReLU(Q[m,h,:] @ KV[n,:]^T) * weights[m,h] )
    with optional masking.
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Mask for valid positions
    mask_m = offs_m < seq_len
    mask_n = offs_n < seq_len_kv

    # Accumulator for all heads
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over heads
    for h in range(num_heads):
        # Load weights for this head
        w_ptrs = weights_ptr + offs_m * stride_wm + h * stride_wh
        w = tl.load(w_ptrs, mask=mask_m, other=0.0)

        # Accumulator for GEMM across K dimension
        gemm_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # GEMM: Q[m,h,:] @ KV[n,:]^T
        for k in range(0, head_dim, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < head_dim

            # Load Q tile: [BLOCK_M, BLOCK_K]
            q_ptrs = (
                q_ptr
                + offs_m[:, None] * stride_qm
                + h * stride_qh
                + offs_k[None, :] * stride_qd
            )
            q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

            # Load KV tile: [BLOCK_N, BLOCK_K]
            kv_ptrs = (
                kv_ptr + offs_n[:, None] * stride_kvn + offs_k[None, :] * stride_kvd
            )
            kv = tl.load(kv_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

            # Compute dot product: Q @ KV^T
            gemm_acc += tl.dot(q, tl.trans(kv))

        # Apply ReLU
        gemm_acc = tl.maximum(gemm_acc, 0.0)

        # Multiply by weight and accumulate
        acc += gemm_acc * w[:, None]

    # Apply mask if needed
    if apply_mask:
        start_ptrs = cu_seq_len_k_start_ptr + offs_m
        end_ptrs = cu_seq_len_k_end_ptr + offs_m
        start_idx = tl.load(start_ptrs, mask=mask_m, other=0)
        end_idx = tl.load(end_ptrs, mask=mask_m, other=seq_len_kv)

        # Check if each position is valid
        valid = (offs_n[None, :] >= start_idx[:, None]) & (
            offs_n[None, :] < end_idx[:, None]
        )
        acc = tl.where(valid, acc, float("-inf"))

    # Store output
    out_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# All communication/state belongs to this opt-in experiment. No tensor/pointer cache.
import os
import datetime
import torch.distributed as dist
from vllm_fl.dispatch.backends.vendor.metax.impl.dsv4.indexer import (
    bf16_mqa_logits as _reference,
)

_PAIR = None


def _tp_info():
    from vllm.distributed import get_tp_group

    tp = get_tp_group()
    return tp.world_size, list(tp.ranks), tp.rank_in_group


def _pair():
    global _PAIR
    if _PAIR is None:
        size, ranks, local_rank = _tp_info()
        if size != 8 or dist.get_world_size() != 8:
            raise RuntimeError("Query-row sharding requires single-node TP8/DP1/PP1")
        for i in range(0, 8, 2):
            pg = dist.new_group(
                ranks=ranks[i : i + 2],
                backend="nccl",
                timeout=datetime.timedelta(seconds=180),
            )
            if local_rank in (i, i + 1):
                _PAIR = (pg, local_rank % 2)
    return _PAIR


def bf16_mqa_logits_prefill(q, kv, weights, ks, ke):
    (m, h, d) = q.shape
    n = kv.shape[0]
    eligible = (
        m >= 2048
        and n >= 1024
        and (h == 64)
        and (d == 128)
        and (q.dtype == torch.bfloat16)
        and (kv.dtype == torch.bfloat16)
        and (weights.dtype == torch.float32)
        and (ks.dtype == torch.int32)
        and (ke.dtype == torch.int32)
        and q.is_contiguous()
        and kv.is_contiguous()
        and weights.is_contiguous()
        and ks.is_contiguous()
        and ke.is_contiguous()
        and (weights.shape == (m, h))
        and (kv.shape == (n, d))
        and (ks.numel() == m)
        and (ke.numel() == m)
    )
    if not _dsv4.current.indexer_row_sharding or not eligible:
        return _reference(q, kv, weights, ks, ke)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "rowshard eligible prefill unexpectedly inside CUDA graph capture"
        )
    (group, rank) = _pair()
    chunk = (m + 127) // 128 * 64
    lo = rank * chunk
    hi = min(m, lo + chunk)
    qq = q[lo:hi]
    ww = weights[lo:hi]
    kk = ks.reshape(-1)[lo:hi]
    ee = ke.reshape(-1)[lo:hi]
    local = torch.zeros((chunk, n), device=q.device, dtype=torch.float32)
    result = torch.empty((2 * chunk, n), device=q.device, dtype=torch.float32)
    bf16_mqa_logits_rowshard_kernel[(hi - lo + 63) // 64, (n + 63) // 64](
        qq,
        kv,
        ww,
        kk,
        ee,
        local,
        hi - lo,
        n,
        h,
        d,
        qq.stride(0),
        qq.stride(1),
        qq.stride(2),
        kv.stride(0),
        kv.stride(1),
        ww.stride(0),
        ww.stride(1),
        local.stride(0),
        local.stride(1),
        True,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=4,
        num_stages=4,
        pipeline="basic",
        scenario="",
    )
    dist.all_gather_into_tensor(result, local, group=group)
    output = result[:m]
    return output
