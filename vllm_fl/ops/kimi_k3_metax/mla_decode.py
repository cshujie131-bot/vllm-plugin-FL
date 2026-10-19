"""Independent dense MLA split-KV prototype, frozen BF16 products and FP32 merge."""

from flag_gems import runtime

import torch, triton
from vllm_fl.ops.kimi_k3_metax.mla_decode_short import _short_dense_kernel
import triton.language as tl


@triton.jit
def _split_decode_comp_kernel(
    Q_ptr,
    stride_q_b,
    stride_q_sq,
    stride_q_h,
    KV_cache,
    stride_kv_bs,
    stride_kv_block,
    Block_table,
    stride_bt_b,
    Seq_lens,
    Out,
    stride_o_b,
    stride_o_sq,
    stride_o_h,
    LSE,
    PartialMax,
    PartialSum,
    SPLITS: tl.constexpr,
    stride_lse_b,
    stride_lse_h,
    sm_scale,
    SQ,
    HQ: tl.constexpr,
    DQK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HAVE_CAUSAL: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Dense decode kernel with paged attention and online softmax.
    Grid: (ceil(HQ / BLOCK_H), batch_size * seq_q)
    """
    split = tl.program_id(2)
    pid_h_block = tl.program_id(0)
    pid_b_sq = tl.program_id(1)
    i_b = pid_b_sq // SQ
    i_sq = pid_b_sq % SQ

    cur_head = pid_h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_head = cur_head < HQ

    # Load Q: NoPE part [BLOCK_H, HEAD_DIM_V] and RoPE part [BLOCK_H, DQK-HEAD_DIM_V]
    offs_d_nope = tl.arange(0, HEAD_DIM_V)
    offs_q_nope = (
        i_b * stride_q_b
        + i_sq * stride_q_sq
        + cur_head[:, None] * stride_q_h
        + offs_d_nope[None, :]
    )
    q_nope = tl.load(Q_ptr + offs_q_nope, mask=mask_head[:, None], other=0.0)

    offs_d_pe = tl.arange(HEAD_DIM_V, DQK)
    offs_q_pe = (
        i_b * stride_q_b
        + i_sq * stride_q_sq
        + cur_head[:, None] * stride_q_h
        + offs_d_pe[None, :]
    )
    q_pe = tl.load(Q_ptr + offs_q_pe, mask=mask_head[:, None], other=0.0)

    # Online softmax accumulators
    e_max = tl.full([BLOCK_H], value=float("-inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM_V], dtype=tl.float32)

    cur_batch_seq_len = tl.load(Seq_lens + i_b)
    if cur_batch_seq_len <= 256:
        return
    Block_table += i_b * stride_bt_b

    blocks_per_split = tl.cdiv(tl.cdiv(cur_batch_seq_len, BLOCK_N), SPLITS)
    first = split * blocks_per_split * BLOCK_N
    end = tl.minimum((split + 1) * blocks_per_split * BLOCK_N, cur_batch_seq_len)
    count = tl.maximum(end - first, 0)
    offs_n = first + tl.arange(0, BLOCK_N)
    loop_time = count // BLOCK_N
    remainder = count % BLOCK_N

    for i in range(0, loop_time):
        kv_page_number = tl.load(Block_table + offs_n // PAGE_SIZE)
        kv_offset = (
            kv_page_number.to(tl.int64) * stride_kv_block
            + (offs_n % PAGE_SIZE).to(tl.int64) * stride_kv_bs
        )

        # Load V (NoPE part)
        offs_v_c = kv_offset[:, None] + offs_d_nope[None, :]
        v_c = tl.load(KV_cache + offs_v_c)
        k_c = tl.trans(v_c)

        # QK = q_nope @ k_nope^T
        qk = tl.dot(q_nope, k_c)

        # Add RoPE contribution
        offs_k_pe = kv_offset[None, :] + offs_d_pe[:, None]
        k_pe = tl.load(KV_cache + offs_k_pe)
        qk += tl.dot(q_pe, k_pe)
        qk *= sm_scale

        # Online softmax update
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        probability_hi = p.to(v_c.dtype)
        probability_lo = (p - probability_hi.to(tl.float32)).to(v_c.dtype)
        acc += tl.dot(probability_hi, v_c)
        acc += tl.dot(probability_lo, v_c)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max
        offs_n += BLOCK_N

    if remainder:
        mask_kvsplit = offs_n < cur_batch_seq_len
        kv_page_number = tl.load(
            Block_table + offs_n // PAGE_SIZE, mask=mask_kvsplit, other=0
        )
        kv_offset = (
            kv_page_number.to(tl.int64) * stride_kv_block
            + (offs_n % PAGE_SIZE).to(tl.int64) * stride_kv_bs
        )

        offs_v_c = kv_offset[:, None] + offs_d_nope[None, :]
        v_c = tl.load(KV_cache + offs_v_c, mask=mask_kvsplit[:, None], other=0.0)
        k_c = tl.trans(v_c)

        qk = tl.dot(q_nope, k_c)

        offs_k_pe = kv_offset[None, :] + offs_d_pe[:, None]
        k_pe = tl.load(KV_cache + offs_k_pe, mask=mask_kvsplit[None, :], other=0.0)
        qk += tl.dot(q_pe, k_pe)
        qk *= sm_scale

        qk = tl.where(mask_kvsplit[None, :], qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        probability_hi = p.to(v_c.dtype)
        probability_lo = (p - probability_hi.to(tl.float32)).to(v_c.dtype)
        acc += tl.dot(probability_hi, v_c)
        acc += tl.dot(probability_lo, v_c)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    # Keep unnormalized accumulators/statistics in FP32 for the merge.
    slot = ((i_b * SQ + i_sq) * SPLITS + split) * HQ + cur_head
    tl.store(
        Out + slot[:, None] * HEAD_DIM_V + offs_d_nope[None, :], acc, mask_head[:, None]
    )
    tl.store(PartialMax + slot, e_max, mask_head)
    tl.store(PartialSum + slot, e_sum, mask_head)


@triton.jit
def _merge_splits(
    P,
    M,
    S,
    O,
    L,
    Lengths,
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    SPLITS: tl.constexpr,
    RS: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    length = tl.load(Lengths + b)
    if length <= 256:
        return
    split = tl.arange(0, RS)
    d = tl.arange(0, V)
    slot = (b * SPLITS + split) * H + h
    maximum = tl.load(M + slot, split < SPLITS, -float("inf"))
    denom = tl.load(S + slot, split < SPLITS, 0)
    overall = tl.max(maximum, 0)
    scale = tl.where(denom > 0, tl.exp(maximum - overall), 0)
    part = tl.load(P + slot[:, None] * V + d[None, :], split[:, None] < SPLITS, 0)
    numerator = tl.sum(part * scale[:, None], 0)
    total = tl.sum(denom * scale, 0)
    result = tl.where(total > 0, numerator / total, 0)
    tl.store(O + (b * H + h) * V + d, result.to(tl.bfloat16))
    tl.store(L + b * H + h, tl.where(total > 0, overall + tl.log(total), -float("inf")))


def can_use(q, cache, table, lengths, out, lse, scale):
    """Metadata-only guard; unsupported calls retain the existing dispatcher."""
    return (
        runtime.device.vendor_name == "metax"
        and q.ndim == 4
        and q.shape[0]
        in (
            1,
            2,
            3,
            4,
            5,
            7,
            8,
            12,
            16,
            20,
            24,
            28,
            32,
            36,
            40,
            44,
            48,
            52,
            55,
            56,
            60,
            61,
            64,
        )
        and tuple(q.shape[1:]) == (1, 3, 576)
        and tuple(cache.shape[1:]) == (256, 1, 576)
        and q.dtype == cache.dtype == torch.bfloat16
        and table.dtype == lengths.dtype == torch.int32
        and q.is_cuda
        and all(x.device == q.device for x in (cache, table, lengths, out, lse))
        and table.ndim == 2
        and table.shape[0] == q.shape[0]
        and lengths.shape == (q.shape[0],)
        and lengths.is_contiguous()
        and q.stride(-1) == cache.stride(-1) == table.stride(-1) == 1
        and out.shape == (q.shape[0], 1, 3, 512)
        and out.dtype == torch.bfloat16
        and lse.shape == (q.shape[0], 3, 1)
        and lse.dtype == torch.float32
        and out.is_contiguous()
        and lse.is_contiguous()
        and abs(scale - 192**-0.5) <= 1e-15
    )


def run(q, cache, table, lengths, out, lse, *, softmax_scale, causal):
    """Write caller outputs; retain original B2/short-row arithmetic."""
    b = q.shape[0]
    _short_dense_kernel[(1, b)](
        q,
        *q.stride()[:3],
        cache,
        cache.stride(1),
        cache.stride(0),
        table,
        table.stride(0),
        lengths,
        out,
        *out.stride()[:3],
        lse,
        lse.stride(0),
        lse.stride(1),
        softmax_scale,
        1,
        3,
        576,
        512,
        256,
        causal,
        16,
        16,
        SHORT_LIMIT=table.shape[1] * 256 if b == 2 else 256,
        num_warps=4,
        num_stages=1,
    )
    if b == 2:
        return
    splits = 16 if b >= 16 else 32
    partial = torch.empty((b, splits, 3, 512), device=q.device, dtype=torch.float32)
    maximum = torch.empty((b, splits, 3), device=q.device, dtype=torch.float32)
    denominator = torch.empty_like(maximum)
    _split_decode_comp_kernel[(1, b, splits)](
        q,
        *q.stride()[:3],
        cache,
        cache.stride(1),
        cache.stride(0),
        table,
        table.stride(0),
        lengths,
        partial,
        0,
        0,
        0,
        lse,
        maximum,
        denominator,
        splits,
        0,
        0,
        softmax_scale,
        1,
        3,
        576,
        512,
        256,
        causal,
        16,
        16,
        num_warps=1,
        num_stages=1,
    )
    _merge_splits[(b, 3)](
        partial,
        maximum,
        denominator,
        out,
        lse,
        lengths,
        b,
        3,
        512,
        splits,
        triton.next_power_of_2(splits),
        num_warps=4,
        enable_fp_fusion=False,
    )
