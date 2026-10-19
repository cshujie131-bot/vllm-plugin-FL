"""Independent selected MLA: bounded streamed QK states and folded PV replay."""

import torch, triton
import triton.language as tl
from vllm_fl.ops.kimi_k3_metax.mla_decode_short import _short_dense_kernel
from vllm_fl.ops.kimi_k3_metax.mla_decode import _merge_splits


@triton.jit
def _qk_stream_kernel(
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
    ProbabilityLow,
    Alpha,
    MAX_CHUNKS: tl.constexpr,
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
    K_TILE: tl.constexpr,
    CAP: tl.constexpr,
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
    guard_length = tl.load(Seq_lens + i_b)
    if (guard_length <= 256) | (guard_length > CAP):
        return

    cur_head = pid_h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_head = cur_head < HQ

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

    cur_batch_seq_len = guard_length
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

        qk = tl.zeros((BLOCK_H, BLOCK_N), tl.float32)
        for kd in range(0, HEAD_DIM_V, K_TILE):
            col = kd + tl.arange(0, K_TILE)
            qa = tl.load(
                Q_ptr
                + i_b * stride_q_b
                + i_sq * stride_q_sq
                + cur_head[:, None] * stride_q_h
                + col[None, :],
                mask_head[:, None] & (col[None, :] < HEAD_DIM_V),
                0,
            )
            kb = tl.load(
                KV_cache + kv_offset[None, :] + col[:, None],
                (col[:, None] < HEAD_DIM_V) & tl.full((1, BLOCK_N), True, tl.int1),
                0,
            )
            qk = tl.dot(qa, kb, qk)

        # Add RoPE contribution
        offs_k_pe = kv_offset[None, :] + offs_d_pe[:, None]
        k_pe = tl.load(KV_cache + offs_k_pe)
        qk += tl.dot(q_pe, k_pe)
        qk *= sm_scale

        # Online softmax update
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        probability_hi = p.to(tl.bfloat16)
        probability_lo = (p - probability_hi.to(tl.float32)).to(tl.bfloat16)
        chunk_slot = (((i_b * SPLITS + split) * HQ + cur_head) * MAX_CHUNKS + i).to(
            tl.int64
        )
        column = tl.arange(0, BLOCK_N)
        tl.store(
            Out + chunk_slot[:, None] * BLOCK_N + column[None, :],
            probability_hi,
            mask_head[:, None],
        )
        tl.store(
            ProbabilityLow + chunk_slot[:, None] * BLOCK_N + column[None, :],
            probability_lo,
            mask_head[:, None],
        )
        tl.store(Alpha + chunk_slot, re_scale, mask_head)
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

        qk = tl.zeros((BLOCK_H, BLOCK_N), tl.float32)
        for kd in range(0, HEAD_DIM_V, K_TILE):
            col = kd + tl.arange(0, K_TILE)
            qa = tl.load(
                Q_ptr
                + i_b * stride_q_b
                + i_sq * stride_q_sq
                + cur_head[:, None] * stride_q_h
                + col[None, :],
                mask_head[:, None] & (col[None, :] < HEAD_DIM_V),
                0,
            )
            kb = tl.load(
                KV_cache + kv_offset[None, :] + col[:, None],
                (col[:, None] < HEAD_DIM_V) & mask_kvsplit[None, :],
                0,
            )
            qk = tl.dot(qa, kb, qk)

        offs_k_pe = kv_offset[None, :] + offs_d_pe[:, None]
        k_pe = tl.load(KV_cache + offs_k_pe, mask=mask_kvsplit[None, :], other=0.0)
        qk += tl.dot(q_pe, k_pe)
        qk *= sm_scale

        qk = tl.where(mask_kvsplit[None, :], qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        probability_hi = p.to(tl.bfloat16)
        probability_lo = (p - probability_hi.to(tl.float32)).to(tl.bfloat16)
        chunk_slot = (
            ((i_b * SPLITS + split) * HQ + cur_head) * MAX_CHUNKS + loop_time
        ).to(tl.int64)
        column = tl.arange(0, BLOCK_N)
        tl.store(
            Out + chunk_slot[:, None] * BLOCK_N + column[None, :],
            probability_hi,
            mask_head[:, None],
        )
        tl.store(
            ProbabilityLow + chunk_slot[:, None] * BLOCK_N + column[None, :],
            probability_lo,
            mask_head[:, None],
        )
        tl.store(Alpha + chunk_slot, re_scale, mask_head)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    # Keep unnormalized accumulators/statistics in FP32 for the merge.
    slot = ((i_b * SQ + i_sq) * SPLITS + split) * HQ + cur_head
    tl.store(PartialMax + slot, e_max, mask_head)
    tl.store(PartialSum + slot, e_sum, mask_head)


@triton.jit
def _long_fallback_kernel(
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
    CAP: tl.constexpr,
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
    guard_length = tl.load(Seq_lens + i_b)
    if guard_length <= CAP:
        return

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

    cur_batch_seq_len = guard_length
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
def _pv_fold_kernel(
    Cache,
    stride_kv_token,
    stride_kv_block,
    Table,
    stride_table,
    Lengths,
    PH,
    PL,
    A,
    Partial,
    H: tl.constexpr,
    SPLITS: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,
    BN: tl.constexpr,
    FOLD: tl.constexpr,
    CAP: tl.constexpr,
):
    b = tl.program_id(0)
    split = tl.program_id(1)
    length = tl.load(Lengths + b)
    if (length <= 256) | (length > CAP):
        return
    row = tl.arange(0, 16)
    head = row % (16 // FOLD)
    part = row // (16 // FOLD)
    valid = head < H
    d = tl.arange(0, 512)
    n = tl.arange(0, BN)
    blocks = tl.cdiv(tl.cdiv(length, BN), SPLITS)
    first = split * blocks * BN
    end = tl.minimum((split + 1) * blocks * BN, length)
    count = tl.maximum(end - first, 0)
    chunks = tl.cdiv(count, BN)
    acc = tl.zeros((16, 512 // FOLD), tl.float32)
    for step in range(chunks):
        pos = first + step * BN + n
        mask = pos < length
        page = tl.load(Table + b * stride_table + pos // 256, mask, 0).to(tl.int64)
        offset = page * stride_kv_block + (pos % 256) * stride_kv_token
        v = tl.load(Cache + offset[:, None] + d[None, :], mask[:, None], 0)
        slot = (((b * SPLITS + split) * H + head) * MAX_CHUNKS + step).to(tl.int64)
        phi = tl.load(PH + slot[:, None] * BN + n[None, :], valid[:, None], 0)
        plo = tl.load(PL + slot[:, None] * BN + n[None, :], valid[:, None], 0)
        alpha = tl.load(A + slot, valid, 0)
        slice_id = tl.arange(0, FOLD)
        smask = part[:, None, None] == slice_id[None, None, :]
        hp = tl.reshape(
            tl.where(smask, phi[:, :, None], 0).to(tl.bfloat16), (16, BN * FOLD)
        )
        lp = tl.reshape(
            tl.where(smask, plo[:, :, None], 0).to(tl.bfloat16), (16, BN * FOLD)
        )
        vp = tl.reshape(v, (BN * FOLD, 512 // FOLD))
        acc *= alpha[:, None]
        acc += tl.dot(hp, vp)
        acc += tl.dot(lp, vp)
    slot = (b * SPLITS + split) * H + head
    cols = part[:, None] * (512 // FOLD) + tl.arange(0, 512 // FOLD)[None, :]
    tl.store(Partial + slot[:, None] * 512 + cols, acc, valid[:, None])


def try_run(q, cache, table, lengths, out, lse, *, softmax_scale, causal):
    """Write caller outputs for the measured H3/SQ1 layout; False leaves buffers untouched."""
    if q.ndim != 4 or cache.ndim != 4 or table.ndim != 2 or lengths.ndim != 1:
        return False
    b, sq, h, d = q.shape
    if not (1 <= b <= 64 and b != 2 and (sq, h, d) == (1, 3, 576)):
        return False
    if (
        cache.shape[1:] != (256, 1, 576)
        or table.shape[0] != b
        or not 1 <= table.shape[1] <= 272
    ):
        return False
    if lengths.shape != (b,) or out.shape != (b, 1, 3, 512) or lse.shape != (b, 3, 1):
        return False
    if (
        q.dtype != torch.bfloat16
        or cache.dtype != q.dtype
        or out.dtype != q.dtype
        or lse.dtype != torch.float32
    ):
        return False
    if table.dtype != torch.int32 or lengths.dtype != torch.int32:
        return False
    if not q.is_cuda or any(
        t.device != q.device for t in (cache, table, lengths, out, lse)
    ):
        return False
    if q.stride(-1) != 1 or cache.stride(-1) != 1 or table.stride(-1) != 1:
        return False
    if (
        not lengths.is_contiguous()
        or not out.is_contiguous()
        or not lse.is_contiguous()
    ):
        return False
    if softmax_scale != 192**-0.5 or causal not in (False, True):
        return False
    splits = 16 if b >= 16 else 32
    cap = min(4096, table.shape[1] * 256)
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
        h,
        576,
        512,
        256,
        causal,
        16,
        16,
        SHORT_LIMIT=256,
        num_warps=4,
        num_stages=1,
    )
    max_chunks = triton.cdiv(triton.cdiv(cap, 16), splits)
    ph = torch.empty(
        (b, splits, h, max_chunks, 16), dtype=torch.bfloat16, device=q.device
    )
    pl = torch.empty_like(ph)
    alpha = torch.empty(
        (b, splits, h, max_chunks), dtype=torch.float32, device=q.device
    )
    pm = torch.empty((b, splits, h), dtype=torch.float32, device=q.device)
    ps = torch.empty_like(pm)
    partial = torch.empty((b, splits, h, 512), dtype=torch.float32, device=q.device)
    _qk_stream_kernel[(1, b, splits)](
        q,
        *q.stride()[:3],
        cache,
        cache.stride(1),
        cache.stride(0),
        table,
        table.stride(0),
        lengths,
        ph,
        0,
        0,
        0,
        lse,
        pm,
        ps,
        pl,
        alpha,
        max_chunks,
        splits,
        0,
        0,
        softmax_scale,
        1,
        h,
        576,
        512,
        256,
        causal,
        16,
        16,
        64,
        cap,
        num_warps=2,
        num_stages=1,
    )
    _pv_fold_kernel[(b, splits)](
        cache,
        cache.stride(1),
        cache.stride(0),
        table,
        table.stride(0),
        lengths,
        ph,
        pl,
        alpha,
        partial,
        h,
        splits,
        max_chunks,
        16,
        4,
        cap,
        num_warps=2,
        num_stages=1,
    )
    _long_fallback_kernel[(1, b, splits)](
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
        pm,
        ps,
        splits,
        0,
        0,
        softmax_scale,
        1,
        h,
        576,
        512,
        256,
        causal,
        16,
        16,
        cap,
        num_warps=1,
        num_stages=1,
    )
    _merge_splits[(b, h)](
        partial,
        pm,
        ps,
        out,
        lse,
        lengths,
        b,
        h,
        512,
        splits,
        triton.next_power_of_2(splits),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return True
