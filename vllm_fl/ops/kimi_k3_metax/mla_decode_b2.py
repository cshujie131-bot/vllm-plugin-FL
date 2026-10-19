"""Selective B2 split-K decode; old arithmetic retained outside257..4096KV."""

import torch, triton
import triton.language as tl
from vllm_fl.ops.kimi_k3_metax.mla_staged_packing import _pv_fold_kernel


@triton.jit
def _qk_b2_scores_kernel(
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
    Scores,
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
        score_offsets = (i_b * HQ + cur_head[:, None]) * CAP + offs_n[None, :]
        tl.store(
            Scores + score_offsets,
            qk,
            mask_head[:, None] & (offs_n[None, :] < cur_batch_seq_len),
        )

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
        score_offsets = (i_b * HQ + cur_head[:, None]) * CAP + offs_n[None, :]
        tl.store(
            Scores + score_offsets,
            qk,
            mask_head[:, None] & (offs_n[None, :] < cur_batch_seq_len),
        )

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
def _lse_from_scores(Scores, Lengths, LSE, H: tl.constexpr, CAP: tl.constexpr):
    b = tl.program_id(0)
    length = tl.load(Lengths + b)
    if (length <= 256) | (length > CAP):
        return
    heads = tl.arange(0, 16)
    n = tl.arange(0, 16)
    valid = heads < H
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denominator = tl.zeros((16,), tl.float32)
    for step in range(tl.cdiv(length, 16)):
        pos = step * 16 + n
        qk = tl.load(
            Scores + (b * H + heads[:, None]) * CAP + pos[None, :],
            valid[:, None] & (pos[None, :] < length),
            -float("inf"),
        )
        next_max = tl.maximum(tl.max(qk, 1), maximum)
        rescale = tl.exp(maximum - next_max)
        p = tl.exp(qk - next_max[:, None])
        denominator = denominator * rescale + tl.sum(p, 1)
        maximum = next_max
    tl.store(LSE + b * H + heads, maximum + tl.math.log(denominator), valid)


from vllm_fl.ops.kimi_k3_metax import mla_decode


@triton.jit
def _partition_b2_lengths(L, Old, New, CAP: tl.constexpr):
    i = tl.arange(0, 2)
    n = tl.load(L + i)
    use = (n > 256) & (n <= CAP)
    tl.store(Old + i, tl.where(use, 0, n))
    tl.store(New + i, tl.where(use, n, 0))


def try_run(q, cache, table, lengths, out, lse, *, softmax_scale, causal, record=None):
    # Called behind the existing K3 dtype/layout/scale guard. All other batches
    # retain the exact previous dispatcher and issue no additional GPU launches.
    if q.shape != (2, 1, 3, 576) or table.shape[1] <= 1 or softmax_scale != 192**-0.5:
        return False
    if out.untyped_storage().data_ptr() in {
        x.untyped_storage().data_ptr() for x in (q, cache, table, lengths)
    }:
        return False
    splits = 32
    cap = min(4096, table.shape[1] * 256)
    b = 2
    h = 3
    old_lengths = torch.empty_like(lengths)
    new_lengths = torch.empty_like(lengths)
    part = _partition_b2_lengths[(1,)](
        lengths, old_lengths, new_lengths, cap, num_warps=1
    )
    mla_decode.run(
        q,
        cache,
        table,
        old_lengths,
        out,
        lse,
        softmax_scale=softmax_scale,
        causal=causal,
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
    scores = torch.empty((b, h, cap), dtype=torch.float32, device=q.device)
    qo = _qk_b2_scores_kernel[(1, b, splits)](
        q,
        *q.stride()[:3],
        cache,
        cache.stride(1),
        cache.stride(0),
        table,
        table.stride(0),
        new_lengths,
        ph,
        0,
        0,
        0,
        lse,
        pm,
        ps,
        pl,
        alpha,
        scores,
        max_chunks,
        splits,
        0,
        0,
        192**-0.5,
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
    po = _pv_fold_kernel[(b, splits)](
        cache,
        cache.stride(1),
        cache.stride(0),
        table,
        table.stride(0),
        new_lengths,
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
    mo = mla_decode._merge_splits[(b, h)](
        partial,
        pm,
        ps,
        out,
        lse,
        new_lengths,
        b,
        h,
        512,
        splits,
        triton.next_power_of_2(splits),
        num_warps=4,
        enable_fp_fusion=False,
    )
    # Reuse parallel FP32 scores; only the light sequential statistics remain.
    slse = _lse_from_scores[(b,)](scores, new_lengths, lse, h, cap, num_warps=4)
    if record is not None:

        def md(k):
            return {
                "n_regs": k.n_regs,
                "n_spills": k.n_spills,
                "shared": k.metadata.shared,
            }

        record.update(
            qk=md(qo),
            pv=md(po),
            merge=md(mo),
            partition=md(part),
            sequential_lse=md(slse),
            splits=splits,
            cap=cap,
            workspace_bytes=sum(
                x.numel() * x.element_size()
                for x in (
                    old_lengths,
                    new_lengths,
                    ph,
                    pl,
                    alpha,
                    pm,
                    ps,
                    partial,
                    scores,
                )
            ),
        )
    return True
