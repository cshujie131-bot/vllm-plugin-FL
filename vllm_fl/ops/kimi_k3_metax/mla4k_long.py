"""K3 long-context MLA: bounded QK lifetime, staged PV and FP32 split merge."""

from flag_gems import runtime

# SPDX-License-Identifier: Apache-2.0
import torch, triton
import triton.language as tl


@triton.jit
def _qk_long_kernel(
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
    COMPENSATE: tl.constexpr,
    MIN_LENGTH: tl.constexpr = 256,
    SHORT_SPLITS: tl.constexpr = 0,
    SHORT_LIMIT: tl.constexpr = 4096,
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
    if (guard_length <= MIN_LENGTH) | (guard_length > CAP):
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

    if SHORT_SPLITS > 0:
        active_splits = tl.where(cur_batch_seq_len <= SHORT_LIMIT, SHORT_SPLITS, SPLITS)
        if split >= active_splits:
            return
        blocks_per_split = tl.where(
            cur_batch_seq_len <= SHORT_LIMIT,
            tl.cdiv(tl.cdiv(cur_batch_seq_len, BLOCK_N), SHORT_SPLITS),
            tl.cdiv(tl.cdiv(cur_batch_seq_len, BLOCK_N), SPLITS),
        )
    else:
        active_splits = (
            tl.where(cur_batch_seq_len <= 5120, 128, SPLITS)
            if SPLITS == 256
            else SPLITS
        )
        blocks_per_split = tl.cdiv(tl.cdiv(cur_batch_seq_len, BLOCK_N), active_splits)
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
        probability_lo = (
            (p - probability_hi.to(tl.float32)).to(tl.bfloat16)
            if COMPENSATE
            else tl.full((BLOCK_H, BLOCK_N), 0, tl.bfloat16)
        )
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
        probability_lo = (
            (p - probability_hi.to(tl.float32)).to(tl.bfloat16)
            if COMPENSATE
            else tl.full((BLOCK_H, BLOCK_N), 0, tl.bfloat16)
        )
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
def _pv_long_kernel(
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
    COMPENSATE: tl.constexpr,
    VT: tl.constexpr,
    FUSED: tl.constexpr,
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
    if VT == 512 // FOLD:
        d = tl.arange(0, 512)
    else:
        d = tl.reshape(
            tl.arange(0, FOLD)[:, None] * (512 // FOLD)
            + tl.program_id(2) * VT
            + tl.arange(0, VT)[None, :],
            (FOLD * VT,),
        )
    n = tl.arange(0, BN)
    active_splits = tl.where(length <= 5120, 128, SPLITS) if SPLITS == 256 else SPLITS
    blocks = tl.cdiv(tl.cdiv(length, BN), active_splits)
    first = split * blocks * BN
    end = tl.minimum((split + 1) * blocks * BN, length)
    count = tl.maximum(end - first, 0)
    chunks = tl.cdiv(count, BN)
    acc = tl.zeros((16, VT), tl.float32)
    for step in range(chunks):
        pos = first + step * BN + n
        mask = pos < length
        page = tl.load(Table + b * stride_table + pos // 256, mask, 0).to(tl.int64)
        offset = page * stride_kv_block + (pos % 256) * stride_kv_token
        v = tl.load(Cache + offset[:, None] + d[None, :], mask[:, None], 0)
        slot = (((b * SPLITS + split) * H + head) * MAX_CHUNKS + step).to(tl.int64)
        phi = tl.load(PH + slot[:, None] * BN + n[None, :], valid[:, None], 0)
        alpha = tl.load(A + slot, valid, 0)
        slice_id = tl.arange(0, FOLD)
        smask = part[:, None, None] == slice_id[None, None, :]
        hp = tl.reshape(
            tl.where(smask, phi[:, :, None], 0).to(tl.bfloat16), (16, BN * FOLD)
        )
        vp = tl.reshape(v, (BN * FOLD, VT))
        acc *= alpha[:, None]
        if FUSED:
            acc = tl.dot(hp, vp, acc)
        else:
            acc += tl.dot(hp, vp)
        if COMPENSATE:
            plo = tl.load(PL + slot[:, None] * BN + n[None, :], valid[:, None], 0)
            lp = tl.reshape(
                tl.where(smask, plo[:, :, None], 0).to(tl.bfloat16), (16, BN * FOLD)
            )
            if FUSED:
                acc = tl.dot(lp, vp, acc)
            else:
                acc += tl.dot(lp, vp)
    slot = (b * SPLITS + split) * H + head
    cols = (
        part[:, None] * (512 // FOLD)
        + tl.program_id(2) * VT
        + tl.arange(0, VT)[None, :]
    )
    tl.store(Partial + slot[:, None] * 512 + cols, acc, valid[:, None])


@triton.jit
def _merge_long_kernel(
    P,
    M,
    S,
    O,
    L,
    Lengths,
    OriginalLengths,
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    SPLITS: tl.constexpr,
    RS: tl.constexpr,
    VT: tl.constexpr,
    MIN_LENGTH: tl.constexpr = 256,
    SHORT_SPLITS: tl.constexpr = 0,
    SHORT_LIMIT: tl.constexpr = 4096,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    length = tl.load(Lengths + b)
    if length <= MIN_LENGTH:
        if tl.load(OriginalLengths + b) == 0:
            zero_d = tl.program_id(2) * VT + tl.arange(0, VT)
            tl.store(O + (b * H + h) * V + zero_d, 0, zero_d < V)
            if tl.program_id(2) == 0:
                tl.store(L + b * H + h, -float("inf"))
        return
    active_splits = (
        tl.where(length <= SHORT_LIMIT, SHORT_SPLITS, SPLITS)
        if SHORT_SPLITS > 0
        else SPLITS
    )
    split = tl.arange(0, RS)
    d = tl.program_id(2) * VT + tl.arange(0, VT)
    slot = (b * SPLITS + split) * H + h
    maximum = tl.load(M + slot, split < active_splits, -float("inf"))
    denom = tl.load(S + slot, split < active_splits, 0)
    overall = tl.max(maximum, 0)
    scale = tl.where(denom > 0, tl.exp(maximum - overall), 0)
    part = tl.load(
        P + slot[:, None] * V + d[None, :],
        (split[:, None] < active_splits) & (d[None, :] < V),
        0,
    )
    numerator = tl.sum(part * scale[:, None], 0)
    total = tl.sum(denom * scale, 0)
    result = tl.where(total > 0, numerator / total, 0)
    tl.store(O + (b * H + h) * V + d, result.to(tl.bfloat16), d < V)
    if tl.program_id(2) == 0:
        tl.store(
            L + b * H + h, tl.where(total > 0, overall + tl.log(total), -float("inf"))
        )


@triton.jit
def _partition_lengths(
    L, Old, New, B: tl.constexpr, CAP: tl.constexpr, R: tl.constexpr
):
    i = tl.arange(0, R)
    n = tl.load(L + i, i < B, 0)
    use = (n > 4096) & (n <= CAP)
    tl.store(Old + i, tl.where(use, 0, n), i < B)
    tl.store(New + i, tl.where(use, n, 0), i < B)


def supported(q, cache, table, lengths, out, scale):
    if runtime.device.vendor_name != "metax":
        return False
    if not all(isinstance(x, torch.Tensor) for x in (q, cache, table, lengths)):
        return False
    if q.ndim != 4 or cache.ndim != 4 or table.ndim != 2 or lengths.ndim != 1:
        return False
    b, sq, h, d = q.shape
    if not (1 <= b <= 64 and (sq, h, d) == (1, 3, 576)):
        return False
    if (
        tuple(cache.shape[1:]) != (256, 1, 576)
        or table.shape[0] != b
        or not 17 <= table.shape[1] <= 272
    ):
        return False
    if lengths.shape != (b,) or q.dtype != torch.bfloat16 or cache.dtype != q.dtype:
        return False
    if table.dtype != torch.int32 or lengths.dtype != torch.int32:
        return False
    if not q.is_cuda or any(x.device != q.device for x in (cache, table, lengths)):
        return False
    if (
        q.stride(-1) != 1
        or cache.stride(-1) != 1
        or table.stride(-1) != 1
        or not lengths.is_contiguous()
    ):
        return False
    if out is not None and (
        out.shape != (b, 1, 3, 512)
        or out.dtype != q.dtype
        or out.device != q.device
        or not out.is_contiguous()
    ):
        return False
    if out is not None:
        base = out.untyped_storage().data_ptr()
        if any(
            base == x.untyped_storage().data_ptr() for x in (q, cache, table, lengths)
        ):
            return False
    return scale == 192**-0.5


def run(q, cache, table, lengths, *, previous, record=None):
    """Only4097..69632 rows use the staged math; previous dispatcher handles other rows.
    previous(masked_lengths) must execute the canonical prior dispatch and return(out,lse).
    There is no host read of lengths, no extra output copy, and no new registration.
    """
    b, sq, h, d = q.shape
    cap = min(69632, table.shape[1] * 256)
    splits = 16 if b > 33 else 32 if b > 16 else 64 if b > 8 else 128 if b > 2 else 256
    q_warps = 2
    pv_warps = 2
    k_tile = 64
    compensated = True
    old_lengths = torch.empty_like(lengths)
    new_lengths = torch.empty_like(lengths)
    _partition_lengths[(1,)](
        lengths,
        old_lengths,
        new_lengths,
        b,
        cap,
        triton.next_power_of_2(b),
        num_warps=1,
    )
    out, lse = previous(old_lengths)
    max_chunks = max(
        triton.cdiv(triton.cdiv(cap, 16), splits),
        triton.cdiv(triton.cdiv(min(cap, 5120), 16), 128) if splits == 256 else 0,
    )
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
    qo = _qk_long_kernel[(1, b, splits)](
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
        False,
        16,
        16,
        k_tile,
        cap,
        compensated,
        num_warps=q_warps,
        num_stages=1,
    )
    po = _pv_long_kernel[(b, splits, 1)](
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
        compensated,
        128,
        compensated,
        num_warps=pv_warps,
        num_stages=1,
    )
    mo = _merge_long_kernel[(b, h, 4)](
        partial,
        pm,
        ps,
        out,
        lse,
        new_lengths,
        lengths,
        b,
        h,
        512,
        splits,
        triton.next_power_of_2(splits),
        128,
        num_warps=4,
        enable_fp_fusion=False,
    )
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
            splits=splits,
            cap=cap,
            compensated=compensated,
            fused_accumulator=compensated,
            target_workspace_bytes=sum(
                x.numel() * x.element_size()
                for x in (ph, pl, alpha, pm, ps, partial, old_lengths, new_lengths)
            ),
            output_copy=False,
            range_guard="4096<L<=cap;all other rows previous dispatch;empty0/-inf",
        )
    return out, lse
