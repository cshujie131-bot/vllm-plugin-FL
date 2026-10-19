from vllm_fl import dsv4_config as _dsv4
# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems implementations for DeepseekV4 attention operators.
"""

import torch

from vllm_fl.ops.deepseek_v4_metax.decode import (
    FlashMLASchedMeta,
    flash_mla_with_kvcache,
)

from flaggems_vllm import combine_topk_swa_indices as _fgv_combine_topk_swa_indices

from flaggems_vllm import (
    dequantize_and_gather_k_cache as _fgv_dequantize_and_gather_k_cache,
)

from flaggems_vllm import fused_q_kv_rmsnorm as _fgv_fused_q_kv_rmsnorm

from vllm_fl.ops.dsv4_headpack import sparse_prefill as _framework_sparse_prefill


# ==================== Sparse Attention Indexer Ops ====================


def combine_topk_swa_indices_flaggems(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> None:
    """FlagGems implementation of combine_topk_swa_indices."""
    return _fgv_combine_topk_swa_indices(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        topk,
        M,
        N,
    )


def dequantize_and_gather_k_cache_flaggems(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    """FlagGems implementation of dequantize_and_gather_k_cache."""
    _fgv_dequantize_and_gather_k_cache(
        out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
    )


def fused_q_kv_rmsnorm_flaggems(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlagGems implementation of fused_q_kv_rmsnorm."""
    return _fgv_fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)


def flash_mla_with_kvcache_flaggems(
    q,
    k_cache,
    block_table,
    head_dim_v,
    tile_scheduler_metadata,
    cache_seqlens,
    is_fp8_kvcache,
    indices,
    topk_length,
    softmax_scale,
    attn_sink,
    extra_k_cache,
    extra_indices_in_kvcache,
    extra_topk_length,
    out,
):
    original_meta = tile_scheduler_metadata
    if isinstance(original_meta, FlashMLASchedMeta):
        flaggems_meta = original_meta
    else:
        flaggems_meta = FlashMLASchedMeta(
            have_initialized=original_meta.have_initialized,
            config=original_meta.config,
            tile_scheduler_metadata=original_meta.tile_scheduler_metadata,
            num_splits=original_meta.num_splits,
        )
    result = flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=flaggems_meta,
        is_fp8_kvcache=is_fp8_kvcache,
        indices=indices,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices_in_kvcache,
        extra_topk_length=extra_topk_length,
        out=out,
        real_num_heads=_dsv4.current.real_query_heads,
    )
    if flaggems_meta is not original_meta:
        original_meta.have_initialized = flaggems_meta.have_initialized
        original_meta.config = flaggems_meta.config
        original_meta.tile_scheduler_metadata = flaggems_meta.tile_scheduler_metadata
        original_meta.num_splits = flaggems_meta.num_splits
    return result


def flash_mla_sparse_fwd_flaggems(
    q, kv, indices, sm_scale, attn_sink, topk_length, out
):
    (out_tmp, max_logits, lse) = _framework_sparse_prefill(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
    out.copy_(out_tmp)
    return (out, max_logits, lse)
