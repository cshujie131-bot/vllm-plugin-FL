"""Model-specific MLA selection; the general FlagGems-vllm entry stays independent."""

from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.attention.mla_dense_fallback import dense_decode_fallback
from typing import Optional, Tuple
import torch
from flaggems_vllm.ops.flash_mla_with_kvcache import (
    FlashMLASchedMeta,
    flash_mla_with_kvcache as flagos_decode,
)
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.attention.mla4k_long import supported, run

import vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.attention.mla_decode as mla_decode
import vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.attention.mla_decode_b2 as mla_decode_b2
import vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.attention.mla_staged_packing as mla_staged_packing


def try_dense_decode(
    q, kv_cache, block_table, cache_seqlens, out, lse, *, softmax_scale, causal
):
    """Return False for unsupported shapes; no allocation or fallback on failure."""
    if kv_cache.stride(-1) != 1 or block_table.stride(-1) != 1:
        raise NotImplementedError("Dense MLA requires contiguous entry/table columns")
    if not mla_decode.can_use(
        q, kv_cache, block_table, cache_seqlens, out, lse, softmax_scale
    ):
        b, sq, heads, width = q.shape
        dense_decode_fallback(
            q,
            kv_cache,
            block_table,
            cache_seqlens,
            out,
            lse,
            b,
            sq,
            heads,
            width,
            out.shape[-1],
            kv_cache.shape[1],
            softmax_scale,
            causal,
        )
        return True
    if mla_decode_b2.try_run(
        q,
        kv_cache,
        block_table,
        cache_seqlens,
        out,
        lse,
        softmax_scale=softmax_scale,
        causal=causal,
    ):
        return True
    if not mla_staged_packing.try_run(
        q,
        kv_cache,
        block_table,
        cache_seqlens,
        out,
        lse,
        softmax_scale=softmax_scale,
        causal=causal,
    ):
        mla_decode.run(
            q,
            kv_cache,
            block_table,
            cache_seqlens,
            out,
            lse,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    return True


def decode_with_backend(*args, **kwargs):
    return flagos_decode(*args, dense_decode_backend=try_dense_decode, **kwargs)


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # One public entry. The preexisting implementation is the explicit fallback.
    optional_clear = (
        num_splits is None
        and head_dim_v == 512
        and not is_fp8_kvcache
        and causal in (False, True)
        and all(
            x is None
            for x in (
                indices,
                attn_sink,
                extra_k_cache,
                extra_indices_in_kvcache,
                topk_length,
                extra_topk_length,
            )
        )
    )
    if optional_clear and supported(
        q, k_cache, block_table, cache_seqlens, out, softmax_scale
    ):

        def previous(masked_lengths):
            return decode_with_backend(
                q,
                k_cache,
                block_table,
                masked_lengths,
                head_dim_v,
                tile_scheduler_metadata,
                num_splits,
                softmax_scale,
                causal,
                is_fp8_kvcache,
                indices,
                attn_sink,
                extra_k_cache,
                extra_indices_in_kvcache,
                topk_length,
                extra_topk_length,
                out,
            )

        return run(q, k_cache, block_table, cache_seqlens, previous=previous)
    return decode_with_backend(
        q,
        k_cache,
        block_table,
        cache_seqlens,
        head_dim_v,
        tile_scheduler_metadata,
        num_splits,
        softmax_scale,
        causal,
        is_fp8_kvcache,
        indices,
        attn_sink,
        extra_k_cache,
        extra_indices_in_kvcache,
        topk_length,
        extra_topk_length,
        out,
    )
