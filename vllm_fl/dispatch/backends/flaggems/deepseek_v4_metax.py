# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems backend implementation.

This backend provides operator implementations using the FlagGems library.
"""

from __future__ import annotations


import torch

from .flaggems import FlagGemsBackend

from .impl.deepseek_v4_metax import activation as _impl_activation
from .impl.deepseek_v4_metax import deepseek_v4_attn as _impl_deepseek_v4_attn
from .impl.deepseek_v4_metax import deepseek_v4_ops as _impl_deepseek_v4_ops
from .impl.deepseek_v4_metax import fused_moe as _impl_fused_moe
from .impl.deepseek_v4_metax import mhc as _impl_mhc
from .impl.deepseek_v4_metax import normalization as _impl_normalization
from .impl.deepseek_v4_metax import rotary as _impl_rotary
from .impl.deepseek_v4_metax import router_gemm as _impl_router_gemm


class DeepseekV4MetaXBackend(FlagGemsBackend):
    # ==================== Operator Implementations ====================

    router_gemm_bf16_fp32 = staticmethod(
        _impl_router_gemm.router_gemm_bf16_fp32_flaggems
    )

    silu_and_mul = staticmethod(_impl_activation.silu_and_mul_flaggems)

    gelu_and_mul = staticmethod(_impl_activation.gelu_and_mul_flaggems)

    def silu_and_mul_with_clamp(
        self, x: torch.Tensor, swiglu_limit: float, swiglu_limit_tensor: torch.Tensor
    ) -> torch.Tensor:
        return _impl_activation.silu_and_mul_with_clamp_flaggems(x, swiglu_limit_tensor)

    rms_norm = staticmethod(_impl_normalization.rms_norm_flaggems)

    rotary_embedding = staticmethod(_impl_rotary.rotary_embedding_flaggems)

    moe_align_block_size = staticmethod(_impl_fused_moe.moe_align_block_size_flaggems)

    moe_sum = staticmethod(_impl_fused_moe.moe_sum_flaggems)

    fused_topk_bias = staticmethod(_impl_fused_moe.fused_topk_bias_flaggems)

    topk_softmax = staticmethod(_impl_fused_moe.topk_softmax_flaggems)

    invoke_fused_moe_triton_kernel = staticmethod(
        _impl_fused_moe.invoke_fused_moe_triton_kernel_flaggems
    )

    grouped_topk = staticmethod(_impl_fused_moe.grouped_topk_flaggems)

    mhc_pre = staticmethod(_impl_mhc.mhc_pre_flaggems)

    mhc_post = staticmethod(_impl_mhc.mhc_post_flaggems)

    hc_head_fused_kernel = staticmethod(_impl_mhc.hc_head_fused_kernel_flaggems)

    deepseek_v4_fp8_einsum = staticmethod(
        _impl_deepseek_v4_attn.deepseek_v4_fp8_einsum_flaggems
    )

    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert = staticmethod(
        _impl_deepseek_v4_attn.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_flaggems
    )

    combine_topk_swa_indices = staticmethod(
        _impl_deepseek_v4_attn.combine_topk_swa_indices_flaggems
    )

    compute_global_topk_indices_and_lens = staticmethod(
        _impl_deepseek_v4_attn.compute_global_topk_indices_and_lens_flaggems
    )

    dequantize_and_gather_k_cache = staticmethod(
        _impl_deepseek_v4_attn.dequantize_and_gather_k_cache_flaggems
    )

    fused_indexer_q_rope_quant = staticmethod(
        _impl_deepseek_v4_attn.fused_indexer_q_rope_quant_flaggems
    )

    fused_inv_rope_fp8_quant = staticmethod(
        _impl_deepseek_v4_attn.fused_inv_rope_fp8_quant_flaggems
    )

    fused_q_kv_rmsnorm = staticmethod(
        _impl_deepseek_v4_attn.fused_q_kv_rmsnorm_flaggems
    )

    indexer_k_quant_and_cache = staticmethod(
        _impl_deepseek_v4_attn.indexer_k_quant_and_cache_flaggems
    )

    cp_gather_indexer_k_quant_cache = staticmethod(
        _impl_deepseek_v4_attn.cp_gather_indexer_k_quant_cache_flaggems
    )

    top_k_per_row_prefill = staticmethod(
        _impl_deepseek_v4_attn.top_k_per_row_prefill_flaggems
    )

    pack_seq_triton = staticmethod(_impl_deepseek_v4_attn.pack_seq_triton_flaggems)

    top_k_per_row_decode = staticmethod(
        _impl_deepseek_v4_attn.top_k_per_row_decode_flaggems
    )

    unpack_seq_triton = staticmethod(_impl_deepseek_v4_attn.unpack_seq_triton_flaggems)

    flash_mla_with_kvcache = staticmethod(
        _impl_deepseek_v4_attn.flash_mla_with_kvcache_flaggems
    )

    flash_mla_sparse_fwd = staticmethod(
        _impl_deepseek_v4_attn.flash_mla_sparse_fwd_flaggems
    )

    gather_bf16_kv_from_pages = staticmethod(
        _impl_deepseek_v4_ops.gather_bf16_kv_from_pages_flaggems
    )

    bf16_mqa_logits = staticmethod(_impl_deepseek_v4_ops.bf16_mqa_logits_flaggems)
