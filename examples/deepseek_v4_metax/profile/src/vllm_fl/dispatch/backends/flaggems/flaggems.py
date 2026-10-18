# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems backend implementation.

This backend provides operator implementations using the FlagGems library.
"""

from __future__ import annotations

from typing import Optional, Union, Tuple

import torch
import os

from vllm_fl.dispatch.backends.base import Backend

from .impl import activation as _impl_activation
from .impl import deepseek_v4_attn as _impl_deepseek_v4_attn
from .impl import deepseek_v4_ops as _impl_deepseek_v4_ops
from .impl import fused_moe as _impl_fused_moe
from .impl import mhc as _impl_mhc
from .impl import normalization as _impl_normalization
from .impl import rotary as _impl_rotary
from .impl import router_gemm as _impl_router_gemm


class FlagGemsBackend(Backend):
    """
    FlagGems backend for operator implementations.

    This backend uses the flag_gems library to provide high-performance
    operator implementations.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "flagos"

    def is_available(self) -> bool:
        """Check if FlagGems is available."""
        if FlagGemsBackend._available is None:
            try:
                import flag_gems  # noqa F401

                FlagGemsBackend._available = True
            except ImportError:
                FlagGemsBackend._available = False
        return FlagGemsBackend._available

    # ==================== Operator Implementations ====================

    def fused_marlin_moe(self, *args, **kwargs) -> torch.Tensor:
        from flag_gems.fused.fused_marlin_moe import fused_marlin_moe

        # Current vLLM encodes ScalarType as a structured 64-bit integer,
        # while FlagGems' Marlin MVP still uses its legacy enum values.
        # Translate only at the FlagGems boundary; the CUDA vendor backend
        # must continue receiving the original vLLM ScalarType id.
        from vllm.scalar_type import scalar_types

        quant_type_map = {
            scalar_types.uint4b8.id: 0,
            scalar_types.uint8b128.id: 1,
            scalar_types.float4_e2m1f.id: 6,
        }
        args = list(args)
        if "quant_type_id" in kwargs:
            kwargs = dict(kwargs)
            quant_type_id = kwargs["quant_type_id"]
            if quant_type_id not in quant_type_map:
                raise NotImplementedError(
                    "FlagGems fused_marlin_moe does not support vLLM "
                    f"quant_type_id={quant_type_id}"
                )
            kwargs["quant_type_id"] = quant_type_map[quant_type_id]
        elif len(args) > 9:
            quant_type_id = args[9]
            if quant_type_id not in quant_type_map:
                raise NotImplementedError(
                    "FlagGems fused_marlin_moe does not support vLLM "
                    f"quant_type_id={quant_type_id}"
                )
            args[9] = quant_type_map[quant_type_id]

        return fused_marlin_moe(*args, **kwargs)

    def router_gemm_bf16_fp32(
        self, x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        return _impl_router_gemm.router_gemm_bf16_fp32_flaggems(x, weight)

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        return _impl_activation.silu_and_mul_flaggems(obj, x)

    def gelu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        return _impl_activation.gelu_and_mul_flaggems(obj, x)

    def silu_and_mul_with_clamp(
        self, x: torch.Tensor, swiglu_limit: float, swiglu_limit_tensor: torch.Tensor
    ) -> torch.Tensor:
        return _impl_activation.silu_and_mul_with_clamp_flaggems(x, swiglu_limit_tensor)

    def rms_norm(
        self, obj, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return _impl_normalization.rms_norm_flaggems(obj, x, residual)

    def rotary_embedding(
        self,
        obj,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_interleaved: bool = False,
        inplace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _impl_rotary.rotary_embedding_flaggems(
            obj, query, key, cos, sin, position_ids, rotary_interleaved, inplace
        )

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """
        Get the attention backend class path for FlagGems.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use Deepseek Sparse Attention (DSA)

        Returns:
            Fully qualified class path string
        """
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        # TritonAttentionBackend requires CUDA, check if available
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TritonAttentionBackend requires CUDA but CUDA is not available. "
                "Falling back to vendor implementation."
            )

        if use_mla:
            raise NotImplementedError("NOT support mla now!")

        if use_sparse:
            raise ValueError("use_sparse=True requires use_mla=True.")

        use_flaggems_attn = os.environ.get(
            "VLLM_FL_USE_FLAGGEMS_ATTN", "0"
        ).lower() in ("1", "true", "yes")

        if use_flaggems_attn:
            print("Using FlagGems attention backend.")
            return (
                "vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend"
            )

        return AttentionBackendEnum.TRITON_ATTN.get_path()

    def moe_align_block_size(
        self,
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
        expert_map: Optional[torch.Tensor] = None,
        pad_sorted_ids: bool = False,
        ignore_invalid_experts: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _impl_fused_moe.moe_align_block_size_flaggems(
            topk_ids,
            block_size,
            num_experts,
            expert_map,
            pad_sorted_ids,
            ignore_invalid_experts,
        )

    def moe_sum(self, inp, out):
        return _impl_fused_moe.moe_sum_flaggems(inp, out)

    def fused_topk_bias(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        scoring_func: str,
        e_score_correction_bias: torch.Tensor | None,
        topk: int,
        renormalize: bool,
        indices_type: torch.dtype | None = None,
        input_tokens: torch.Tensor | None = None,
        hash_indices_table: torch.Tensor | None = None,
        routed_scaling_factor: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _impl_fused_moe.fused_topk_bias_flaggems(
            hidden_states,
            gating_output,
            scoring_func,
            e_score_correction_bias,
            topk,
            renormalize,
            indices_type,
            input_tokens,
            hash_indices_table,
            routed_scaling_factor,
        )

    def topk_softmax(
        self,
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
    ):
        return _impl_fused_moe.topk_softmax_flaggems(
            topk_weights, topk_indices, token_expert_indices, gating_output, renormalize
        )

    def invoke_fused_moe_triton_kernel(
        self,
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        config,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        block_shape=None,
        B_bias=None,
    ):
        return _impl_fused_moe.invoke_fused_moe_triton_kernel_flaggems(
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            block_shape,
            B_bias,
        )

    def grouped_topk(
        self,
        scores,
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func=0,
    ):
        return _impl_fused_moe.grouped_topk_flaggems(
            scores,
            n_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
            bias,
            scoring_func,
        )

    def mhc_pre(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _impl_mhc.mhc_pre_flaggems(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
        )

    def mhc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        return _impl_mhc.mhc_post_flaggems(x, residual, post, comb)

    def hc_head_fused_kernel(
        self,
        hs_flat: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        out: torch.Tensor,
        hidden_size: int,
        rms_eps: float,
        hc_eps: float,
        hc_mult: int,
    ) -> None:
        return _impl_mhc.hc_head_fused_kernel_flaggems(
            hs_flat, fn, hc_scale, hc_base, out, hidden_size, rms_eps, hc_eps, hc_mult
        )

    def deepseek_v4_mega_moe_experts(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        out,
        layer_name: str,
        activation_clamp: float | None,
        fast_math: bool,
    ):
        # The new FlagGems fp8_fp4_mega_moe primitive is single-rank and
        # consumes staged tensors/raw weights. This dispatch contract is the
        # distributed vLLM layer-level op, so it is not directly compatible.
        raise NotImplementedError(
            "FlagGems fp8_fp4_mega_moe does not implement the distributed "
            "deepseek_v4_mega_moe_experts contract"
        )

    def deepseek_v4_fp8_einsum(
        self,
        a: torch.Tensor,
        a_scale: torch.Tensor,
        b: torch.Tensor,
        b_scale: torch.Tensor,
        out: torch.Tensor,
        equation: str,
        recipe: list[int],
    ) -> None:
        return _impl_deepseek_v4_attn.deepseek_v4_fp8_einsum_flaggems(
            a, a_scale, b, b_scale, out, equation, recipe
        )

    def fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        swa_kv_cache_2d: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        eps: float,
        block_size: int,
    ) -> None:
        return _impl_deepseek_v4_attn.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_flaggems(
            q,
            kv,
            swa_kv_cache_2d,
            slot_mapping,
            positions,
            cos_sin_cache,
            eps,
            block_size,
        )

    def combine_topk_swa_indices(
        self,
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
        return _impl_deepseek_v4_attn.combine_topk_swa_indices_flaggems(
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

    def compute_global_topk_indices_and_lens(
        self,
        topk_indices: torch.Tensor,
        token_to_req_indices: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        is_valid_token: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _impl_deepseek_v4_attn.compute_global_topk_indices_and_lens_flaggems(
            topk_indices, token_to_req_indices, block_table, block_size, is_valid_token
        )

    def dequantize_and_gather_k_cache(
        self,
        out: torch.Tensor,
        k_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        gather_lens: torch.Tensor | None,
        block_table: torch.Tensor,
        block_size: int,
        offset: int,
    ) -> None:
        return _impl_deepseek_v4_attn.dequantize_and_gather_k_cache_flaggems(
            out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
        )

    def fused_indexer_q_rope_quant(
        self,
        positions: torch.Tensor,
        index_q: torch.Tensor,
        index_q_cos_sin_cache: torch.Tensor,
        index_weights: torch.Tensor,
        index_weights_softmax_scale: float,
        index_weights_head_scale: float,
        use_fp4: bool = False,
    ):
        return _impl_deepseek_v4_attn.fused_indexer_q_rope_quant_flaggems(
            positions,
            index_q,
            index_q_cos_sin_cache,
            index_weights,
            index_weights_softmax_scale,
            index_weights_head_scale,
            use_fp4,
        )

    def fused_inv_rope_fp8_quant(
        self,
        o: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        n_groups: int,
        heads_per_group: int,
        nope_dim: int = 448,
        rope_dim: int = 64,
        quant_group_size: int = 128,
        tma_aligned_scales: bool = False,
    ):
        return _impl_deepseek_v4_attn.fused_inv_rope_fp8_quant_flaggems(
            o,
            positions,
            cos_sin_cache,
            n_groups,
            heads_per_group,
            nope_dim,
            rope_dim,
            quant_group_size,
            tma_aligned_scales,
        )

    def fused_q_kv_rmsnorm(
        self,
        qr: torch.Tensor,
        kv: torch.Tensor,
        q_weight: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _impl_deepseek_v4_attn.fused_q_kv_rmsnorm_flaggems(
            qr, kv, q_weight, kv_weight, eps
        )

    def indexer_k_quant_and_cache(
        self,
        k: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        quant_block_size: int,
        scale_fmt: str,
    ) -> None:
        return _impl_deepseek_v4_attn.indexer_k_quant_and_cache_flaggems(
            k, kv_cache, slot_mapping, quant_block_size, scale_fmt
        )

    def cp_gather_indexer_k_quant_cache(
        self,
        kv_cache: torch.Tensor,
        dst_k: torch.Tensor,
        dst_scale: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ) -> None:
        return _impl_deepseek_v4_attn.cp_gather_indexer_k_quant_cache_flaggems(
            kv_cache, dst_k, dst_scale, block_table, cu_seq_lens
        )

    def top_k_per_row_prefill(
        self,
        logits: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        raw_topk_indices: torch.Tensor,
        num_rows: int,
        stride0: int,
        stride1: int,
        topk_tokens: int,
    ) -> None:
        return _impl_deepseek_v4_attn.top_k_per_row_prefill_flaggems(
            logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            raw_topk_indices,
            num_rows,
            stride0,
            stride1,
            topk_tokens,
        )

    def pack_seq_triton(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        pad_value: float | int = -float("inf"),
    ) -> torch.Tensor:
        return _impl_deepseek_v4_attn.pack_seq_triton_flaggems(x, lengths, pad_value)

    def top_k_per_row_decode(
        self,
        logits: torch.Tensor,
        next_n: int,
        seq_lens: torch.Tensor,
        raw_topk_indices: torch.Tensor,
        num_rows: int,
        stride0: int,
        stride1: int,
        topk_tokens: int,
    ) -> None:
        return _impl_deepseek_v4_attn.top_k_per_row_decode_flaggems(
            logits,
            next_n,
            seq_lens,
            raw_topk_indices,
            num_rows,
            stride0,
            stride1,
            topk_tokens,
        )

    def unpack_seq_triton(
        self, packed_tensor: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        return _impl_deepseek_v4_attn.unpack_seq_triton_flaggems(packed_tensor, lengths)

    def flash_mla_with_kvcache(
        self,
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
        return _impl_deepseek_v4_attn.flash_mla_with_kvcache_flaggems(
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
        )

    def flash_mla_sparse_fwd(
        self, q, kv, indices, sm_scale, attn_sink, topk_length, out
    ):
        return _impl_deepseek_v4_attn.flash_mla_sparse_fwd_flaggems(
            q, kv, indices, sm_scale, attn_sink, topk_length, out
        )

    def gather_bf16_kv_from_pages(
        self, kv_cache, block_table, cu_seq_lens, token_to_seq, total_seq_lens, dst
    ):
        return _impl_deepseek_v4_ops.gather_bf16_kv_from_pages_flaggems(
            kv_cache, block_table, cu_seq_lens, token_to_seq, total_seq_lens, dst
        )

    def bf16_mqa_logits(
        self, q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits
    ):
        return _impl_deepseek_v4_ops.bf16_mqa_logits_flaggems(
            q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits
        )
