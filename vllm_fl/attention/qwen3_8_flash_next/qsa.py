# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from plugin-FL PR573 at 45b24bc7748720e0022a72e47a983398ddda36d3.
from __future__ import annotations

from typing import Any, ClassVar, cast

import torch

from torch import nn

from vllm.config import VllmConfig

from vllm.config.cache import CacheDType

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
)

from vllm.forward_context import get_forward_context

from vllm.model_executor.layers.attention.attention import set_default_quant_scales

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

from vllm.model_executor.layers.layernorm import GemmaRMSNorm

from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm.model_executor.layers.rotary_embedding import get_rope

from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
)

from vllm.platforms import current_platform

from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    canonicalize_singleton_dim_strides,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionType,
)

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from vllm_fl.dispatch import resolve_op

from vllm_fl.attention.qwen3_8_flash_next.cache import (
    QSACompressedKeyCache,
    QSAForwardMetadata,
    QSAKeyStateCache,
    QSAMetadataBuilder,
    canonical_qsa_rope_positions,
)

_qsa_compress_groups_with_ratio = resolve_op("qsa_compress_groups_with_ratio")
_qsa_select_paged_tokens = resolve_op("qsa_select_paged_tokens")
_qsa_store_cache_rows = resolve_op("qsa_store_cache_rows")

from vllm_fl.models.qwen3_8_flash_next.loading import (
    without_modelopt_fp4,
)


def apply_qsa_rope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Apply the main attention's exact 1D/MRoPE composition to QSA heads."""

    rotary_dim = rotary_emb.rotary_dim
    cache = rotary_emb._match_cos_sin_cache_dtype(tensor)  # noqa: SLF001
    cos_sin = cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        sections = rotary_emb.mrope_section
        if rotary_emb.mrope_interleaved:
            axis_cos, axis_sin = cos, sin
            channels = torch.arange(cos.shape[-1], device=cos.device)
            is_height = (channels % 3 == 1) & (channels < sections[1] * 3)
            is_width = (channels % 3 == 2) & (channels < sections[2] * 3)
            cos = torch.where(is_height, axis_cos[1], axis_cos[0])
            cos = torch.where(is_width, axis_cos[2], cos)
            sin = torch.where(is_height, axis_sin[1], axis_sin[0])
            sin = torch.where(is_width, axis_sin[2], sin)
        else:
            cos = torch.cat(
                [axis[index] for index, axis in enumerate(cos.split(sections, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [axis[index] for index, axis in enumerate(sin.split(sections, dim=-1))],
                dim=-1,
            )

    # Cross-vendor fallback: public/native tensor composition lets FlagGems or
    # the vendor PyTorch runtime dispatch every primitive appropriately.
    rotated = rotary_emb.apply_rotary_emb.forward_native(
        tensor[..., :rotary_dim],
        cos,
        sin,
    )
    return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)


class QSAIndexer(nn.Module):
    """Replicated Q/K projection plus paged, weight-free QSA selection.

    ``prefix`` must be the checkpoint's indexer prefix, normally
    ``model.layers.N.self_attn.indexer``.  Consequently the trainable names are
    ``index_qk_proj``, ``q_layernorm`` and ``k_layernorm`` under that prefix.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Any,
        layer_id: int,
        rotary_emb: nn.Module,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if vllm_config.cache_config is None:
            raise ValueError("QSA requires a paged KV cache")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.8-Flash-Next QSA currently requires BF16")

        self.layer_id = int(layer_id)
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.token_topk = int(config.indexer_budget)
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.rotary_emb = rotary_emb
        self.prefix = prefix

        self.index_qk_proj = ReplicatedLinear(
            int(config.hidden_size),
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj" if prefix else "index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )
        self.k_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )

        cache_config = vllm_config.cache_config
        cache_prefix = f"{prefix}." if prefix else ""
        self.raw_key_cache = QSAKeyStateCache(
            head_size=self.index_head_dim,
            dtype=torch.bfloat16,
            cache_rope_positions=vllm_config.model_config.uses_mrope,
            prefix=f"{cache_prefix}raw_key_cache",
            cache_config=cache_config,
            vllm_config=vllm_config,
        )
        self.compressed_key_cache = QSACompressedKeyCache(
            head_size=self.index_head_dim,
            dtype=torch.bfloat16,
            compress_ratio=self.compress_ratio,
            prefix=f"{cache_prefix}compressed_key_cache",
            cache_config=cache_config,
            vllm_config=vllm_config,
        )

    @property
    def output_width(self) -> int:
        return self.token_topk + self.compress_ratio - 1

    def project_qk(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project replicated Q/K, normalize+rotate Q, and preserve raw K."""

        qk, _ = self.index_qk_proj(hidden_states)
        q_raw, token_k = qk.split(
            (
                self.index_n_heads * self.index_head_dim,
                self.index_kv_heads * self.index_head_dim,
            ),
            dim=-1,
        )
        q = q_raw.reshape(-1, self.index_n_heads, self.index_head_dim)
        flat_q = q.reshape(-1, self.index_head_dim)
        normalized_q = self.q_layernorm(flat_q)
        q = normalized_q.reshape_as(q)
        q = apply_qsa_rope(self.rotary_emb, positions, q)
        return q, token_k.reshape(-1, 1, self.index_head_dim)

    def normalize_compressed_keys(
        self,
        compressed_keys: torch.Tensor,
        first_rope_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize pooled K and apply the first token's exact group position."""

        keys = compressed_keys.reshape(-1, self.index_head_dim)
        normalized_keys = self.k_layernorm(keys)
        keys = normalized_keys.reshape(-1, 1, self.index_head_dim)
        if getattr(self.rotary_emb, "mrope_section", None):
            positions = first_rope_positions.transpose(0, 1)
        else:
            positions = first_rope_positions[:, 0]
        return apply_qsa_rope(self.rotary_emb, positions, keys)

    def _metadata(
        self,
    ) -> tuple[QSAForwardMetadata, QSAForwardMetadata] | None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            return None
        raw = cast(QSAForwardMetadata, metadata[self.raw_key_cache.prefix])
        compressed = cast(
            QSAForwardMetadata, metadata[self.compressed_key_cache.prefix]
        )
        if raw.num_actual_tokens != compressed.num_actual_tokens:
            raise RuntimeError("QSA side-cache metadata token counts disagree")
        if raw.logical_positions.device.type == "cpu" and (
            not torch.equal(raw.logical_positions, compressed.logical_positions)
        ):
            raise RuntimeError("QSA side-cache metadata positions disagree")
        return raw, compressed

    def _update_and_compress(
        self,
        token_k: torch.Tensor,
        positions: torch.Tensor,
        raw_metadata: QSAForwardMetadata,
        compressed_metadata: QSAForwardMetadata,
    ) -> None:
        num_tokens = raw_metadata.num_actual_tokens
        raw_key_cache = self.raw_key_cache.key_cache
        rope_position_cache = self.raw_key_cache.rope_position_cache
        _qsa_store_cache_rows(
            raw_key_cache,
            raw_metadata.slot_mapping,
            token_k[:num_tokens],
        )
        if rope_position_cache is not None:
            position_rows = canonical_qsa_rope_positions(positions)[:num_tokens].to(
                device=rope_position_cache.device
            )
            _qsa_store_cache_rows(
                rope_position_cache,
                raw_metadata.slot_mapping,
                position_rows,
            )
        pooled, first_positions = _qsa_compress_groups_with_ratio(
            raw_key_cache,
            raw_metadata.block_table,
            raw_metadata.token_to_req,
            raw_metadata.logical_positions,
            compressed_metadata.slot_mapping,
            self.compress_ratio,
            rope_position_cache,
        )
        normalized = self.normalize_compressed_keys(pooled, first_positions)
        _qsa_store_cache_rows(
            self.compressed_key_cache.kv_cache,
            compressed_metadata.slot_mapping,
            normalized,
        )

    def _select(
        self,
        q: torch.Tensor,
        metadata: QSAForwardMetadata,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        return _qsa_select_paged_tokens(
            q,
            self.compressed_key_cache.kv_cache,
            metadata.block_table,
            metadata.token_to_req,
            metadata.logical_positions,
            metadata.seq_lens,
            self.token_topk,
            self.compress_ratio,
            out,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return fixed-width request-relative token indices padded with ``-1``."""

        metadata = self._metadata()
        if metadata is None:
            result = torch.full(
                (hidden_states.shape[0], self.output_width),
                -1,
                dtype=torch.int32,
                device=hidden_states.device,
            )
            if out is not None:
                out.copy_(result)
                return out
            return result
        raw_metadata, compressed_metadata = metadata
        num_tokens = raw_metadata.num_actual_tokens
        q, token_k = self.project_qk(
            hidden_states[:num_tokens], positions[..., :num_tokens]
        )
        self._update_and_compress(
            token_k,
            positions[..., :num_tokens],
            raw_metadata,
            compressed_metadata,
        )
        return self._select(q, compressed_metadata, out)


_qsa_sparse_paged_attention = resolve_op("qsa_sparse_paged_attention")


def _unpack_qsa_kv_cache(
    kv_cache: torch.Tensor,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return paged K/V views in ``(blocks, tokens, heads, dim)`` order.

    vLLM 0.24 stores FlashAttention caches as logical
    ``(blocks, 2, tokens, heads, dim)`` tensors.  This is the only layout
    supported by the QSA store path: unbinding the K/V dimension preserves
    views into the allocator-owned backing storage, including the non-unit
    stride between pages.

    A newer packed ``(blocks, heads, tokens, 2*dim)`` ABI cannot be safely
    adapted here.  Transposing that layout makes the subsequent head/dim
    flatten non-viewable, so a cache store would update a temporary copy
    instead of the allocator backing.  Reject it until the store kernel and
    allocator contract support that layout end to end.
    """
    if kv_cache.ndim == 5:
        if kv_cache.shape[1] != 2 or kv_cache.shape[-1] != head_size:
            raise ValueError(
                "invalid legacy QSA KV cache shape: expected "
                f"(blocks, 2, tokens, heads, {head_size}), got "
                f"{tuple(kv_cache.shape)}"
            )
        key_cache, value_cache = kv_cache.unbind(1)
    elif kv_cache.ndim == 4:
        raise ValueError(
            "QSA does not support the packed 4-D KV cache layout; expected "
            "the allocator-backed vLLM 0.24 5-D layout "
            f"(blocks, 2, tokens, heads, {head_size}), got "
            f"{tuple(kv_cache.shape)}"
        )
    else:
        raise ValueError(
            "invalid QSA KV cache rank: expected the vLLM legacy 5-D or "
            f"packed 4-D layout, got {kv_cache.ndim}-D"
        )
    return (
        canonicalize_singleton_dim_strides(key_cache),
        canonicalize_singleton_dim_strides(value_cache),
    )


class Qwen3_8FlashNextQSAAttentionBackend(AttentionBackend):
    """Main K/V cache owner for the Triton QSA transaction.

    QSA performs cache update and sparse attention in its model custom op, so
    it needs vLLM only for cache allocation and device-side metadata building.
    Owning those interfaces directly avoids coupling the model to a vendor's
    FlashAttention extension or cache-update ABI.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "QWEN38_FLASH_NEXT_QSA_FLAGTREE"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError(
            "QSA executes out-of-band through its model-owned Triton transaction"
        )

    @staticmethod
    def get_builder_cls() -> type[QSAMetadataBuilder]:
        return QSAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        # The cache is num-blocks-first and the QSA transaction passes the
        # tensor's physical block stride to every Triton kernel.  Opt in to
        # vLLM 0.24's padded-page view for hybrid cache page-size alignment.
        return True

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen3_8FlashNextQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Any,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = False,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen3.8-Flash-Next QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.8-Flash-Next QSA currently requires BF16")
        if cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"
            )
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support KV quantization"
            )
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires causal decoder attention"
            )

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support dual-chunk RoPE"
            )
        # Qwen3.8-Flash-Next full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Use the framework's normalization/rotary layers on every vendor.
        # A fused vendor implementation belongs in dispatch, not this model.
        self.use_fused_qk_norm_rope_gate = False

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if self.kv_cache_torch_dtype != torch.bfloat16:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires BF16 cache storage"
            )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend = Qwen3_8FlashNextQSAAttentionBackend
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                self.indexer.output_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def _run_qsa(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(QSAForwardMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        selected = self.indexer(
            hidden_states,
            positions,
            self.topk_indices_buffer[:num_tokens],
        )
        if selected.shape != (
            num_tokens,
            self.indexer.output_width,
        ):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        key_cache, value_cache = _unpack_qsa_kv_cache(self.kv_cache, self.head_dim)
        if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.8-Flash-Next QSA requires BF16 Q/K/V")

        slot_mapping = main_metadata.slot_mapping[:num_tokens]
        # Portable cache update. ``view`` (rather than reshape) makes a
        # non-viewable allocator layout fail instead of silently copying
        # rows away from the backing cache.
        flat_width = self.num_kv_heads * self.head_dim
        flat_key_cache = key_cache.view(
            key_cache.shape[0], key_cache.shape[1], 1, flat_width
        )
        flat_value_cache = value_cache.view(
            value_cache.shape[0], value_cache.shape[1], 1, flat_width
        )
        _qsa_store_cache_rows(
            flat_key_cache,
            slot_mapping,
            key[:num_tokens].reshape(num_tokens, 1, flat_width),
        )
        _qsa_store_cache_rows(
            flat_value_cache,
            slot_mapping,
            value[:num_tokens].reshape(num_tokens, 1, flat_width),
        )

        output.zero_()
        if num_tokens:
            _qsa_sparse_paged_attention(
                query[:num_tokens],
                key_cache,
                value_cache,
                self.topk_indices_buffer[:num_tokens],
                main_metadata.block_table,
                side_metadata.token_to_req[:num_tokens],
                self.scaling,
                output[:num_tokens],
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        encoded_layer_name = _encode_layer_name(self.layer_name)
        if current_platform.opaque_attention_op():
            torch.ops.vllm.qwen3_8_flash_next_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        else:
            qwen3_8_flash_next_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        flat_output = attn_output.view(num_tokens, -1)
        if gate is not None:
            flat_output = flat_output * torch.sigmoid(gate)
        output, _ = self.o_proj(flat_output)
        return output


def qwen3_8_flash_next_qsa_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Run the complete QSA state/update/attend transaction."""

    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen3_8FlashNextQSAAttention):
        raise TypeError(f"{layer_name} is not a Qwen3.8-Flash-Next QSA owner")
    layer._run_qsa(
        hidden_states,
        positions,
        query,
        key,
        value,
        output,
    )


def qwen3_8_flash_next_qsa_with_output_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    del hidden_states, positions, query, key, value, output, layer_name


direct_register_custom_op(
    op_name="qwen3_8_flash_next_qsa_with_output",
    op_func=qwen3_8_flash_next_qsa_with_output,
    mutates_args=["output"],
    fake_impl=qwen3_8_flash_next_qsa_with_output_fake,
)
