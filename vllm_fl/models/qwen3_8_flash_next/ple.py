# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from plugin-FL PR573 at 45b24bc7748720e0022a72e47a983398ddda36d3.
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.utils.torch_utils import (
    direct_register_custom_op,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
)

from vllm_fl.attention.qwen3_8_flash_next.ple_metadata import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadata,
)
from vllm_fl.dispatch import resolve_op
from vllm_fl.models.qwen3_8_flash_next.config import (
    Qwen3_8FlashNextTextConfig,
)
from vllm_fl.models.qwen3_8_flash_next.ngram import (
    Qwen3_8FlashNextNGramEmbedding,
    Qwen3_8FlashNextPLEGroupedNorm,
)

_ple_short_conv = resolve_op("ple_short_conv_")
_ple_gate = resolve_op("qwen38_ple_gate")


class Qwen3_8FlashNextPLELayer(nn.Module, MambaBase):
    def __init__(
        self,
        config: Qwen3_8FlashNextTextConfig,
        vllm_config: VllmConfig,
        layer_idx: int = 0,
        ple_dense_layer_id: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.model_config: ModelConfig = model_config
        self.cache_config: CacheConfig = cache_config
        self.layer_idx = layer_idx
        self.ple_dense_layer_id = (
            int(ple_dense_layer_id)
            if ple_dense_layer_id is not None
            else int(layer_idx)
        )
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.short_conv_dilation = int(config.ngram_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.activation = "silu"
        self.ple_embedding: nn.Module = Qwen3_8FlashNextNGramEmbedding(
            config,
            int(config.ple_embed_dim),
            self.ple_dense_layer_id,
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.scheduler_config.max_num_seqs,
            f"{prefix}.ple_embedding",
            params_dtype=model_config.dtype,
            host_lookup_mode=getattr(vllm_config, "additional_config", {}).get(
                "qwen38_ngram_mode", "sharded"
            ),
        )
        self.key_proj = ReplicatedLinear(
            int(config.ple_embed_dim),
            self.hc_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.key_proj",
        )
        self.value_proj = ReplicatedLinear(
            int(config.ple_embed_dim),
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.value_proj",
        )
        norm_args = (
            self.hc_hidden_size,
            config.rms_norm_eps,
            self.hidden_size,
            model_config.dtype,
        )
        self.norm_key = Qwen3_8FlashNextPLEGroupedNorm(*norm_args)
        self.norm_query = Qwen3_8FlashNextPLEGroupedNorm(*norm_args)
        self.norm_conv = Qwen3_8FlashNextPLEGroupedNorm(*norm_args)
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            padding=self.conv_state_len,
            dilation=self.short_conv_dilation,
            bias=False,
            dtype=model_config.dtype,
        )
        nn.init.zeros_(self.conv1d.weight)
        self.conv1d.weight._no_reinit = True
        self.kv_cache = (torch.tensor([]),)
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    def get_attn_backend(self) -> type[PleShortConvAttentionBackend]:
        return PleShortConvAttentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> Sequence[tuple[int, ...]]:
        # vLLM 0.24 has no explicit ``num_spec`` parameter; folding it into
        # the effective kernel yields the same state length:
        # (kernel - 1) + num_spec.
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=self.hc_hidden_size,
            conv_kernel=self.conv_state_len + self.num_spec_tokens + 1,
        )

    def _apply_norm(
        self, norm: Qwen3_8FlashNextPLEGroupedNorm, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        shape = hidden_states.shape
        return norm(hidden_states.flatten(-2)).reshape(shape)

    def _short_conv_fallback(self, inputs: torch.Tensor) -> torch.Tensor:
        # Profiling / CUDA graph capture only; conv state is not updated.
        inputs_t = inputs.transpose(0, 1).unsqueeze(0)
        output = self.conv1d(inputs_t)[..., : inputs_t.size(-1)]
        return F.silu(output).squeeze(0).transpose(0, 1)

    def prepare_host_lookup(
        self, input_ids, query_start_loc, ngram_context, *, dummy: bool = False
    ) -> None:
        """Graph-external producer called by the Qwen worker hook."""
        self.ple_embedding.prepare_host_lookup(
            input_ids, query_start_loc, ngram_context, dummy=dummy
        )

    def _short_conv_dilated_dispatch(
        self,
        inputs: torch.Tensor,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch packed requests without packing or scattering the cache.

        The FGV operator reads the previous speculative history at accepted-1
        and retains all newly verified prefixes. In particular a one-token
        decode following a rejected draft still needs the accepted offset.
        """
        x = inputs[: metadata.num_actual_tokens]
        output = torch.zeros_like(x)
        has_spec = metadata.spec_sequence_masks is not None
        has_non_spec = metadata.num_decodes > 0 or metadata.num_prefills > 0
        if has_spec:
            if (
                metadata.spec_state_indices_tensor is None
                or metadata.spec_query_start_loc is None
                or metadata.num_accepted_tokens is None
            ):
                raise ValueError("PLE speculative metadata is incomplete")
            if has_non_spec:
                if metadata.spec_token_indx is None:
                    raise ValueError("PLE mixed batch lacks speculative token mapping")
                spec_x = x.index_select(0, metadata.spec_token_indx.long())
                spec_out = torch.empty_like(spec_x)
            else:
                # Keep graph padding in the same fixed output allocation; the
                # varlen operator zeroes tokens beyond the query offsets.
                spec_x, spec_out = x, output
            _ple_short_conv(
                spec_x,
                spec_out,
                conv_state,
                conv_weights,
                # FGV's state writer retains the index dtype for row*stride.
                # Hybrid page strides can overflow int32 even for small row IDs.
                metadata.spec_state_indices_tensor.to(torch.int64),
                dilation=self.short_conv_dilation,
                query_start_loc=metadata.spec_query_start_loc,
                accepted_tokens=metadata.num_accepted_tokens,
                max_query_len=metadata.spec_query_len,
                null_block_id=NULL_BLOCK_ID,
            )
            if has_non_spec:
                output.index_copy_(0, metadata.spec_token_indx.long(), spec_out)
        if has_non_spec:
            if (
                metadata.state_indices_tensor is None
                or metadata.non_spec_query_start_loc is None
            ):
                raise ValueError("PLE non-speculative metadata is incomplete")
            if has_spec:
                if metadata.non_spec_token_indx is None:
                    raise ValueError("PLE mixed batch lacks non-spec token mapping")
                non_x = x.index_select(0, metadata.non_spec_token_indx.long())
                non_out = torch.empty_like(non_x)
            else:
                non_x, non_out = x, output
            if metadata.num_decodes and metadata.num_prefills:
                if (
                    metadata.has_initial_states_d is None
                    or metadata.has_initial_states_p is None
                ):
                    raise ValueError(
                        "PLE mixed decode/prefill lacks initial-state flags"
                    )
                initial = torch.cat(
                    (
                        metadata.has_initial_states_d[: metadata.num_decodes],
                        metadata.has_initial_states_p[: metadata.num_prefills],
                    )
                )
            elif metadata.num_prefills:
                initial = metadata.has_initial_states_p
            else:
                initial = metadata.has_initial_states_d
            if initial is None:
                raise ValueError("PLE metadata lacks initial-state flags")
            _ple_short_conv(
                non_x,
                non_out,
                conv_state,
                conv_weights,
                metadata.state_indices_tensor.to(torch.int64),
                dilation=self.short_conv_dilation,
                query_start_loc=metadata.non_spec_query_start_loc,
                has_initial_state=initial,
                null_block_id=NULL_BLOCK_ID,
            )
            if has_spec:
                output.index_copy_(0, metadata.non_spec_token_indx.long(), non_out)
        return output

    def _short_conv(self, inputs: torch.Tensor) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return self._short_conv_fallback(inputs)

        if not isinstance(attn_metadata, dict):
            raise RuntimeError(
                "PLE short-conv expects per-layer attention metadata dict "
                f"during inference, got {type(attn_metadata).__name__}."
            )

        layer_attn_metadata = attn_metadata.get(self.prefix)
        if layer_attn_metadata is None:
            raise RuntimeError(
                f"Missing short-conv metadata for layer '{self.prefix}'. "
                "This would bypass conv-state updates and is not allowed."
            )
        if not isinstance(layer_attn_metadata, PleShortConvAttentionMetadata):
            raise TypeError(
                "Expected PleShortConvAttentionMetadata for layer "
                f"'{self.prefix}', got "
                f"{type(layer_attn_metadata).__name__}."
            )

        conv_state = self.kv_cache[0]
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_weights = self.conv1d.weight.squeeze(1)

        state_capacity = self.conv_state_len + self.num_spec_tokens
        if state_capacity > 0:
            if conv_state.size(-1) < state_capacity:
                raise RuntimeError(
                    "PLE short-conv cache is smaller than expected for "
                    f"dilated convolution: got {conv_state.size(-1)}, "
                    f"expect at least {state_capacity}."
                )
            conv_state = conv_state[..., -state_capacity:]
        return self._short_conv_dilated_dispatch(
            inputs,
            layer_attn_metadata,
            conv_state,
            conv_weights.to(dtype=inputs.dtype),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        if input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "PLE expects input_ids and hidden_states to have the same "
                f"token length, got {input_ids.shape[0]} and "
                f"{hidden_states.shape[0]}"
            )
        embeddings = self.ple_embedding(input_ids, query_start_loc, ngram_context)
        key, _ = self.key_proj(embeddings)
        value, _ = self.value_proj(embeddings)
        token_count = hidden_states.shape[0]
        key = key.reshape(token_count, self.hc_count, self.hidden_size)
        query = hidden_states.reshape(token_count, self.hc_count, self.hidden_size)
        key = self._apply_norm(self.norm_key, key)
        query = self._apply_norm(self.norm_query, query)
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = _ple_gate(gate)
        gated_value = gate * value.unsqueeze(-2)
        normalized = self._apply_norm(self.norm_conv, gated_value).flatten(-2)
        conv_output = torch.zeros_like(normalized)
        torch.ops.vllm.qwen3_8_flash_next_ple_short_conv(
            normalized,
            conv_output,
            self.prefix,
        )
        return gated_value.flatten(-2) + conv_output


def qwen3_8_flash_next_ple_short_conv(
    inputs: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    result = layer._short_conv(inputs)
    output[: result.shape[0]].copy_(result)


def qwen3_8_flash_next_ple_short_conv_fake(
    inputs: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="qwen3_8_flash_next_ple_short_conv",
    op_func=qwen3_8_flash_next_ple_short_conv,
    mutates_args=["output"],
    fake_impl=qwen3_8_flash_next_ple_short_conv_fake,
)
