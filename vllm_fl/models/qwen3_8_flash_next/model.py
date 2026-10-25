# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from plugin-FL PR573 at 45b24bc7748720e0022a72e47a983398ddda36d3.
from __future__ import annotations

import inspect

from collections.abc import Iterable

from itertools import islice

import torch

from torch import nn

from vllm.compilation.decorators import support_torch_compile

from vllm.config import VllmConfig

from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)

from vllm.model_executor.layers.logits_processor import LogitsProcessor


from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)

from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    _require_is_multimodal,
)

from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)

from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextMLP,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)

from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
    Qwen3_VisionTransformer,
)

from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    StageMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm.multimodal.inputs import MultiModalFeatureSpec

from vllm.sequence import IntermediateTensors

from vllm.tokenizers.registry import cached_tokenizer_from_config

from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm.v1.kv_cache_interface import (
    MambaSpec,
)

try:
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateCopyFuncsByType,
    )
except ImportError:  # vLLM 0.24 type-only compatibility
    MambaStateCopyFuncsByType = dict

try:
    from vllm.model_executor.models.utils import maybe_fuse_shared_experts
except ImportError:  # vLLM 0.24 predates optional AITER shared-expert fusion

    def maybe_fuse_shared_experts(weights, **_kwargs):
        return weights


from vllm_fl.dispatch import resolve_op

from vllm_fl.attention.qwen3_8_flash_next.qsa import (
    Qwen3_8FlashNextQSAAttention,
)

from vllm_fl.models.qwen3_8_flash_next.config import (
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextTextConfig,
)

from vllm_fl.models.qwen3_8_flash_next.loading import (
    _QWEN35_HF_TO_VLLM_MAPPER,
    _HAS_NATIVE_STACKED_WEIGHTS_MAPPER,
    _QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES,
    _VLLM024StackedAutoWeightsLoader,
    _get_vllm024_expert_mappings,
    _load_vllm024_fused_expert_weight,
    _load_vllm024_single_expert_weight,
    _map_vllm024_stacked_weights,
    _remap_qsa_cache_scale_name,
)

from vllm_fl.models.qwen3_8_flash_next.ple import (
    Qwen3_8FlashNextPLELayer,
)

from vllm_fl.models.qwen3_8_flash_next.projections import (
    use_qwen38_shared_expert_projection,
)

from vllm_fl.models.qwen3_8_flash_next.residual import (
    GatedResidualSimple,
    HyperConnectionConfig,
)


from vllm_fl.models.qwen3_8_flash_next.gdn import (
    Qwen38GatedDeltaNetAttention as QwenGatedDeltaNetAttention,
)


class Qwen3_8FlashNextSparseMoeBlock(Qwen3NextSparseMoeBlock):
    """Qwen3Next MoE with Qwen3.8-Flash-Next HC validation."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        if vllm_config.parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next HC does not support sequence-parallel MoE"
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        use_qwen38_shared_expert_projection(self.shared_expert_gate)
        # The current FusedMoEFactory owns its final tensor-parallel
        # reduction. Do not reduce the result a second time in the HC caller.
        self.requires_tp_all_reduce = False


class Qwen3_8FlashNextDecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
        force_disable_ple: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        if vllm_config.parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next HC does not support sequence-parallel MoE"
            )
        self.ple: Qwen3_8FlashNextPLELayer | None = None
        ple_layer_ids = config.ple_layer_ids
        if (self.layer_idx + 1) in ple_layer_ids and not force_disable_ple:
            ple_layer_ids_sorted = sorted(set(ple_layer_ids))
            ple_dense_layer_id_map = {
                abs_id: idx for idx, abs_id in enumerate(ple_layer_ids_sorted)
            }
            ple_dense_layer_id = ple_dense_layer_id_map[self.layer_idx + 1]
            self.ple = Qwen3_8FlashNextPLELayer(
                config,
                vllm_config=vllm_config,
                layer_idx=self.layer_idx,
                ple_dense_layer_id=ple_dense_layer_id,
                prefix=f"{prefix}.ple",
            )

        if layer_type == "linear_attention":
            gdn_kwargs = {
                "vllm_config": vllm_config,
                "prefix": f"{prefix}.linear_attn",
                "gqa_interleaved_layout": False,
            }
            if (
                "reduce_results"
                in inspect.signature(QwenGatedDeltaNetAttention.__init__).parameters
            ):
                gdn_kwargs["reduce_results"] = False
            self.linear_attn = QwenGatedDeltaNetAttention(config, **gdn_kwargs)
            # vLLM 0.24 predates the constructor option. HC owns the single TP
            # reduction after each branch, so disable the GDN output linear's
            # internal reduction explicitly to avoid a double all-reduce.
            if (
                "reduce_results"
                not in inspect.signature(QwenGatedDeltaNetAttention.__init__).parameters
            ):
                self.linear_attn.out_proj.reduce_results = False
            self._gdn_requires_output_buffer = (
                "output"
                in inspect.signature(QwenGatedDeltaNetAttention.forward).parameters
            )
        elif layer_type == "full_attention":
            use_qsa = getattr(config, "indexer_n_heads", None) is not None
            if not use_qsa:
                self.self_attn = Qwen3NextAttention(
                    config,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    reduce_results=False,
                    prefix=f"{prefix}.self_attn",
                )
            else:
                self.self_attn = Qwen3_8FlashNextQSAAttention(
                    vllm_config=vllm_config,
                    config=config,
                    layer_id=self.layer_idx,
                    quant_config=quant_config,
                    reduce_results=False,
                    prefix=f"{prefix}.self_attn",
                )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        num_experts = getattr(config, "num_experts", 0) or 0
        absolute_layer_id = self.layer_idx + 1
        is_moe_layer = self.layer_idx not in mlp_only_layers and (
            num_experts > 0 and absolute_layer_id % config.decoder_sparse_step == 0
        )
        if is_moe_layer:
            self.mlp = Qwen3_8FlashNextSparseMoeBlock(
                vllm_config=vllm_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.mlp",
            )

        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.attn_hyper_connection = GatedResidualSimple(
            hc_config,
            layer_idx=self.layer_idx,
            role="attn",
            small_projection=resolve_op("qwen38_small_projection"),
        )
        self.mlp_hyper_connection = GatedResidualSimple(
            hc_config,
            layer_idx=self.layer_idx,
            role="mlp",
            small_projection=resolve_op("qwen38_small_projection"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del kwargs
        if residual is not None:
            raise ValueError("HC layers do not use a separate residual tensor")
        if self.ple is not None:
            if input_ids is None:
                raise ValueError("PLE requires input_ids")
            if query_start_loc is None:
                raise ValueError("ngram PLE requires query_start_loc")
            if ngram_context is None:
                raise ValueError("ngram PLE requires ngram_context")
            hidden_states = hidden_states + self.ple(
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
            )

        mixed, hc_residual = self.attn_hyper_connection.mix(hidden_states)
        if self.layer_type == "linear_attention":
            if self._gdn_requires_output_buffer:
                # vLLM 0.24's pluggable GDN ABI writes into a caller-provided
                # buffer; newer vLLM returns the projected tensor directly.
                self_attention_output = torch.empty_like(mixed)
                self.linear_attn(mixed, self_attention_output)
            else:
                self_attention_output = self.linear_attn(hidden_states=mixed)
        elif self.layer_type == "full_attention":
            self_attention_output = self.self_attn(
                hidden_states=mixed,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output
        if get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        hidden_states = self.attn_hyper_connection.combine(hidden_states, hc_residual)

        mixed, hc_residual = self.mlp_hyper_connection.mix(hidden_states)
        hidden_states = self.mlp(mixed)
        if get_tensor_model_parallel_world_size() > 1 and getattr(
            self.mlp, "requires_tp_all_reduce", True
        ):
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        hidden_states = self.mlp_hyper_connection.combine(hidden_states, hc_residual)
        return hidden_states, None


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "query_start_loc": 0,
        "ngram_context": 0,
        "deepstack_input_embeds": 0,
    }
)
class Qwen3_8FlashNextModel(nn.Module):
    hf_to_vllm_mapper = _QWEN35_HF_TO_VLLM_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config
        self.config = config
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        self.vocab_size = config.vocab_size
        self._qsa_layer_ids = frozenset(
            layer_idx
            for layer_idx, layer_type in enumerate(config.layer_types)
            if layer_type == "full_attention"
            and getattr(config, "indexer_n_heads", None) is not None
        )
        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)

        def get_layer(prefix: str) -> Qwen3_8FlashNextDecoderLayer:
            layer_idx = extract_layer_index(prefix)
            return Qwen3_8FlashNextDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[layer_idx],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        intermediate_size = config.hidden_size * config.hc_count
        # The proposer reads this storage after CUDA graph replay, when Python
        # forward does not run. Keep its address stable across all batches.
        self.register_buffer(
            "_mtp_target_hidden_states",
            torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                intermediate_size,
                dtype=vllm_config.model_config.dtype,
            )
            if vllm_config.speculative_config is not None
            else None,
            persistent=False,
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], intermediate_size
        )

        self.hyper_connection_mixer: GatedResidualSimple | None
        if get_pp_group().is_last_rank:
            hc_config = HyperConnectionConfig(
                hc_count=config.hc_count,
                hidden_size=config.hidden_size,
                params_dtype=torch.bfloat16,
                hc_lowrank=config.hc_lowrank,
                rms_norm_eps=config.rms_norm_eps,
                hc_per_branch_norm=True,
            )
            self.hyper_connection_mixer = GatedResidualSimple(
                hc_config, use_combine=False, role="final"
            )
        else:
            self.hyper_connection_mixer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
            hidden_states = hidden_states.repeat(1, self.config.hc_count)
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]
            residual = None

        for layer_idx, layer in islice(
            enumerate(self.layers), self.start_layer, self.end_layer
        ):
            hidden_states, residual = layer(
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
            )
            if deepstack_input_embeds is not None and layer_idx < len(
                deepstack_input_embeds
            ):
                deepstack_embed = deepstack_input_embeds[
                    f"deepstack_input_embeds_{layer_idx}"
                ]
                deepstack_embed = (
                    deepstack_embed.unsqueeze(-2)
                    .expand(
                        *deepstack_embed.shape[:-1],
                        self.config.hc_count,
                        self.config.hidden_size,
                    )
                    .flatten(-2)
                )
                hidden_states = hidden_states + deepstack_embed

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        assert self.hyper_connection_mixer is not None
        if self._mtp_target_hidden_states is not None:
            self._mtp_target_hidden_states[: hidden_states.shape[0]].copy_(
                hidden_states
            )
        hidden_states, _ = self.hyper_connection_mixer.mix(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = (
            (
                _remap_qsa_cache_scale_name(name, self._qsa_layer_ids),
                weight,
            )
            for name, weight in weights
        )
        weights = maybe_fuse_shared_experts(
            weights,
            n_routed_experts=getattr(self.config, "num_experts", 0) or 0,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        # Non-persistent PLE state rebuilt in __init__; skip any ckpt
        # column for them.
        skip_substrs = [
            "hashstats_",
            "token_lookup",
            "hyper_connection_mixer.block_inject_weight",
        ]
        mapper = self.hf_to_vllm_mapper
        loader_cls = AutoWeightsLoader
        legacy_loaded: set[str] = set()
        if not _HAS_NATIVE_STACKED_WEIGHTS_MAPPER:
            expert_mappings = _get_vllm024_expert_mappings(
                self,
                num_experts=getattr(self.config, "num_experts", 0) or 0,
                num_redundant_experts=self.num_redundant_experts,
            )
            if expert_mappings is not None:
                regular_expert_mapping, fused_expert_mapping = expert_mappings
                params_dict = dict(self.named_parameters())
                legacy_input = weights

                def _legacy_weights() -> Iterable[tuple[str, torch.Tensor]]:
                    for name, loaded_weight in legacy_input:
                        handled, loaded = _load_vllm024_fused_expert_weight(
                            name,
                            loaded_weight,
                            params_dict,
                            fused_expert_mapping,
                            getattr(self.config, "num_experts", 0) or 0,
                        )
                        if handled:
                            legacy_loaded.update(loaded)
                            continue
                        handled, loaded = _load_vllm024_single_expert_weight(
                            name,
                            loaded_weight,
                            params_dict,
                            regular_expert_mapping,
                        )
                        if handled:
                            legacy_loaded.update(loaded)
                            continue
                        yield from _map_vllm024_stacked_weights(
                            ((name, loaded_weight),)
                        )

                weights = _legacy_weights()
            else:
                weights = _map_vllm024_stacked_weights(weights)
            mapper = None
            loader_cls = _VLLM024StackedAutoWeightsLoader
        loader = loader_cls(
            self,
            skip_substrs=skip_substrs,
            ignore_unexpected_suffixes=_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy(),
        )
        loaded = loader.load_weights(
            weights,
            mapper=mapper,
        )
        loaded.update(legacy_loaded)
        return loaded


class Qwen3_8FlashNextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    QwenNextMixtureOfExperts,
    IsHybrid,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model."}
    )
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.scheduler_config = vllm_config.scheduler_config
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.8-Flash-Next currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.model = Qwen3_8FlashNextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        # Set MoE hyperparameters
        if (getattr(config, "num_experts", 0) or 0) > 0:
            QwenNextMixtureOfExperts.set_moe_parameters(self)

    def get_mtp_target_hidden_states(self) -> torch.Tensor:
        states = self.model._mtp_target_hidden_states
        if states is None:
            raise RuntimeError("MTP target states require speculative decoding")
        return states

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # Forward kwargs unchanged so the runner's _maybe_add_ngram_kwargs
        # path (query_start_loc / ngram_context) reaches Qwen3_8FlashNextModel.
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    @classmethod
    def get_ple_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_ple_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int]]:
        hf_config = vllm_config.model_config.hf_text_config
        conv_kernel_size = hf_config.ple_conv_kernel_size
        short_conv_dilation = hf_config.ngram_size
        conv_state_len = (conv_kernel_size - 1) * short_conv_dilation
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        hc_count = hf_config.hc_count
        hc_hidden_size = hf_config.hidden_size * hc_count
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=hc_hidden_size,
            conv_kernel=conv_state_len + num_spec + 1,
        )

    @classmethod
    def get_gdn_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_gdn_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return cls.get_gdn_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return cls.get_gdn_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(
        cls,
        mamba_types: set[MambaAttentionBackendEnum],
    ) -> MambaStateCopyFuncsByType:
        copy_funcs_by_type = {
            MambaAttentionBackendEnum.GDN_ATTN: cls.get_mamba_state_copy_func(),
            MambaAttentionBackendEnum.SHORT_CONV: (
                MambaStateCopyFuncCalculator.short_conv_state_copy_func()
            ),
        }
        missing_types = mamba_types - copy_funcs_by_type.keys()
        assert not missing_types, f"missing state copy funcs for {missing_types}"
        return {
            mamba_type: copy_funcs_by_type[mamba_type] for mamba_type in mamba_types
        }

    @classmethod
    def get_mamba_specs_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[MambaSpec, ...]:
        """Return all MambaSpecs for this model (GDN layers + PLE layer).

        The PLE layer uses a separate short_conv MambaSpec whose page_size_bytes
        may exceed the GDN spec; callers should take the maximum.
        """
        return (
            MambaSpec(
                shapes=cls.get_gdn_mamba_state_shape_from_config(vllm_config),
                dtypes=cls.get_gdn_mamba_state_dtype_from_config(vllm_config),
                block_size=-1,
            ),
            MambaSpec(
                shapes=cls.get_ple_mamba_state_shape_from_config(vllm_config),
                dtypes=cls.get_ple_mamba_state_dtype_from_config(vllm_config),
                block_size=-1,
            ),
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        del mm_features
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_substrs=["mtp."],
            ignore_unexpected_suffixes=_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen3_8FlashNextMixtureOfExperts(MixtureOfExperts):
    """Expose Qwen3.8-Flash-Next routed experts through vLLM's EPLB protocol."""

    language_model: Qwen3_8FlashNextForCausalLM

    def _set_moe_parameters(self) -> None:
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        language_model = getattr(self, "model", None)
        if language_model is None:
            language_model = self.language_model.model
        for layer in language_model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.num_moe_layers = len(self.moe_layers)
        if example_moe is None:
            self.num_expert_groups = 1
            self.num_shared_experts = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0
            return

        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_physical_experts = num_physical_experts
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class Qwen3_8FlashNextProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self) -> Qwen3_8FlashNextConfig:
        return self.ctx.get_hf_config(Qwen3_8FlashNextConfig)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_8FlashNextProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_8FlashNextForConditionalGeneration(
    Qwen3_5ForConditionalGeneration,
    HasInnerState,
    Qwen3_8FlashNextMixtureOfExperts,
):
    """Qwen3-VL vision tower backed by the Qwen3.8-Flash-Next language model."""

    requires_raw_input_tokens = True

    packed_modules_mapping = Qwen3_5ForConditionalGeneration.packed_modules_mapping

    def _init_video_pruning(self, multimodal_config) -> None:
        if not hasattr(multimodal_config, "get_video_pruning_spec"):
            self.is_multimodal_pruning_enabled = False
            self.video_pruning_method = None
            self.video_pruning_rate = 0.0
            return
        pruning_spec = multimodal_config.get_video_pruning_spec()
        if pruning_spec is None:
            self.video_pruning_method = None
            self.video_pruning_rate = multimodal_config.video_pruning_rate
        else:
            self.video_pruning_method, self.video_pruning_rate = pruning_spec
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        nn.Module.__init__(self)
        config: Qwen3_8FlashNextConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is None:
            raise ValueError(
                "Qwen3_8FlashNextForConditionalGeneration requires multimodal_config"
            )

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.language_model_only = multimodal_config.language_model_only
        if self.language_model_only:
            self.use_data_parallel = False
            self.is_multimodal_pruning_enabled = False
            self.video_pruning_method = None
            self.video_pruning_rate = 0.0
            self._tokenizer = None
            self.visual = StageMissingLayer("vision_tower")
            self._tower_model_names = []
        else:
            self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
            self._init_video_pruning(multimodal_config)
            self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    config.vision_config,
                    norm_eps=config.text_config.rms_norm_eps,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )

        self.use_deepstack = (
            not self.language_model_only
            and bool(config.vision_config.deepstack_visual_indexes)
            and not isinstance(self.visual, StageMissingLayer)
        )
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        if self.use_deepstack:
            self.deepstack_input_embeds = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]
            self.deepstack_input_embeds_num_tokens = 0

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_8FlashNextForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        if not get_pp_group().is_first_rank and self.use_deepstack:
            assert self.language_model.model.start_layer >= len(
                config.vision_config.deepstack_visual_indexes
            ), (
                "start_layer should be greater than or equal to "
                "len(deepstack_visual_indexes)"
            )
        self._set_moe_parameters()

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if self.language_model_only:
            raise ValueError(
                "Qwen3.8-Flash-Next language_model_only does not accept "
                "multimodal embeddings"
            )

        is_multimodal = _require_is_multimodal(is_multimodal)
        if self.use_deepstack:
            deepstack_input_embeds, multimodal_embeddings = (
                self._compute_deepstack_embeds(
                    inputs_embeds=inputs_embeds,
                    multimodal_embeddings=multimodal_embeddings,
                    is_multimodal=is_multimodal,
                )
            )
        else:
            deepstack_input_embeds = None

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
        if deepstack_input_embeds is not None:
            self._set_deepstack_input_embeds(deepstack_input_embeds)
        return inputs_embeds

    def get_mtp_target_hidden_states(self) -> torch.Tensor:
        return self.language_model.get_mtp_target_hidden_states()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        if inputs_embeds is not None and get_pp_group().is_first_rank:
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.size(0)
            )
        else:
            deepstack_input_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            query_start_loc=kwargs.get("query_start_loc"),
            ngram_context=kwargs.get("ngram_context"),
            deepstack_input_embeds=deepstack_input_embeds,
        )
        if inputs_embeds is not None and get_pp_group().is_first_rank:
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["visual."] if self.language_model_only else None,
            skip_substrs=["mtp."],
            ignore_unexpected_suffixes=_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return Qwen3_8FlashNextForCausalLM.get_mamba_state_dtype_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return Qwen3_8FlashNextForCausalLM.get_mamba_state_shape_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return Qwen3_8FlashNextForCausalLM.get_mamba_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(
        cls,
        mamba_types: set[MambaAttentionBackendEnum],
    ) -> MambaStateCopyFuncsByType:
        return Qwen3_8FlashNextForCausalLM.get_mamba_state_copy_funcs(mamba_types)

    @classmethod
    def get_mamba_specs_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[MambaSpec, ...]:
        return Qwen3_8FlashNextForCausalLM.get_mamba_specs_from_config(vllm_config)


Qwen4ExpForCausalLM = Qwen3_8FlashNextForCausalLM


Qwen4ExpForConditionalGeneration = Qwen3_8FlashNextForConditionalGeneration


__all__ = [
    "GatedResidualSimple",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
    "Qwen3_8FlashNextConfig",
    "Qwen3_8FlashNextTextConfig",
    "Qwen3_8FlashNextVisionConfig",
    "Qwen3_8FlashNextForCausalLM",
    "Qwen3_8FlashNextForConditionalGeneration",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
]
