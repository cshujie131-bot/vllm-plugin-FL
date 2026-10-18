# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""K3 model components selected at construction through native factories."""

from functools import partial

import torch
import vllm.envs as envs
from vllm.distributed import get_pp_group, tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors,
)
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.models.kimi_k3.nvidia.model import (
    KimiDecoderLayer,
    KimiK3ForConditionalGeneration,
    KimiLinearForCausalLM,
    KimiLinearModel,
    KimiMLP,
    KimiMoE,
    LatentMoEComponents,
)
from vllm.models.kimi_k3.nvidia.ops import attn_res
from vllm.sequence import IntermediateTensors

from .activations import K3SituAndMul
from .attn_res_windows import (
    AttnResTokenWindow,
    configure_layer,
)
from .attn_res_windows import (
    eligible as attn_res_window_eligible,
)
from .context import execution_config
from .ep_ar import reduce_mlp, reduce_sensitive
from .ep_striped import install
from .kda_adapter import K3DeltaAttention
from .latent_moe import LatentColumnParallelLinear, ShardedLatentMoERunner
from .mla_adapter import K3MultiHeadLatentAttention
from .tp8_gate_projection import Tp8SharedGateUpProjection
from .vision_adapter import K3VisionModel


class K3MLP(KimiMLP):
    gate_up_projection_cls = Tp8SharedGateUpProjection
    situ_activation_cls = K3SituAndMul

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        hidden_act,
        quant_config=None,
        reduce_results=True,
        use_sequence_parallel=False,
        can_shard_sequence_parallel=False,
        run_gemm_rs=False,
        prefix="",
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
    ):
        self.k3_execution = execution_config()
        self.ep_ar_reduce = (
            self.k3_execution.ep_allreduce
            and not use_sequence_parallel
            and reduce_results
        )
        super().__init__(
            hidden_size,
            intermediate_size,
            hidden_act,
            quant_config=quant_config,
            reduce_results=reduce_results and not self.ep_ar_reduce,
            use_sequence_parallel=use_sequence_parallel,
            can_shard_sequence_parallel=can_shard_sequence_parallel,
            run_gemm_rs=run_gemm_rs,
            prefix=prefix,
            activation_situ_beta=activation_situ_beta,
            activation_situ_linear_beta=activation_situ_linear_beta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super().forward(x)
        return reduce_mlp(output) if self.ep_ar_reduce else output


class K3MoE(KimiMoE):
    shared_expert_cls = K3MLP

    def __init__(self, *args, **kwargs):
        self.k3_execution = execution_config()
        super().__init__(*args, **kwargs)

    def _get_expert_partition_alignment(self, config, quant_config):
        return getattr(
            quant_config,
            "min_moe_intermediate_per_partition",
            super()._get_expert_partition_alignment(config, quant_config),
        )

    def _get_latent_moe_components(self, quant_config, use_sequence_parallel):
        shard = getattr(quant_config, "shard_latent_projections", False)
        overlap = not use_sequence_parallel and not self.k3_execution.ep_allreduce
        if not shard:
            return LatentMoEComponents(overlap_down_projection=overlap)
        return LatentMoEComponents(
            down_projection=partial(
                LatentColumnParallelLinear,
                gather_output=True,
                sequence_parallel=use_sequence_parallel,
            ),
            up_projection=partial(
                LatentColumnParallelLinear,
                gather_output=use_sequence_parallel,
                sequence_parallel=use_sequence_parallel,
            ),
            runner_cls=ShardedLatentMoERunner,
            shard_shared_experts=True,
            overlap_down_projection=overlap,
        )


class _K3ParallelPolicy:
    def _select_sequence_parallel(self, parallel_config, use_mega_moe):
        return not self.k3_execution.ep_allreduce and super()._select_sequence_parallel(
            parallel_config, use_mega_moe
        )


class K3DecoderLayer(_K3ParallelPolicy, KimiDecoderLayer):
    delta_attention_cls = K3DeltaAttention
    mla_cls = K3MultiHeadLatentAttention
    moe_cls = K3MoE
    mlp_cls = K3MLP

    def __init__(
        self, config, vllm_config, prefix="", *, aux_stream=None, run_gemm_rs=False
    ):
        self.k3_execution = execution_config(vllm_config)
        self.ep_ar_mode = self.k3_execution.ep_allreduce
        super().__init__(
            config,
            vllm_config,
            prefix,
            aux_stream=aux_stream,
            run_gemm_rs=run_gemm_rs,
        )
        if self.ep_ar_mode:
            self.self_attn.o_proj.reduce_results = False
        self.attn_res_window_enabled = configure_layer(self, vllm_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor | None,
        prefix_sum: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        ar_window = kwargs.get("attn_res_window")
        if ar_window is not None:
            if not self.attn_res_window_enabled:
                raise RuntimeError("AttnRes window entered an unsupported decoder")
            hidden_states, prefix_sum, residual = ar_window.pre(
                self, hidden_states, prefix_sum, residual
            )
            hidden_states = self._run_self_attn(positions, hidden_states)
            hidden_states, prefix_sum, residual = ar_window.post(
                self, hidden_states, prefix_sum, residual
            )
            hidden_states = self.mlp(hidden_states)
            return hidden_states, prefix_sum, residual

        hidden_states, prefix_sum, residual = self._pre_attn_norm(
            hidden_states, residual, prefix_sum
        )
        assert hidden_states is not None

        M = None
        if self.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)
            # Remove SP padding before attention.
            hidden_states = hidden_states[: positions.shape[0]]
            M = hidden_states.shape[0]

        # Attention.
        hidden_states = self._run_self_attn(positions, hidden_states)
        if self.ep_ar_mode:
            hidden_states = reduce_sensitive(hidden_states)
        if self.attn_res_window_enabled:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)

        # GEMM-RS returns the local sequence shard; standard O-proj preserves M.
        if self.use_sequence_parallel and hidden_states.shape[0] == M:
            # Add SP padding if needed, and then perform reduce scatter.
            hidden_states = sp_reduce_scatter(hidden_states)

        hidden_states, prefix_sum, residual = self._post_attn_norm(
            hidden_states, residual, prefix_sum
        )

        # MoE/MLP.
        hidden_states = self.mlp(hidden_states)
        if self.attn_res_window_enabled:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        return hidden_states, prefix_sum, residual


class K3LinearModel(_K3ParallelPolicy, KimiLinearModel):
    decoder_layer_cls = K3DecoderLayer

    def __init__(self, *, vllm_config, prefix=""):
        self.k3_execution = execution_config(vllm_config)
        self.ep_ar_mode = self.k3_execution.ep_allreduce
        if self.k3_execution.feature_striped:
            install()
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.attn_res_window_enabled = all(
            layer.attn_res_window_enabled
            for layer in self.layers[self.start_layer : self.end_layer]
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        assert hidden_states is not None

        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding, hidden_states
                )
            hidden_states = sp_shard(hidden_states)
            assert residual is None, "Currently, SP is not supported with PP"

        # sharded aux hidden states when sp is enabled
        aux_hidden_states: list[torch.Tensor] = []
        if self.start_layer in self.aux_hidden_state_layers:
            if self.use_attn_res or residual is None:
                aux_hidden_states.append(hidden_states)
            else:
                aux_hidden_states.append(hidden_states + residual)

        ar_window = None
        if (
            self.attn_res_window_enabled
            and self.use_attn_res
            and not self.aux_hidden_state_layers
            and attn_res_window_eligible(hidden_states)
        ):
            ar_window = AttnResTokenWindow(hidden_states, self.num_attn_res_blocks)
        prefix_sum = None
        if ar_window is not None:
            prefix_sum, residual = ar_window.initial_state(hidden_states)
            hidden_states = None
        elif self.use_attn_res:
            block_residual = hidden_states.new_empty(
                hidden_states.size(0),
                self.num_attn_res_blocks,
                hidden_states.size(1),
            )
            if residual is not None:
                block_residual[:, : residual.size(1), :].copy_(residual)
            prefix_sum = hidden_states
            hidden_states = None
            residual = block_residual

        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer],
            start=self.start_layer,
        ):
            hidden_states, prefix_sum, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                prefix_sum=prefix_sum,
                residual=residual,
                attn_res_window=ar_window,
            )
            if (layer_idx + 1) in self.aux_hidden_state_layers:
                if self.use_attn_res:
                    assert prefix_sum is not None
                    assert residual is not None
                    aux_hidden_state = self._capture_aux_hidden_stream(
                        layer_idx, prefix_sum, hidden_states, residual
                    )
                else:
                    assert residual is not None
                    aux_hidden_state = hidden_states + residual

                aux_hidden_states.append(aux_hidden_state)

        assert hidden_states is not None
        assert residual is not None
        if not get_pp_group().is_last_rank:
            assert not self.use_sequence_parallel, (
                "Currently, SP is not supported with PP"
            )
            if prefix_sum is not None:
                hidden_states = hidden_states + prefix_sum
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if ar_window is not None:
            hidden_states = ar_window.finish(self, hidden_states, prefix_sum, residual)
        elif self.use_attn_res:
            assert prefix_sum is not None
            hidden_states = attn_res(
                prefix_sum,
                hidden_states,
                residual,
                self.output_attn_res_norm.weight,
                self.output_attn_res_proj.weight.squeeze(0),
                None,
                num_blocks=self.num_attn_res_blocks,
                block_write_idx=-1,
                eps=self.output_attn_res_norm.variance_epsilon,
                output_norm_eps=0.0,
            )
        else:
            hidden_states = hidden_states + residual

        if self.use_sequence_parallel:
            if aux_hidden_states:
                hidden_size = hidden_states.shape[-1]
                packed_hidden_states = torch.cat(
                    [hidden_states, *aux_hidden_states], dim=-1
                )
                packed_hidden_states = sp_all_gather(packed_hidden_states)
                packed_hidden_states = packed_hidden_states[:full_num_tokens]
                hidden_states, *aux_hidden_states = packed_hidden_states.split(
                    hidden_size, dim=-1
                )
            else:
                hidden_states = sp_all_gather(hidden_states)
                hidden_states = hidden_states[:full_num_tokens]

        # NOTE: the final norm is applied in compute_logits instead of here, so
        # the MTP draft model receives the pre-norm hidden states.
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class K3LinearForCausalLM(KimiLinearForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        policy = execution_config(vllm_config)
        self.model_cls = K3LinearModel if policy.enabled else KimiLinearModel
        super().__init__(vllm_config=vllm_config, prefix=prefix)


class K3ForConditionalGeneration(KimiK3ForConditionalGeneration):
    vision_tower_cls = K3VisionModel

    def _maybe_ignore_quant_config(
        self, quant_config: QuantizationConfig | None
    ) -> QuantizationConfig | None:
        if isinstance(
            quant_config, (compressed_tensors.CompressedTensorsConfig, Mxfp4Config)
        ):
            return None
        return quant_config
