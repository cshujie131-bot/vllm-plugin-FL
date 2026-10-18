"""Token-sharded AttnRes for the two measured large-M TP32 shapes.

Only A/B are split. Expert latent communication (C), projection GEMMs, routing,
and all small-M arithmetic retain their existing implementations.
"""

from vllm_fl.k3.latent_moe import ShardedLatentMoERunner

import torch
from vllm.distributed import get_tp_group
from vllm.platforms import current_platform
from vllm.triton_utils import triton, tl
from vllm.utils.torch_utils import current_stream
from vllm.models.kimi_k3.nvidia.ops.attn_res import _attn_res_kernel

ACTIVE_TOKEN_COUNTS = (3553, 4096)


def supported_config(config):
    p = config.parallel_config
    h = config.model_config.hf_text_config
    return (
        config.model_config.dtype == torch.bfloat16
        and p.tensor_parallel_size == 32
        and p.data_parallel_size == p.pipeline_parallel_size == 1
        and not p.enable_expert_parallel
        and not p.enable_dbo
        and not p.use_ubatching
        and h.hidden_size == 7168
        and h.attn_res_block_size == 12
        and bool(h.linear_attn_config.get("use_full_rank_gate", False))
    )


def configure_layer(layer, config):
    """Move only the two AR boundaries to the decoder caller, even on fallback.

    Inactive shapes still issue the original BF16 all_reduce immediately after
    the same projection/MLP, before any dependent arithmetic.
    """
    if not supported_config(config):
        return False
    if layer._self_attn_writes_output or layer.use_sequence_parallel:
        raise ValueError("AttnRes windows require ordinary TP K3 attention outputs")
    projection = layer.self_attn.o_proj
    if projection.bias is not None or not projection.reduce_results:
        raise ValueError("Expected unrevised bias-free attention O reduction")
    if layer.is_moe_layer:
        runner = layer.mlp.experts
        if (
            not isinstance(runner, ShardedLatentMoERunner)
            or not runner._use_fused_path()
        ):
            raise ValueError("Expected the frozen sharded latent MoE runner")
        runner.defer_attn_res_reduction = True
    else:
        if (
            layer.mlp.down_proj.bias is not None
            or not layer.mlp.down_proj.reduce_results
        ):
            raise ValueError("Expected bias-free dense down projection")
        layer.mlp.down_proj.reduce_results = False
    projection.reduce_results = False
    return True


def eligible(x):
    return (
        x.ndim == 2
        and x.shape[0] in ACTIVE_TOKEN_COUNTS
        and x.shape[1] == 7168
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
    )


from flaggems_vllm.ops.kimi_k3_metax.attn_res_padding import _bf16_pad


class AttnResTokenWindow:
    """One forward's token ownership and reusable communication scratch."""

    def __init__(self, hidden_states, num_blocks):
        if not eligible(hidden_states):
            raise ValueError("Unqualified AttnRes window shape")
        self.m, self.h = hidden_states.shape
        group = get_tp_group()
        if group.world_size != 32:
            raise ValueError("AttnRes windows require TP32")
        self.rank = group.rank_in_group
        self.comm = group.device_communicator.pynccl_comm
        if self.comm is None or not self.comm.available or self.comm.disabled:
            raise RuntimeError("Native TP communicator is unavailable")
        self.rows = triton.cdiv(self.m, 32)
        self.padded_m = self.rows * 32
        self.start = self.rank * self.rows
        self.valid = min(self.rows, self.m - self.start)
        self.padded_input = (
            hidden_states.new_empty((self.padded_m, self.h))
            if self.padded_m != self.m
            else None
        )
        # Padding remains zero across kernels; valid rows are overwritten.
        self.local_output = hidden_states.new_zeros((self.rows, self.h))
        self.full_output = hidden_states.new_empty((self.padded_m, self.h))
        self.num_blocks = num_blocks

    def initial_state(self, hidden_states):
        prefix = hidden_states.new_zeros((self.rows, self.h))
        prefix[: self.valid].copy_(hidden_states[self.start : self.start + self.valid])
        history = hidden_states.new_empty((self.rows, self.num_blocks, self.h))
        return prefix, history

    def reduce(self, partial):
        if partial.shape != (self.m, self.h) or partial.dtype != torch.bfloat16:
            raise ValueError("Expected full-token BF16 TP partial")
        if not partial.is_contiguous():
            partial = partial.contiguous()
        if self.padded_input is not None:
            _bf16_pad[(triton.cdiv(self.padded_m * self.h, 1024),)](
                partial,
                self.padded_input,
                self.m * self.h,
                self.padded_m * self.h,
                1024,
            )
            partial = self.padded_input
        # Do not reuse this buffer: block reset retains the result as prefix.
        reduced = partial.new_empty((self.rows, self.h))
        self.comm.reduce_scatter(reduced, partial, stream=current_stream())
        return reduced

    def mix(
        self,
        prefix,
        delta,
        history,
        norm,
        query,
        output_norm,
        num_blocks,
        block_write_idx,
        eps,
        output_norm_eps,
    ):
        _attn_res_kernel[(self.valid,)](
            prefix,
            delta,
            history,
            norm,
            query,
            output_norm,
            self.local_output,
            prefix.stride(0),
            0 if delta is None else delta.stride(0),
            history.stride(0),
            history.stride(1),
            self.local_output.stride(0),
            num_blocks,
            self.h,
            block_write_idx,
            eps,
            output_norm_eps,
            HAS_DELTA=delta is not None,
            WRITE_BLOCK=block_write_idx >= 0,
            APPLY_OUTPUT_NORM=output_norm is not None,
            BLOCK_L=1,
            BLOCK_D=8192,
            num_warps=4,
            num_stages=2,
            launch_pdl=current_platform.is_arch_support_pdl(),
        )
        self.comm.all_gather(
            self.full_output, self.local_output, stream=current_stream()
        )
        return self.full_output[: self.m]

    def pre(self, layer, hidden_states, prefix, history):
        delta = None if hidden_states is None else self.reduce(hidden_states)
        output = self.mix(
            prefix,
            delta,
            history,
            layer.self_attention_res_norm.weight,
            layer.self_attention_res_proj.weight.squeeze(0),
            layer.input_layernorm.weight,
            layer.prev_valid_blocks,
            layer.block_write_idx if layer.is_block_write_layer else -1,
            layer.self_attention_res_norm.variance_epsilon,
            layer.input_layernorm.variance_epsilon,
        )
        return output, prefix, history

    def post(self, layer, hidden_states, prefix, history):
        reduced = self.reduce(hidden_states)
        if layer.is_block_write_layer:
            prefix, delta = reduced, None
        else:
            delta = reduced
        output = self.mix(
            prefix,
            delta,
            history,
            layer.mlp_res_norm.weight,
            layer.mlp_res_proj.weight.squeeze(0),
            layer.post_attention_layernorm.weight,
            layer.prev_valid_blocks + layer.is_block_write_layer,
            -1,
            layer.mlp_res_norm.variance_epsilon,
            layer.post_attention_layernorm.variance_epsilon,
        )
        return output, prefix, history

    def finish(self, model, hidden_states, prefix, history):
        return self.mix(
            prefix,
            self.reduce(hidden_states),
            history,
            model.output_attn_res_norm.weight,
            model.output_attn_res_proj.weight.squeeze(0),
            None,
            model.num_attn_res_blocks,
            -1,
            model.output_attn_res_norm.variance_epsilon,
            0.0,
        )
