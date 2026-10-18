"""MXFP4 loading with bounded BF16 expert execution through vLLM's router."""

from vllm_fl.k3.context import execution_config

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import mxfp4_w4a16_moe_quant_config
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config, Mxfp4MoEMethod

import torch
from flaggems_vllm.ops.kimi_k3_metax.tp8_experts import (
    prepare as prepare_tp8_experts,
    run as run_tp8_experts,
)
from flaggems_vllm.ops.kimi_k3_metax.grouped_moe import grouped_experts
from flaggems_vllm.ops.kimi_k3_metax.ep_packed import (
    prepare as prepare_ep_experts,
    run as run_ep_experts,
)
from flaggems_vllm.ops.kimi_k3_metax.marlin_mxfp4 import repack_experts
from flaggems_vllm.ops.kimi_k3_metax.decode_experts import (
    prepare_decode_experts,
    grouped_experts_decode,
)
from vllm.logger import init_logger
import json

logger = init_logger(__name__)
from flag_gems import runtime
from vllm.model_executor.utils import replace_parameter


class K3Mxfp4Config(Mxfp4Config):
    """Use vLLM's canonical MXFP4 config after checkpoint normalization."""

    # The grouped BF16 kernel accepts every complete MXFP4 group.
    model_family = "kimi_k3"
    min_moe_intermediate_per_partition = 32
    shard_latent_projections = True

    def _make_moe_method(self, moe):
        return K3Bf16MoEMethod(moe)


class K3Bf16MoEMethod(Mxfp4MoEMethod):
    """Keep vLLM routing/shared experts/TP reduction and original packed weights.

    Expert token alignment, tile dequantization, both grouped GEMMs and output
    combination execute on GPU. Workspace sizes use shape metadata only.
    """

    def __init__(self, moe):
        self.k3_execution = execution_config()
        FusedMoEMethodBase.__init__(self, moe)
        if moe.activation != MoEActivation.SITU or moe.has_bias:
            raise NotImplementedError("The verified K3 expert path is bias-free SiTU")
        if moe.ep_size > 1:
            if (
                (not moe.is_sequence_parallel and not self.k3_execution.ep_allreduce)
                or moe.dp_size < 2
                or moe.moe_parallel_config.all2all_backend != "allgather_reducescatter"
            ):
                raise NotImplementedError("K3 EP requires DP/SP and AG/RS")
        elif moe.dp_size != 1 or moe.is_sequence_parallel:
            raise NotImplementedError("K3 non-EP requires ordinary TP")
        self.weight_dtype = "mxfp4"
        self.weights_ready = False
        self.use_packed_experts = False
        self.decode_state = None
        self.ep_prepared = None
        self.tp8_prepared = None
        self.prefill_workspace = None
        self.beta = moe.activation_situ_beta
        self.linear_beta = moe.activation_situ_linear_beta

    @property
    def supports_eplb(self):
        return False

    @property
    def skip_forward_padding(self):
        return False

    @property
    def has_unpadded_output(self):
        return False

    def maybe_roundup_sizes(
        self,
        hidden_size,
        intermediate_size_per_partition,
        act_dtype,
        moe_parallel_config,
    ):
        if hidden_size % 32 or intermediate_size_per_partition % 32:
            raise ValueError(
                "MXFP4 dimensions must align with the checkpoint's group32 scales"
            )
        return hidden_size, intermediate_size_per_partition

    def get_fused_moe_quant_config(self, layer):
        return mxfp4_w4a16_moe_quant_config(
            w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale
        )

    def process_weights_after_loading(self, layer):
        if self.weights_ready:
            return
        for scales in (layer.w13_weight_scale, layer.w2_weight_scale):
            if bool((scales == 255).any()):
                raise ValueError("Non-finite E8M0 expert scale in checkpoint")
        if self.moe.ep_size > 1:
            if self.moe.ep_size != 64 or self.moe.dp_size != 8:
                raise NotImplementedError(
                    "This EP layout is qualified for TP8/DP8/EP64"
                )
            values, self.ep_prepared = prepare_ep_experts(
                layer.w13_weight,
                layer.w13_weight_scale,
                layer.w2_weight,
                layer.w2_weight_scale,
            )
            names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
            for name, value in zip(names, values):
                if name.endswith("_scale"):
                    value.quant_method = "block"
                replace_parameter(layer, name, value)
            self.weights_ready = True
            logger.info("K3_EP_PACKED_PREPARED %s", json.dumps(self.ep_prepared))
            return
        if (
            runtime.device.vendor_name == "metax"
            and self.moe.tp_size == 8
            and tuple(layer.w13_weight.shape) == (896, 768, 1792)
            and tuple(layer.w2_weight.shape) == (896, 3584, 192)
            and self.beta == 4.0
            and self.linear_beta == 25.0
        ):
            values, self.tp8_prepared = prepare_tp8_experts(
                layer.w13_weight,
                layer.w13_weight_scale,
                layer.w2_weight,
                layer.w2_weight_scale,
            )
            names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
            for name, value in zip(names, values):
                if name.endswith("_scale"):
                    value.quant_method = "block"
                replace_parameter(layer, name, value)
            self.weights_ready = True
            logger.info("K3_TP8_PACKED_PREPARED %s", json.dumps(self.tp8_prepared))
            return
        if (
            runtime.device.vendor_name == "metax"
            and tuple(layer.w13_weight.shape) == (896, 192, 1792)
            and tuple(layer.w2_weight.shape) == (896, 3584, 48)
            and self.beta == 4.0
            and self.linear_beta == 25.0
        ):
            values = repack_experts(
                layer.w13_weight,
                layer.w13_weight_scale,
                layer.w2_weight,
                layer.w2_weight_scale,
            )
            names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
            for name, value in zip(names, values):
                if name.endswith("_scale"):
                    value.quant_method = "block"
                replace_parameter(layer, name, value)
            self.use_packed_experts = True
            self.decode_state = prepare_decode_experts(
                tuple(getattr(layer, n) for n in names)
            )
            logger.info(
                "K3_DECODE_PREPARED %s",
                json.dumps(
                    {
                        "normal_flags": self.decode_state["normal_flags"],
                        "scale_ranges": self.decode_state["scale_ranges"],
                        "sms": self.decode_state["sms"],
                        "weight_pointers": self.decode_state["pointers"],
                    }
                ),
            )
        self.weights_ready = True

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if not self.weights_ready:
            raise RuntimeError("Expert weights have not completed loading")
        if layer.apply_router_weight_on_input:
            raise NotImplementedError("K3 weights outputs after G2")
        if self.moe.ep_size > 1:
            if layer.expert_map is None:
                raise ValueError("EP requires the canonical global-to-local map")
            if self.ep_prepared is None:
                raise RuntimeError("EP packed weights were not prepared")
            return run_ep_experts(
                x,
                (
                    layer.w13_weight,
                    layer.w13_weight_scale,
                    layer.w2_weight,
                    layer.w2_weight_scale,
                ),
                topk_weights,
                topk_ids,
                layer.expert_map,
                self.ep_prepared,
                bn=128,
                warps=8,
                gm=8,
            )
        if layer.expert_map is not None:
            raise ValueError("Unexpected EP map in ordinary TP")
        if self.tp8_prepared is not None:
            return run_tp8_experts(
                x,
                (
                    layer.w13_weight,
                    layer.w13_weight_scale,
                    layer.w2_weight,
                    layer.w2_weight_scale,
                ),
                topk_weights,
                topk_ids,
                self.tp8_prepared,
                workspace=self.prefill_workspace,
            )
        if self.use_packed_experts:
            return grouped_experts_decode(
                x,
                (
                    layer.w13_weight,
                    layer.w13_weight_scale,
                    layer.w2_weight,
                    layer.w2_weight_scale,
                ),
                topk_weights,
                topk_ids,
                self.decode_state,
            )
        return grouped_experts(
            x,
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.w2_weight,
            layer.w2_weight_scale,
            topk_weights,
            topk_ids,
            beta=self.beta,
            linear_beta=self.linear_beta if self.linear_beta is not None else -1.0,
        )
