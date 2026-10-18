# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""M3-local gate subclass. Original GateLinear and other models remain untouched."""

import torch
from vllm_fl.minimax_m3_config import load, thaw
from flaggems_vllm.ops.minimax_m3.router import fused_kernel
from vllm.model_executor.layers.fused_moe import GateLinear


def router_bf16_load_fp32_dot(hidden, weight):
    m, n, k = load().linear["router_shape"]
    assert hidden.shape == (m, k) and hidden.dtype == torch.bfloat16
    assert weight.shape == (n, k) and weight.dtype == torch.float32
    assert hidden.is_cuda and weight.device == hidden.device
    assert hidden.is_contiguous() and weight.is_contiguous()
    out = torch.empty((m, n), device=hidden.device, dtype=torch.float32)
    config = thaw(load().linear["router_config"])
    fused_kernel[(m // config["BLOCK_SIZE_M"], n // config["BLOCK_SIZE_N"])](
        hidden, weight, weight, out, m, n, k, k, 1, k, 1, n, 1, 0, BIAS=False, **config
    )
    return out


@torch.library.custom_op("vllm_fl::m3_router_bf16_fp32", mutates_args=())
def bf16_load_fp32_dot(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return router_bf16_load_fp32_dot(hidden, weight)


@bf16_load_fp32_dot.register_fake
def _fake(hidden, weight):
    return hidden.new_empty((hidden.shape[0], weight.shape[0]), dtype=torch.float32)


class M3GateLinear(GateLinear):
    def forward(self, hidden):
        weight = getattr(self, "weight", None)
        m, n, k = load().linear["router_shape"]
        ok = (
            not torch.is_grad_enabled()
            and not self.training
            and isinstance(hidden, torch.Tensor)
            and hidden.ndim == 2
            and tuple(hidden.shape) == (m, k)
            and hidden.dtype == torch.bfloat16
            and hidden.is_cuda
            and hidden.is_contiguous()
            and weight is not None
            and tuple(weight.shape) == (n, k)
            and weight.dtype == torch.float32
            and weight.device == hidden.device
            and weight.is_contiguous()
            and getattr(self, "out_dtype", None) == torch.float32
            and getattr(self, "bias", None) is None
            and getattr(self, "return_bias", True)
        )
        if ok:
            return bf16_load_fp32_dot(hidden, weight), None
        return super().forward(hidden)
