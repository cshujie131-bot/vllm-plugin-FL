# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

# Copyright (c) 2025 BAAI. All rights reserved.
"""Explicit M3 sigmoid-bias router; the native router classes remain unchanged."""

import torch
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
    fused_topk_bias,
)
from vllm_fl.dispatch import CachedOp

_grouped_topk = CachedOp("m3_grouped_topk")


class FusedTopKBiasRouterFL(FusedTopKBiasRouter):
    """FL router that routes topk_softmax (with bias) through call_op."""

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scoring_func == "sigmoid" and self.e_score_correction_bias is not None:
            # One group selects from all experts; bias affects selection only.
            topk_weights, topk_ids = _grouped_topk(
                router_logits,
                1,
                1,
                self.top_k,
                self.renormalize,
                1.0,
                self.e_score_correction_bias.data,
                1,
            )
            if indices_type is not None:
                topk_ids = topk_ids.to(indices_type)
        else:
            topk_weights, topk_ids = fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=router_logits,
                e_score_correction_bias=self.e_score_correction_bias.data
                if self.e_score_correction_bias is not None
                else None,
                topk=self.top_k,
                renormalize=self.renormalize,
                scoring_func=self.scoring_func,
                indices_type=indices_type,
            )

        if self.routed_scaling_factor != 1.0:
            topk_weights *= self.routed_scaling_factor

        return topk_weights, topk_ids
