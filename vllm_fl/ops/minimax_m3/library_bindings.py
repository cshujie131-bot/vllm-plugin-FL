# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""M3 library bindings; source package and numerical preparation are explicit."""

import torch
import triton
import flag_gems
from vllm.utils.math_utils import round_up


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from flag_gems.fused.moe_align_block_size import moe_align_block_size_triton

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    moe_align_block_size_triton(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        allow_tle=False,
    )
    if expert_map is not None:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad


def invoke_fused_moe_triton_kernel(*args, **kwargs):
    return flag_gems.invoke_fused_moe_triton_kernel(*args, **kwargs)


def moe_sum(inp, out):
    return flag_gems.moe_sum(inp, out)


def grouped_topk(*args, **kwargs):
    return flag_gems.grouped_topk(*args, **kwargs)
