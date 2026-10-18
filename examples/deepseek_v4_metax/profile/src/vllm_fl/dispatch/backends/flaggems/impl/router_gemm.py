# Copyright (c) 2026 BAAI. All rights reserved.

"""FlagGems MoE router GEMM implementation."""

import torch

from flag_gems import router_gemm


def router_gemm_bf16_fp32_flaggems(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return router_gemm(x, weight)
