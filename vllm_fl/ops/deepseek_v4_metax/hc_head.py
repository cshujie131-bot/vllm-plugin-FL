# SPDX-License-Identifier: Apache-2.0
"""Graph boundary for the existing FlagGems HC head; no library source edits."""

import torch
from flag_gems import hc_head_fused_kernel as _native_hc_head


@torch.library.custom_op("vllm_fl::dsv4_hc_head", mutates_args=("out",))
def hc_head(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    _native_hc_head(
        hs_flat, fn, hc_scale, hc_base, out, hidden_size, rms_eps, hc_eps, hc_mult
    )


@hc_head.register_fake
def _hc_head_fake(
    hs_flat, fn, hc_scale, hc_base, out, hidden_size, rms_eps, hc_eps, hc_mult
):
    return None
