"""Model-scoped materialized-sigmoid routing through the ordinary FL registry."""

import torch
from flag_gems import grouped_topk as native_grouped_topk, runtime
from flaggems_vllm.ops.grouped_topk import grouped_topk as materialized_grouped_topk
from vllm_fl.dispatch import get_default_manager
from vllm_fl.utils import use_flaggems_op
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


def grouped_topk(
    scores,
    n_group,
    topk_group,
    topk,
    renormalize,
    routed_scaling_factor,
    bias,
    scoring_func=0,
):
    if (
        runtime.device.vendor_name == "metax"
        and scores.ndim == 2
        and 1 <= scores.shape[0] <= 4096
        and scores.shape[1] == 896
        and n_group == topk_group == 1
        and topk == 16
        and renormalize
        and routed_scaling_factor == 1.0
        and scoring_func == 1
        and scores.dtype in (torch.bfloat16, torch.float32)
        and bias.dtype in (torch.bfloat16, torch.float32)
        and scores.is_cuda
        and bias.device == scores.device
        and scores.stride(1) == 1
        and bias.numel() == 896
        and bias.is_contiguous()
    ):
        return materialized_grouped_topk(
            scores,
            n_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
            bias,
            scoring_func,
            sigmoid_mode="fp32_then_input",
        )
    return native_grouped_topk(
        scores,
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )


def register():
    """Called once at K3 worker initialization, before model/Graph construction."""
    if not use_flaggems_op("grouped_topk"):
        return
    manager = get_default_manager()
    manager.ensure_initialized()
    identity = "flagos.kimi_k3.grouped_topk"
    existing = manager.registry.get_implementation("grouped_topk", identity)
    if existing is not None:
        if existing.fn is not grouped_topk:
            raise RuntimeError("Conflicting K3 grouped-topk provider")
        return
    manager.registry.register_impl(
        OpImpl(
            op_name="grouped_topk",
            impl_id=identity,
            kind=BackendImplKind.DEFAULT,
            fn=grouped_topk,
            priority=BackendPriority.DEFAULT + 10,
        )
    )
    manager.bump_policy_epoch()
