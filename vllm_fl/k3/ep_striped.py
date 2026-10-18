"""K3 TP8/DP8 feature-striped dispatch/combine over existing native groups."""

import torch
from vllm.distributed import get_tp_group, get_dp_group, get_ep_group
from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
from vllm.forward_context import get_forward_context


def selected_for_shape(sizes, cg_mode):
    if sizes is None or len(sizes) != 8:
        return False
    m = sizes[0]
    if m <= 0 or any(n != m for n in sizes):
        return False
    mode = getattr(cg_mode, "name", str(cg_mode))
    # Small eager calls keep the baseline: extra packing/launches lost there.
    return mode == "FULL" and m in (1, 2, 4, 8, 16)


def selected():
    context = get_forward_context()
    mode = context.additional_kwargs.get(
        "cudagraph_capture_mode", getattr(context.cudagraph_runtime_mode, "name", "")
    )
    if mode != "FULL":
        return False
    metadata = context.dp_metadata
    if metadata is None:
        return False
    # Non-SP K3 uses the full per-DP counts. local_sizes is temporary inside
    # MoERunner and is already cleared when the latent output tail executes.
    counts = metadata.num_tokens_across_dp_cpu
    if counts.device.type != "cpu":
        raise TypeError("Striped selection requires existing CPU DP metadata")
    return selected_for_shape(counts.tolist(), mode)


class FeatureStripedAgRs(AgRsAll2AllManager):
    def dispatch_router_logits(
        self,
        hidden_states,
        router_logits,
        is_sequence_parallel=False,
        extra_tensors=None,
    ):
        if not selected():
            return super().dispatch_router_logits(
                hidden_states, router_logits, is_sequence_parallel, extra_tensors
            )
        if (
            is_sequence_parallel
            or extra_tensors is not None
            or hidden_states.shape[1] != 3584
            or router_logits.shape[1] != 896
            or hidden_states.dtype != torch.bfloat16
            or router_logits.dtype != torch.float32
        ):
            raise ValueError("K3 striped dispatch requires BF16 latent/FP32 logits")
        tp, dp = get_tp_group(), get_dp_group()
        slot = tp.rank_in_group
        sizes = self._get_sizes(hidden_states.shape[0], dp)
        h = hidden_states[:, slot * 448 : (slot + 1) * 448].contiguous()
        r = router_logits[:, slot * 112 : (slot + 1) * 112].contiguous()
        h, r = dp.all_gatherv([h, r], dim=0, sizes=sizes)
        h, r = tp.all_gatherv([h.T.contiguous(), r.T.contiguous()], dim=0, sizes=None)
        return h.T.contiguous(), r.T.contiguous()

    def dispatch(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        is_sequence_parallel=False,
        extra_tensors=None,
    ):
        if selected():
            raise NotImplementedError("K3 striped route uses dispatch_router_logits")
        return super().dispatch(
            hidden_states, topk_weights, topk_ids, is_sequence_parallel, extra_tensors
        )

    def combine(self, hidden_states, is_sequence_parallel=False):
        if not selected():
            return super().combine(hidden_states, is_sequence_parallel)
        if (
            is_sequence_parallel
            or hidden_states.shape[1] != 3584
            or hidden_states.dtype != torch.float32
        ):
            raise ValueError("K3 striped combine requires FP32 expert partials")
        tp, dp = get_tp_group(), get_dp_group()
        sizes = self._get_sizes(hidden_states.shape[0] // 8, dp)
        node_partial = tp.reduce_scatter(hidden_states, dim=1)
        local_tokens = dp.reduce_scatterv(node_partial, dim=0, sizes=sizes)
        # The return value is complete across all64 expert owners. Preserve
        # FP32 until the original caller's BF16 materialization/RMSNorm.
        return tp.all_gather(local_tokens, dim=1)

    def output_is_complete(self):
        return selected()


def install():
    tp, dp, ep = get_tp_group(), get_dp_group(), get_ep_group()
    if (tp.world_size, dp.world_size, ep.world_size) != (8, 8, 64):
        raise ValueError("K3 striped transport requires TP8/DP8/EP64")
    communicator = ep.device_communicator
    if communicator is None:
        raise RuntimeError("Expert communicator is not initialized")
    current = communicator.all2all_manager
    if isinstance(current, FeatureStripedAgRs):
        return True
    if type(current) is not AgRsAll2AllManager:
        raise TypeError("K3 striped transport requires the native AgRs backend")
    communicator.all2all_manager = FeatureStripedAgRs(ep.cpu_group)
    return True
