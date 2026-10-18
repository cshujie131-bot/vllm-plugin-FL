"""Initialize K3 EP communication buffers before visual activation profiling."""

import torch

from vllm.distributed import get_dp_group, get_ep_group, get_tp_group


def warmup_ep_collectives(
    device: torch.device,
    max_num_batched_tokens: int,
    latent_size: int,
    num_experts: int,
) -> None:
    """Exercise the same TP and EP collectives once, outside CUDA graphs.

    MCCL lazily allocates buffers for different collective algorithms. Its
    allocations do not reclaim PyTorch's inactive cache. Initialize those
    buffers before the maximum-image forward can fill that cache, so memory
    profiling accounts for the communication buffers as persistent runtime
    memory. Model tensors, arithmetic and per-request communication are unchanged.
    """
    tp, ep = get_tp_group(), get_ep_group()
    dp_size = get_dp_group().world_size
    if dp_size < 2 or ep.world_size != tp.world_size * dp_size:
        raise ValueError("Expected K3 TP/DP/EP groups without PP or PCP")

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    for group, dtype in ((tp, torch.bfloat16), (ep, torch.float32)):
        # A large message also initializes the channels used by prefill.
        rows = (max_num_batched_tokens + tp.world_size - 1) // tp.world_size
        features = 7168 if group is tp else latent_size
        send = torch.zeros(rows * features, dtype=dtype, device=device)
        gathered = group.all_gather(send, dim=0)
        reduced = group.reduce_scatter(gathered, dim=0)
        torch.cuda.synchronize(device)
        del send, gathered, reduced

    # Uneven DP batches use all_gatherv/reduce_scatterv (broadcast/reduce
    # internally), which can require different transport connections.
    max_local = max(
        (max_num_batched_tokens + tp.world_size - 1) // tp.world_size,
        dp_size,
    )
    sizes = [max_local - rank // tp.world_size for rank in range(ep.world_size)]
    local_tokens = sizes[ep.rank_in_group]
    hidden = torch.zeros(
        (local_tokens, latent_size), dtype=torch.bfloat16, device=device
    )
    logits = torch.zeros(
        (local_tokens, num_experts), dtype=torch.float32, device=device
    )
    hidden_all, logits_all = ep.all_gatherv([hidden, logits], dim=0, sizes=sizes)
    combined = ep.reduce_scatterv(hidden_all.float(), dim=0, sizes=sizes)
    torch.cuda.synchronize(device)
    del hidden, logits, hidden_all, logits_all, combined
    torch.cuda.empty_cache()
