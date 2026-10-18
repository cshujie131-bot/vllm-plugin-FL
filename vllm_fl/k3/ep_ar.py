"""Explicit TP8/DP8/EP64 non-SP communication mode for independent validation."""

import torch
from vllm.distributed import tensor_model_parallel_all_reduce


def reduce_sensitive(x):
    # FP32 sensitive sum, then the original BF16 materialization boundary.
    return tensor_model_parallel_all_reduce(x.float()).to(x.dtype)


def reduce_mlp(x):
    # Only the M8 BF16 collective window was bit-exact in the prior tests.
    if x.shape[0] == 8:
        return tensor_model_parallel_all_reduce(x)
    return reduce_sensitive(x)
