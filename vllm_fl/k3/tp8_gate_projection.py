"""Local TP8 MLA output gate, using a qualified small-M split-K projection."""

import torch
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
)
from vllm_fl.k3.tp8_linear import _PARTIAL, _REDUCE


class _SmallSplitProjection:
    """Same BF16 products and output rounding; FP32 K-sum is partitioned."""

    def forward(self, x):
        w = self.weight
        if not (
            self.bias is None
            and self.return_bias
            and not self.gather_output
            and x.shape == (8, 7168)
            and w.shape == (1536, 7168)
            and x.dtype == w.dtype == torch.bfloat16
            and x.is_cuda
            and x.device == w.device
            and x.is_contiguous()
            and w.is_contiguous()
        ):
            return super().forward(x)
        split = self._k_splits
        stages = self._pipeline_stages
        partial = torch.empty((split, 8, 1536), device=x.device, dtype=torch.float32)
        out = torch.empty((8, 1536), device=x.device, dtype=torch.bfloat16)
        _PARTIAL[(24, split)](
            x,
            w.T,
            partial,
            8,
            1536,
            7168,
            7168,
            1,
            1,
            7168,
            SPLIT_K=split,
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_K=64,
            pipeline="null",
            num_warps=4,
            num_stages=stages,
            num_ctas=1,
        )
        _REDUCE[(48,)](
            partial,
            out,
            8,
            1536,
            1536,
            1,
            SPLIT_K=split,
            BLOCK_SIZE=256,
            num_warps=4,
        )
        return out, None


class Tp8GateProjection(_SmallSplitProjection, ColumnParallelLinear):
    """MLA output sigmoid gate; original column-parallel weight loader."""

    _k_splits = 8
    _pipeline_stages = 3


class Tp8SharedGateUpProjection(_SmallSplitProjection, MergedColumnParallelLinear):
    """Shared MLP gate/up; retain the merged checkpoint weight loader."""

    _k_splits = 16
    _pipeline_stages = 4
