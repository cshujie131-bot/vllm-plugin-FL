"""Store K3 latent projections as TP shards while retaining the original tail."""

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed import get_ep_group

from vllm_fl.k3.context import execution_config

from vllm_fl.k3.latent_d import project_grouped
from flag_gems import runtime
from flag_gems.ops.addmm import addmm_out, _K3_ADDMM_KERNEL
import torch
from vllm_fl.k3.ep_striped import FeatureStripedAgRs
from .ep_ar import reduce_mlp
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.models.common.ops.sequence_parallel import sp_all_gather, sp_shard
from vllm.models.kimi_k3.nvidia.latent_moe_runner import (
    LatentMoERunner,
    LatentTailTier,
)


def accumulate_latent_up(hidden_shard, latent, weight):
    """Direct U accumulation for verified TP8/TP32 layouts and disjoint inputs."""
    m = latent.shape[0]
    if (
        runtime.device.vendor_name == "metax"
        and m in (1, 2, 4, 8, 16, 32, 64, 128, 512)
        and latent.shape == (m, 3584)
        and weight.shape == (896, 3584)
        and hidden_shard.shape == (m, 896)
        and latent.dtype == weight.dtype == hidden_shard.dtype == torch.bfloat16
        and latent.is_cuda
        and latent.device == weight.device == hidden_shard.device
        and latent.is_contiguous()
        and weight.is_contiguous()
        and hidden_shard.stride(1) == 1
        and (m == 1 or hidden_shard.stride(0) == 7168)
        and hidden_shard.untyped_storage().data_ptr()
        not in (
            latent.untyped_storage().data_ptr(),
            weight.untyped_storage().data_ptr(),
        )
    ):
        # Same canonical addmm arithmetic, now tiled for the TP8 U shard.
        bm, bn, bk, warps, stages = (
            (32, 64, 64, 4, 2) if m == 512 else (16, 32, 32, 4, 4)
        )
        wt = weight.t()
        _K3_ADDMM_KERNEL[((m + bm - 1) // bm, (896 + bn - 1) // bn)](
            latent,
            wt,
            hidden_shard,
            hidden_shard,
            1,
            1,
            m,
            896,
            3584,
            latent.stride(0),
            latent.stride(1),
            wt.stride(0),
            wt.stride(1),
            hidden_shard.stride(0),
            hidden_shard.stride(1),
            hidden_shard.stride(0),
            hidden_shard.stride(1),
            BIAS_IS_VECTOR=False,
            BIAS_IS_SCALAR=False,
            HAS_K=True,
            IS_FP64=False,
            BLOCK_SIZE_M=bm,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=bk,
            num_warps=warps,
            num_stages=stages,
        )
        return hidden_shard
    if (
        runtime.device.vendor_name == "metax"
        and m in (1, 2, 4, 8, 16, 32, 64, 1024, 3553, 4096)
        and latent.shape == (m, 3584)
        and weight.shape == (224, 3584)
        and hidden_shard.shape == (m, 224)
        and latent.dtype == weight.dtype == hidden_shard.dtype == torch.bfloat16
        and latent.is_contiguous()
        and weight.is_contiguous()
        and hidden_shard.stride(1) == 1
        and (m == 1 or hidden_shard.stride(0) == 7168)
        and hidden_shard.untyped_storage().data_ptr()
        not in (
            latent.untyped_storage().data_ptr(),
            weight.untyped_storage().data_ptr(),
        )
    ):
        if (
            m in (2, 4, 8, 16, 32)
            and latent.is_cuda
            and weight.device == latent.device == hidden_shard.device
        ):
            # Qualified Graph2/4/8/16/32: expose more N tiles.
            # Reuse the canonical addmm arithmetic and in-place output semantics.
            wt = weight.t()
            _K3_ADDMM_KERNEL[((m + 15) // 16, 7)](
                latent,
                wt,
                hidden_shard,
                hidden_shard,
                1,
                1,
                m,
                224,
                3584,
                latent.stride(0),
                latent.stride(1),
                wt.stride(0),
                wt.stride(1),
                hidden_shard.stride(0),
                hidden_shard.stride(1),
                hidden_shard.stride(0),
                hidden_shard.stride(1),
                BIAS_IS_VECTOR=False,
                BIAS_IS_SCALAR=False,
                HAS_K=True,
                IS_FP64=False,
                BLOCK_SIZE_M=16,
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=32,
                num_warps=4,
                num_stages=4,
            )
            return hidden_shard
        return addmm_out(hidden_shard, latent, weight.t(), out=hidden_shard)
    return hidden_shard.addmm_(latent, weight.t())


class LatentColumnParallelLinear(ColumnParallelLinear):
    """Use FlagGems mm with row tiling and vLLM's canonical weight loader."""

    def __init__(self, *args, sequence_parallel: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_parallel = sequence_parallel
        if sequence_parallel and not self.gather_output:
            raise ValueError("SP latent projection must restore the full feature axis")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        if x.ndim != 2 or self.bias is not None or not self.return_bias:
            raise ValueError("K3 latent projections require a bias-free 2D input")
        # Different TP ranks own different token rows under SP. Gather rows
        # first, then gather projected columns, and restore the owned rows.
        if self.sequence_parallel:
            x = sp_all_gather(x)
        output = self.project_local(x)
        if self.gather_output and self.tp_size > 1:
            output = tensor_model_parallel_all_gather(output)
        if self.sequence_parallel:
            output = sp_shard(output)
        return output, None

    def project_local(
        self, x: torch.Tensor, *, out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        return project_grouped(x, self.weight, out_dtype=out_dtype)


class ShardedLatentMoERunner(LatentMoERunner):
    """Use the existing column tail with a physically sharded up weight."""

    def __init__(self, *args, **kwargs):
        self.k3_execution = execution_config()
        super().__init__(*args, **kwargs)
        self.defer_attn_res_reduction = False
        sequence_parallel = self.moe_config.is_sequence_parallel
        if sequence_parallel:
            if self.moe_config.ep_size <= 1 or self.moe_config.dp_size <= 1:
                raise NotImplementedError("K3 SP candidate requires DP and EP")
            if not getattr(
                self._shared_experts._layer, "shard_sequence_parallel", False
            ):
                raise ValueError("SP shared experts must keep TP-sharded weights")
        elif not self.k3_execution.ep_allreduce and (
            self.moe_config.ep_size != 1
            or self.moe_config.dp_size != 1
            or not self._use_fused_path()
        ):
            raise NotImplementedError("K3 non-SP tail requires ordinary TP")
        transform = self.routed_output_transform
        if transform is None or transform.norm is None:
            raise ValueError("The K3 latent output transform requires RMSNorm")
        projection = transform.up_proj
        if (
            (
                not sequence_parallel
                and projection.tp_size
                != (
                    get_tensor_model_parallel_world_size()
                    if self.k3_execution.ep_allreduce
                    else self.moe_config.tp_size
                )
            )
            or projection.gather_output != sequence_parallel
            or projection.weight.shape[0] * projection.tp_size != projection.output_size
        ):
            raise ValueError("Expected a column-parallel up-projection shard")

    def _use_fused_path(self):
        if (
            self.k3_execution.ep_allreduce
            and self.moe_config.ep_size == 64
            and not self.moe_config.is_sequence_parallel
        ):
            # MoE's tp_size is 1 under EP, while latent/shared projections
            # still use the physical TP8 group.
            return (
                self._shared_experts is not None and not self._fused_output_is_reduced
            )
        return super()._use_fused_path()

    def apply_routed_output_transform(self, hidden_states: torch.Tensor):
        if self.moe_config.is_sequence_parallel:
            # The EP reduce-scatter has completed its FP32 sum. Round once at
            # the existing BF16 latent boundary before RMSNorm and up projection.
            hidden_states = hidden_states.to(torch.bfloat16)
        return super().apply_routed_output_transform(hidden_states)

    def _select_tail_tier(
        self, fused_output: torch.Tensor, shared_output: torch.Tensor
    ) -> LatentTailTier:
        return LatentTailTier.COLUMN_PARALLEL

    def _shard_up_proj_tail(
        self,
        fused_output: torch.Tensor,
        shared_output: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:
        transform = self.routed_output_transform
        assert transform is not None and transform.norm is not None
        if self.k3_execution.ep_allreduce and self.moe_config.ep_size == 64:
            # DP8 RS has summed the 8 expert owners at this TP coordinate.
            # TP8 AR finishes the other coordinate before nonlinear RMSNorm.
            if (
                self.moe_config.is_sequence_parallel
                or fused_output.dtype != torch.float32
            ):
                raise ValueError("EP AR tail expects non-SP FP32 DP partials")
            manager = get_ep_group().device_communicator.all2all_manager
            complete = (
                isinstance(manager, FeatureStripedAgRs) and manager.output_is_complete()
            )
            latent = (
                fused_output
                if complete
                else tensor_model_parallel_all_reduce(fused_output)
            ).to(torch.bfloat16)
            latent = transform.norm(latent)
            up_local = transform.up_proj.project_local(latent)
            routed = tensor_model_parallel_all_gather(up_local)
            shared = reduce_mlp(shared_output)
            output = shared + routed
            return output[..., :trunc_size] if trunc_size is not None else output
        latent = self.allreduce_norm_latent_out(fused_output, transform.norm)
        weight = transform.up_proj.weight
        shard_start = get_tensor_model_parallel_rank() * weight.shape[0]
        hidden_shard = shared_output.narrow(-1, shard_start, weight.shape[0])
        accumulate_latent_up(hidden_shard, latent, weight)
        if self.defer_attn_res_reduction:
            # The decoder caller performs either the original AR or the
            # qualified token RS before the next AttnRes consumer.
            return (
                shared_output[..., :trunc_size]
                if trunc_size is not None
                else shared_output
            )
        return self._maybe_reduce_final_output(
            shared_output, trunc_size, output_is_reduced=False
        )
