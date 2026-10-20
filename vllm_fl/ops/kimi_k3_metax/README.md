# Kimi K3 source review map

These modules implement the TP8/PP4 MetaX path. A file newly added to FL is not
necessarily a new algorithm: the following table distinguishes framework-specific
work, upstream reuse and retained numerical/runtime compatibility.

## Upstream reuse and small differences

| FL module | Upstream implementation reused | Difference retained in FL |
|---|---|---|
| `kda_kernels/chunk.py` | vLLM K3 `recompute_w_u_fwd_kernel` and `chunk_gla_fwd_kernel_o` | Local tile/warp/stage policy, original host buffer lifetimes, and the existing FP64 gate-cumsum compatibility kernel |
| `kda_kernels/chunk_delta_h.py` | vLLM FLA `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | Local BV16/warp4/stage1 launch policy and the wrapper selecting that local handle |
| `kda_kernels/chunk_intra.py` | vLLM K3 inter/solve and sub-chunk kernels, plus the complete native token-parallel entry | Original local autotune candidates and the direct-load branch selection |
| `kda_kernels/launch.py` | Native Triton JIT objects | New local wrappers only; upstream exports, wrappers and configs are not modified |
| `../..` → `k3/model_adapter.py` | vLLM K3 model classes and loaders | Component selection and the changed methods only; no duplicate full model or loader |
| `k3/vision_adapter.py` | vLLM ViT classes and `MLP2.forward` | Bounded row processing and real-view complex layout operations |

The pinned vLLM source adds one optional `DEFAULT_DOT_PRECISION` constexpr to
five kernels across three existing files. Only previously implicit dot calls use
that parameter. Its default is `None`, preserving native behavior. FL supplies
`"ieee"` in its own tuning configurations, matching the previous explicit dot
arguments. Existing explicit precision arguments retain their precedence.
This is a source-level argument, not a compiler default or a global override.

The [precision-interface diff](https://github.com/cshujie131-bot/vllm/compare/cc1c8274e47fd9148cc49a8d9e0c02fb037b09f8...c88c603e1f3d3bde7ad18bd7299cb3d502f7b7a7) is confined to:

- `vllm/models/kimi_k3/nvidia/ops/third_party/kda/chunk.py`
- `vllm/models/kimi_k3/nvidia/ops/third_party/kda/chunk_intra.py`
- `vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py`

The local host wrappers remain because calling the native host wrappers would
select the upstream global launch handles. Their buffer allocation, aliasing,
call order and release points are preserved. The gate-cumsum compatibility
kernel remains explicit because the upstream FP32 accumulation is not the
validated local FP64 accumulation.

## Framework-specific implementations retained

| Files / family | Why a native call cannot replace this code directly |
|---|---|
| `tp8_experts.py`, `tp8_expert_g1.py`, `expert_gemm.py`, `prefill_mma.py`, `grouped_moe.py`, `decode_experts.py`, `mxfp4_kernels.py`, `decode_bits.py`, `marlin_mxfp4.py` | TP-local packed MXFP4/E8M0 layout, shard dimensions, SiTU boundary, ordered reduction and runner-owned workspace. The generic fused-expert APIs are not equivalent to this complete contract. |
| `mla_tp8.py`, `mla_staged_packing.py`, `mla4k_long.py`, `mla_decode*.py`, `mla_bmm.py`, `prefill_kernel.py` | TP-local heads, packing, split/reduce and short/long shape policies. Short-query and small-batch paths remain reachable. |
| `mla_dense_fallback.py` | The native dense MLA kernel assumes contiguous physical pages and dot-with-accumulator rounding. This fallback uses explicit page strides, int64 addresses and the accepted separate dot/add order. Those changes cannot be expressed by a launch tile alone. |
| `cache_gather.py`, `kda_preprocess.py`, `normalization.py` | Cache ABI and the retained KDA normalization/materialization contract. These are compatibility/adaptation code, not newly claimed library speedups. |
| `activations.py`, `small_alignment.py` | Framework shape/buffer policy around the pinned FGV SiTU and alignment entries. Pure operator implementations are not duplicated here. |
| `workspace_contract.py` and `k3/tp8_prefill_workspace.py` | Runner-owned allocation lifetime and stream registration; this is not an operator-library responsibility. |

`vllm_fl/k3/` otherwise contains component subclasses, TP projection/layout
selection, registry integration, graph boundaries and PP feedback. Constructors,
weight loading, tokenizer/processor logic and the model execution skeleton are
inherited from the pinned vLLM classes. The existing MetaX backend owns hardware
runtime and vendor-library integration.

No runtime source rewriting, exported-library function replacement, compiler
patch or extra environment switch is used to reduce this diff. The dependency
lock and the example's installation instructions remain the reproduction entry.
