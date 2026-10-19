# Changes against pinned upstream sources

Comparison bases are public source revisions, not the earlier M3 PR:

- FL: [ca6fb176](https://github.com/flagos-ai/vllm-plugin-FL/tree/ca6fb176ba704e0022736fd143539cef017032ee), the vLLM 0.24 main branch used by this PR.
- vLLM: [ee0da84a](https://github.com/vllm-project/vllm/tree/ee0da84ab9e04ac7610e28580af62c365e898389), v0.24.0.
- FlagGems: [a7620cc1](https://github.com/flagos-ai/FlagGems/tree/a7620cc191a0b42e040194622c5758b22a7a25dc).
- FlagGems-vllm: [b5c2c224](https://github.com/flagos-ai/FlagGems-vllm/tree/b5c2c2249319277ccda4d41aa0f65164bd9b2a19).

## Framework differences

| Area | Pinned upstream implementation | Actual difference in this PR |
|---|---|---|
| MoE pipeline | FL `ops/fused_moe/fused_moe_utils.py`: backend selection, assignment, `TritonExpertsFL.apply` | Small explicit extension points for expert class, assignment, per-call launch context, GEMM and activation. Default paths preserve upstream behavior. |
| M3 MoE | The shared FL pipeline above | `ops/minimax_m3/experts.py` inherits that pipeline and contains only M3 dispatch bindings, G1/G2 context and activation overrides. `moe.py` retains the accepted shape/tile/GEMV policy. No second backend-selector or full apply implementation. |
| Biased router | FL `ops/fused_moe/router.py::FusedTopKBiasRouterFL` | M3 adds the sigmoid-plus-bias branch; other cases delegate to the upstream router. |
| MoE construction | vLLM `fused_moe/layer.py::FusedMoE`, `routed_experts.py::RoutedExperts` | Use explicit router/expert constructor arguments and a local quant-method subclass. Do not copy the native constructor or replace global factories. |
| Text/vision model | vLLM `models/minimax_m3/nvidia/model.py`, `common/vision_tower.py` | Changed norm, MoE construction, QKV insertion and vision methods only, plus model-scoped bindings. Loading, metadata and multimodal preprocessing remain upstream. |
| Vision partial RoPE | FL #560: native M3 vision attention → ApplyRotaryEmb → apply_rotary_emb dispatch | Reused directly. The M3-specific attention override, RoPE implementation, schema and registration are removed. FP32 rotary computation and the two unrotated tail dimensions are preserved. |
| Warmup | vLLM `model_executor/warmup/kernel_warmup.py`, `minimax_m3_msa_warmup.py` | Reused directly. The copied M3 warmup files were removed. |
| MSA orchestration | vLLM M3 caller/cache semantics | M3-owned TP16/head layout guards, query packing, length/shape selection and launch metadata. These are framework adaptations, not newly invented attention math. |
| Torch schemas and dispatch | FL dispatch and PyTorch custom-op interfaces | `torch_ops.py`, `registration.py`, `layers.py`, `msa.py` expose the M3 calls and fake/mutation contracts. Schema/wrapper lines are not new numerical kernels. |
| SDK and communication | FL MetaX SDK wrappers | Reuse the existing wrappers, add the M3-scoped ABI compatibility and retained graph/collective boundary. Vendor MCCL remains the communication implementation. |

## Migrated kernels and compiler

**Source origin and compiler are separate.** The imported kernels originate in
vLLM and still compile through **FlagTree 0.6.1+metax3.6 / Triton API 3.6.0**.
The launcher selects FlagTree before imports. Startup checks both its package
version and that the actual `triton.__file__` is under `FLAGTREE_ROOT`.
This source preparation does not switch execution to another Triton installation
or a native binary operator.

| Imported source at vLLM ee0da84a | FGV destination | Actual delta |
|---|---|---|
| `models/minimax_m3/common/ops/index_topk.py` | `ops/minimax_m3/index_native.py` | Remove framework dependencies; retain MetaX-compatible launch choices and exact decode selection. The seven underlying JIT function bodies are reused. |
| `models/minimax_m3/common/ops/sparse_attn.py` | `ops/minimax_m3/sparse_native.py` | Remove framework dependencies; explicit head/stage policy and decode launch settings. The three underlying JIT function bodies are reused. |

Dependency preparation reads exactly these two files from the pinned public
source, verifies their hashes, and copies them into the new FGV checkout before
applying patch 01. The PR therefore contains their **incremental changes**, not
a full added copy of the original vLLM code. Neither the source checkout nor an
installed vLLM package is modified; FGV has no runtime import of vLLM or FL.

The other MSA files implement the accepted additional execution paths:
vector/dot index scoring, packed prefill, subtile decode and long-context launch
variants. The Linear, norm/QKV and router/GEMV patches retain their existing
numerical implementations. Moving an implementation or selecting FlagTree is
not presented as a new algorithm or a new performance measurement.

## Validation of this reduction

The generic MoE default path expands to the pinned FL statements, and the M3
MetaX path expands to the accepted implementation, preserving argument order,
workspace use, quantization, activation, LoRA ordering and reduction.
Dependency preparation reconstructs the same operator files as before; compiler
selection, tuning policy and numerical kernels remain unchanged.

The partial-RoPE replacement also passed 25 standalone cases on MetaX C550
with FlagTree, including three graph/eager comparisons. It uses the shared
default.flagos dispatch and FGV rotary kernel. Old/new results meet the accepted
BF16 tolerance but are not all bitwise identical. The installed-entry TP16
smoke remains pending; this PR stays Draft.
