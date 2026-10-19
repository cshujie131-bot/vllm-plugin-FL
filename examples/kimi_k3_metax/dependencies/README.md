# Pinned dependency deltas

The base commits and application order are in lock.json. Apply these patches
only to the dedicated K3 MetaX environment. They preserve the accepted numerical
paths, including existing library calls and fixed decorator-time tuning.

| Patch | Source and purpose | Upstream status |
| --- | --- | --- |
| 01 | K3 backend-lowering baseline: default component factories in K3 model/MLA/KDA/ViT | Carried here; no separate K3 PR yet |
| 02 | Accepted PP host feedback, input-copy hook, Graph metadata and CUSTOM prefill selection | Carried here; scheduler unchanged |
| 03 | Legal MetaX addmm configuration and shared TP8 arithmetic handle | Not claimed merged |
| 04 | Accepted argmax NaN/tie/index and shape corrections | Not claimed merged |
| 05 | Accepted grouped-topk implementation using the existing FlagGems route | Not replaced by a different fusion |
| 06 | Accepted MLA tiles and explicit dense-backend callable | No library-to-plugin import |
| 07 | Qualified K3 expert/MLA/cache/SiTU kernels relocated without mathematical changes | New library namespace, explicit imports |
| 08 | Accepted KDA specialization; derived from pinned vLLM/FLA source with attribution retained | Public default KDA files remain unchanged |

Implementation source: k3_tp8pp4_backend_lowering_r2, the unified version
subsequently used by text_accuracy_eval_20261017-r1 and
erqa_accuracy_eval_20261017-r1. These identifiers describe historical acceptance;
they are not filesystem dependencies.

The pinned FlagGems-vllm also has a general FLA KDA implementation. It is not
interchanged with this accepted K3 path: dtype, gate accumulation, chunk layout,
state and launch-resource choices differ. Independent new kernels retain full
source; wrappers that bind modified kernel globals remain alongside them.
Workspace allocation and communication/model orchestration remain in FL.

The fixed MetaX parameters are intentionally retained rather than adding a new
autotune sweep. Existing Torch/FlagGems fallback computations and precision
boundaries are preserved, as required for this publication-only task. They are
not presented as new universally portable operator-library acceptance.

EP/TP32-only entry points and kernels have been removed. Shared packed FP4
decode/scheduling primitives used by TP8 retain their accepted mathematical bodies.
