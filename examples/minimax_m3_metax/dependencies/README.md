# Operator dependencies

Use prepare_dependencies.py rather than applying the patches to an unprepared
tree. It first imports two source files from pinned public vLLM into the FGV
copy, then applies the patches in manifest order. The import paths and hashes
are explicit in manifest.json; the source checkouts remain read-only.

| Library | Public base | Patch group |
|---|---|---|
| FlagGems | a7620cc191a0b42e040194622c5758b22a7a25dc (v5.3.5) | 01 Linear kernels/policy; 02 generic masked_fill_ selection; 03 explicit alignment policy |
| FlagGems-vllm | b5c2c2249319277ccda4d41aa0f65164bd9b2a19 (release v0.2.0) | 01 MSA; 02 norm/activation/QKV; 03 MoE/router; 04 cache/import compatibility |

For index_native.py and sparse_native.py, the baseline is vLLM
ee0da84ab9e04ac7610e28580af62c365e898389, respectively
vllm/models/minimax_m3/common/ops/index_topk.py and sparse_attn.py.
Their unchanged computation is not counted as a new optimization. Patch 01
shows only the changes to these imported files, plus the genuinely additional
MSA paths. FGV does not import vLLM at runtime.

The remaining implementations originate from the accepted MiniMax M3 adaptation and
optimization work. vLLM and FlagGems copyright/license attribution is retained.
This packaging preserves the MetaX launch choices and numerical contract; it does
not run a new NVIDIA autotuning search or present migration as a new kernel
speedup. The MSA APIs are consumed explicitly by FL; custom Torch operator schemas
are registered in FL rather than added to FGV's global ATen registration.

Related upstream work: FL #560 already supplies generic partial-RoPE dispatch.
FlagGems #6480 addresses masked_fill_ on newer sources; this pinned profile selects
the already validated generic implementation explicitly. FGV's existing M3 QKV
fusion uses a different rounding contract, so the retained variant is separate.
The MSA work is related to FGV #58/#721; this patch set does not assume those
proposals are part of the pinned release.

Previously validated FlagGems MoE GEMM, grouped top-k, ordered sum, varlen
attention and dense-cache insertion remain in FlagGems at the pinned revision.
The alignment patch adds an explicit allow_tle keyword (default true); M3 passes
false to preserve its accepted four-stage implementation. New custom MSA,
norm/QKV and router/GEMV computation is carried in the FGV patches without a
reverse dependency on FL or TP state.

prepare_dependencies.py rejects source commit mismatches, dirty tracked sources
and existing output trees. Installation uses --no-deps and --no-build-isolation;
the vendor Torch/compiler stack must already be available. The FGV release's
package version string is 0.1.0; use the commit in manifest.json as its identity.
