# MiniMax M3 BF16 on MetaX C550

This example enables the MiniMax M3 optimizations in the installed FL package.
Model/TP orchestration lives in FL; operator changes are readable patches against
pinned public FlagGems and FlagGems-vllm sources. Other model profiles keep the
normal FL entry points and dispatch policy.

## Environment

The target is two nodes, each with 8 × MetaX C550 (64 GiB), using TP16 and the
original BF16 checkpoint. The serving configuration uses 32 CPU cores per node.

| Component | Version |
|---|---|
| Python | 3.12.13 |
| MACA SDK/runtime | 3.8.1.3 |
| Host KMD | 3.8.1 |
| MetaX Torch | 2.10.0+metax3.8.1.0 |
| Triton API / FlagTree | 3.6.0 / 0.6.1+metax3.6 |
| vLLM | 0.24.0, public commit ee0da84ab9e04ac7610e28580af62c365e898389 |
| FL branch | main, with this PR |
| FlagGems | v5.3.5, a7620cc191a0b42e040194622c5758b22a7a25dc + patches |
| FlagGems-vllm | release v0.2.0, b5c2c2249319277ccda4d41aa0f65164bd9b2a19 + patches |
| Vendor FlashAttention | 2.6.3+metax3.8.1.0torch2.10 |

The FlagGems-vllm release reports 0.1.0 in package metadata; the public commit is
the source identity. Do not infer its revision from that metadata string.

Use the matching vendor Torch, MACA/MCCL, FlagTree and FlashAttention binaries.
The code here does not build or redistribute those components and is not a
source-only recipe for a complete vendor image. Keep the matching host verbs
stack and Broadcom RDMA provider available. Vendor GEMMs and dense attention
remain where required; the vllm_metax framework is not loaded. FlagCX is not used.

## Prepare dependencies and install

Run in an isolated vendor environment on both nodes. Keep its Torch, compiler,
NumPy and SDK versions; the commands below do not resolve or upgrade dependencies.

~~~sh
git clone https://github.com/vllm-project/vllm.git /work/vllm
git -C /work/vllm checkout ee0da84ab9e04ac7610e28580af62c365e898389
VLLM_TARGET_DEVICE=empty python -m pip install --no-deps --no-build-isolation /work/vllm

git clone https://github.com/flagos-ai/FlagGems.git /work/FlagGems
git -C /work/FlagGems checkout a7620cc191a0b42e040194622c5758b22a7a25dc
git clone https://github.com/flagos-ai/FlagGems-vllm.git /work/FlagGems-vllm
git -C /work/FlagGems-vllm checkout b5c2c2249319277ccda4d41aa0f65164bd9b2a19

python examples/minimax_m3_metax/prepare_dependencies.py \
  --flag-gems-source /work/FlagGems \
  --fgv-source /work/FlagGems-vllm \
  --vllm-source /work/vllm \
  --output /work/m3-dependencies --install

python -m pip install --no-deps --no-build-isolation -e .
~~~

Preparation checks the public commits, creates separate copies, and reads two
MSA kernel files from the pinned vLLM source into the FGV copy. Their exact
hashes are checked before applying the incremental patches in manifest order.
No installed vLLM source is modified, and FGV has no runtime dependency on vLLM. It refuses a dirty source or an existing output
directory. It does not modify the supplied checkouts. An empty-target wheel labeled 0.24.0+flagos is accepted only when its guarded
APIs and M3 model files match the public v0.24.0 sources. Old vendor images with
locally modified M3 files must install the clean source above.

Use the BF16 checkpoint from
[ModelScope MiniMax/MiniMax-M3](https://www.modelscope.cn/models/MiniMax/MiniMax-M3).
The two nodes must see the same complete checkpoint and tokenizer snapshot:

~~~sh
modelscope download --model MiniMax/MiniMax-M3 --local_dir /models/MiniMax-M3
~~~

Mount weights read-only. Do not substitute a quantized checkpoint or reduce TP to
fit one node. With containers, expose /dev/dri, /dev/mxcd and /dev/infiniband,
use host networking/IPC and allow locked memory. Set the vendor library search
paths for the matching MCCL and verbs provider before starting Python.

## Launch

config.json is the single model/optimization configuration, passed through
VLLM_FL_CONFIG. It includes the accepted Linear tables, MoE launch policy,
attention shape rules, dispatch policy and standard runtime/communication
variables. The launcher clears obsolete optimization switches before starting a
fresh Python process.

The migrated vLLM kernels are compiled with FlagTree; source reuse does not
change their compiler or switch them to native binary operators.
Supply the installed FlagTree root containing triton/; it is used as the compiler
import prefix. The example prints its command by default. Add --execute only when
the GPUs and ports are available.

~~~sh
# Node 0
python examples/minimax_m3_metax/serve.py \
  --model /models/MiniMax-M3 --flagtree-root "$FLAGTREE_ROOT" \
  --cache-dir /work/m3-cache --node-rank 0 --master-addr "$MASTER_ADDR" \
  --port 18050 --master-port 29750 --execute

# Node 1
python examples/minimax_m3_metax/serve.py \
  --model /models/MiniMax-M3 --flagtree-root "$FLAGTREE_ROOT" \
  --cache-dir /work/m3-cache --node-rank 1 --master-addr "$MASTER_ADDR" \
  --port 18050 --master-port 29750 --execute
~~~

Each node gets separate Triton, Inductor, vLLM, FG and FGV caches. An incompatible
cache identity or an unknown nonempty directory is rejected; nothing is deleted
and no existing service is stopped. No compiler caches or timing databases are
distributed. Fresh caches require warmup/compilation.

Defaults: TP16, BF16 main/index KV, block128, maxlen131072, maxseq64, MBT8192,
gpu-memory-utilization0.96, standard vLLM Scheduler, watermark0, no prefix cache
and no async scheduling. The accepted FULL_AND_PIECEWISE graph configuration is
retained, with one capture warmup; the model's Inductor compilation mode is NONE.
There is no custom KV admission scheduler in this profile.

The communication example uses four MCCL channels, P2P cap4 and a 1MiB buffer.
tuning-ar512k.cfg retains the tested message-size rule. Set the socket interfaces,
HCA names and GID index in config.json for the actual RDMA fabric. An ACTIVE port
alone does not establish that traffic is using it: check MCCL NET/IB/GDRDMA logs
and port counters during the smoke.

## Review map

[Changes against pinned upstream sources](UPSTREAM_DIFF.md) identifies reused
code, source imports and the actual M3 deltas. It does not use an older PR as
the comparison baseline.

| Area | Main files or patches | What to review |
|---|---|---|
| Configuration/startup | vllm_fl/minimax_m3_config.py; patches/minimax_m3_metax/ | Typed immutable policy, effective blacklists, scoped initialization and cache identity |
| Model | vllm_fl/models/minimax_m3/ | Only changed text/vision methods and explicit M3 module bindings; unchanged model code stays in vLLM |
| MSA orchestration | ops/minimax_m3/attention.py; base_index.py; base_sparse.py; *_policy.py; attention_launch.py | TP/owner checks, local query/head packing, length/shape selection and output slicing |
| MSA kernels | dependencies/flaggems_vllm/01-msa-kernels.patch | Exact block top-k, packed prefill, subtile/split decode and long-context computation |
| Linear | dependencies/flag_gems/01-linear-policy.patch | Retained 35 configurations, small-M dot/GEMV and M8192 program ordering; explicit policy supplied by FL |
| Gate and MoE | ops/fused_moe/fused_moe_utils.py; ops/minimax_m3/experts.py; expert_layer.py; moe.py; router.py; routing.py; dependencies/flaggems_vllm/03-moe-router.patch | Small shared extension points; M3 overrides only GEMM context, activation and bindings; retained BF16-load/FP32 gate and routed GEMV |
| Norm/activation/QKV | dependencies/flaggems_vllm/02-norm-activation-qkv.patch | Gemma residual semantics, clamped SwiGLU and BF16 rounding between QK normalization and RoPE/cache insertion |
| Vision and library runtime | ops/minimax_m3/vision.py; vision_chunk.py; dependencies/flaggems_vllm/04-launch-runtime.patch | Shared partial-RoPE dispatch from #560, patch chunking, retained FlagGems varlen call, isolated FGV cache |
| In-place mask | dependencies/flag_gems/02-generic-masked-fill.patch | Select the existing generic masked_fill_ function; non-inplace masked_fill stays excluded |
| Vendor/communication | patches/minimax_m3_metax/compat.py; sdk/; tuning-ar512k.cfg | MCCL ABI, in-place AllReduce and required SDK compatibility; no vendor framework import |

The M3 computation modules take tensors, workspace and launch parameters; they
do not discover TP groups or import FL. Existing rotary and ordinary SiLU
implementations are reused. The M3 QKV variant preserves the accepted intermediate
BF16 rounding, so it is not replaced with a newer fusion that changes that
contract. The alignment patch exposes an explicit allow_tle argument; M3 passes false,
preserving the accepted four-stage computation. Existing verified FlagGems
MoE GEMM, sigmoid top-k, ordered sum, vision attention and dense KV insertion
remain on that pinned implementation. They are not migrated to different FGV
fusions as part of packaging.

Vision RoPE now follows the native M3 attention class through the shared
apply_rotary_emb dispatch from #560. Its FGV implementation compiles with
FlagTree; the duplicate M3 RoPE op and model override are removed. The separate
text QKV/norm/RoPE/cache-insertion fusion is retained.

The entry has real runtime hooks for the matched Torch/vLLM/MCCL ABI. They are
centralized, guarded and idempotent, and only the explicit M3 profile installs
them. Model hooks replace only named bindings inside vLLM's M3 modules, after checking
their public source revision. General router and MoE factories stay unchanged.
Unchanged SDK wrappers are imported from FL instead of copied. No installed
vLLM source is rewritten and no fake module facade is installed by this entry.

## Validation and measurements

Model evaluation uses ordinary thinking and final answers:

| Task | Accuracy |
|---|---:|
| GPQA Diamond | 181/198, 91.41% |
| ERQA | 214/400, 53.50% |

GPQA includes the five budget-limited empty final answers in its denominator;
reasoning text is not used to recover intermediate choices.

Retained formal serving measurements on two 8-C550 nodes, TP16, BF16 KV, MBT8192:
256 requests, concurrency64, O1024; one warmup plus one formal round.

| Input tokens | Output tok/s | Total tok/s |
|---:|---:|---:|
| 1024 | 683.47 | 1366.94 |
| 4096 | 349.67 | 1748.37 |
| 16384 | 125.90 | 2140.37 |
| 65536 | 39.28 | 2553.05 |

The formal run IDs are short-formal-best-20260921/run02 and run04 for 1K/4K,
and long-shape-specialization-20260920/optimized-formal/run02 and run04 for
16K/64K. These are retained measurements with Triton3.6.0, not a new benchmark of
the publication layout. The current deployment keeps FlagTree enabled.

The accepted deployment passed 104 operator checks, 27 MSA cases, 26 MoE-chain
cases, 198 service regressions and four chat/image checks. The refactored and
previous outputs matched under the established per-operator criteria.

The shared vision RoPE entry passed 25 standalone BF16 cases on C550 with
FlagTree, including three graph/eager checks. The accepted FP32-reference
tolerance is rtol=0.01 and atol=0.01; input and unrotated tails stay exact.
Old/new relative L2 was at most 9.84e-6; this is numerical agreement, not a
claim of bitwise equality or a new full-model accuracy run.

Packaging checks: CPU configuration/launcher tests, syntax, critical lint,
patch application and numerical-body/configuration comparisons. The installed
entry with clean public vLLM sources still needs a TP16 smoke; this PR stays
Draft until that passes. The model evaluations and serving results above are
existing measurements, not new runs of this packaging.

~~~sh
python tests/unit_tests/test_minimax_m3_packaging.py -v
~~~
