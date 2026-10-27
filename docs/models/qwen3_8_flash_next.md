# Qwen3.8-Flash-Next on MetaX

This integrates `Qwen4ExpForConditionalGeneration` on the fixed vLLM 0.24
baseline. The original basic acceptance passed TP8/PP1 BF16 text generation,
target eager prefill/FULL decode, MTP K1 draft eager, and CPU-sharded N-gram.
Its sequential 1K/4K/16K/64K capacity prompts each generated 32 tokens.

The subsequent opt-in shared CPU lookup and scoped MTP forward graph also
passed short-answer checks. Formal P1K and P4K performance rounds completed
with these options enabled, as recorded below. Formal model accuracy remains
unmeasured. Historical baseline checks retain their original configuration.

The model/QSA/PLE integration derives from [PR #573 at 45b24bc](https://github.com/flagos-ai/vllm-plugin-FL/pull/573/commits/45b24bc7748720e0022a72e47a983398ddda36d3),
with existing source attribution retained. This branch carries the
Qwen-specific integration and MetaX contracts; it does not inherit that PR's
unrelated HY4/T-Head history. [PR #541](https://github.com/flagos-ai/vllm-plugin-FL/pull/541)
is a reference for responsibility-based code organization, not a runtime
dependency or performance baseline.

The current target is MetaX, TP8/PP1, BF16, text-only, with FP32 recurrent
state. Use `MODEL_PATH` for the local BF16 checkpoint. The capacity target is
64K context support; it does not require 64 concurrent full-length requests.
Prefill stays eager; decode uses `FULL_DECODE_ONLY` with compilation mode
`NONE`. Prefix caching is disabled and `mamba_cache_mode=none`. Current
acceptance uses **ModelRunnerFL/MRV1** with `VLLM_USE_V2_MODEL_RUNNER=0`;
MRV2 is explicitly rejected by the integration verifier.

| Dependency | Fixed integration baseline |
|---|---|
| plugin-FL base for this change | `aae3c1c3c2e993f5ff0ffea03af3553da9df9cd3` |
| vLLM v0.24.0 | `ee0da84ab9e04ac7610e28580af62c365e898389`; empty-target build |
| FlagGems-vllm | `0f3df8ecbff9b48b2f8745c7160e0df8cb5486b5` |
| Observed MetaX Torch | `2.10.0+metax3.8.1.0` |
| Observed FlagTree | `0.6.1+metax3.6` |
| Observed FlagGems / Python | `5.3.5` / Python 3.12 |

Keep the matching MACA SDK, driver, Torch ABI and native extensions together.
The wheel suffix alone does not verify driver compatibility. Triton imports
must resolve to FlagTree with its MetaX backend; ordinary CUDA wheels or a
different Triton installation are not equivalent. ATen operations prefer
FlagGems, non-ATen operators prefer FGV, and communication uses the matching
MCCL/`torch.distributed` runtime. FlagCX is not used.

For vLLM 0.24 config compatibility, MetaX applies legacy `AttentionConfig`
flags only when the fields exist. Optional MLA prefill flags are read with
safe defaults. This avoids adding removed fields that make dataclass
`replace()` fail when MTP copies its config. The narrow compatibility changes
are in [platform.py](../../vllm_fl/platform.py) and
[MetaX MLA common.py](../../vllm_fl/dispatch/backends/vendor/metax/impl/attention/mla/common.py).

| Responsibility | Repository location |
|---|---|
| Model composition, GR, BF16 loader, MTP, PLE/N-gram and local GDN adapters | [models/qwen3_8_flash_next](../../vllm_fl/models/qwen3_8_flash_next/) |
| QSA layer, indexer and main-cache backend | [models/qwen3_8_flash_next/qsa.py](../../vllm_fl/models/qwen3_8_flash_next/qsa.py) |
| QSA metadata/slot mapping and raw/compressed cache owners | [models/qwen3_8_flash_next/qsa_cache.py](../../vllm_fl/models/qwen3_8_flash_next/qsa_cache.py) |
| PLE recurrent-state metadata | [models/qwen3_8_flash_next/ple_metadata.py](../../vllm_fl/models/qwen3_8_flash_next/ple_metadata.py) |
| Graph-external N-gram preparation and runner integration | [worker/qwen3_8_flash_next.py](../../vllm_fl/worker/qwen3_8_flash_next.py) |
| Multi-cache-group MTP metadata and native proposer extension | [spec_decode/qwen3_8_flash_next.py](../../vllm_fl/spec_decode/qwen3_8_flash_next.py) |
| Architecture/config registration and version-scoped MTP compatibility | [patches/qwen3_8_flash_next](../../vllm_fl/patches/qwen3_8_flash_next/) |
| FGV thin providers and QSA selection composition | [flaggems/impl/qwen3_8_flash_next.py](../../vllm_fl/dispatch/backends/flaggems/impl/qwen3_8_flash_next.py) |
| MetaX-only contracts | [vendor/metax/impl/qwen3_8_flash_next](../../vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next/) |

Model-specific attention layers, cache owners and metadata builders live together
under `models/qwen3_8_flash_next/`. Implementing vLLM's `AttentionBackend` or
metadata interfaces does not make these QSA/PLE definitions shared framework
components. Like the model ownership in PR #573, they belong with the model;
separate files retain readable responsibilities without a parallel
`attention/qwen3_8_flash_next` package or forwarding aliases. Worker,
speculative-decoding and dispatch extension points remain in their existing
framework locations.

The N-gram embedding table is vocabulary-sharded across TP ranks and remains
in pageable CPU memory. Each rank gathers its local rows outside the model
graph, copies through bounded pinned staging, performs the TP reduction and
publishes a fixed-address device buffer. Decode replay consumes that prepared
buffer. CPU hashing, host copies and staging reuse synchronization must remain
outside capture; loader completeness checks still include the host parameter.

MTP is required for the integration target. The initial acceptance setting is
**K1**, sequential drafting, with index sharing disabled and actual
recomputation. Target logits use H hidden state while draft feedback preserves
the four residual streams (4H). Upstream sampling and accepted-token
bookkeeping remain in vLLM. The Qwen-specific proposer extends native
`EagleProposer`: main/raw/compressed QSA caches retain their separate group
metadata and slot maps. K1 requires `disable_padded_drafter_batch=True` so
chunked-prefill backup tokens follow the current chunk. Target graph unpadding
preserves the original CPU sequence-length upper bound, which vLLM 0.24's
`CommonAttentionMetadata.unpadded()` otherwise drops for non-exact buckets.
The integration does not patch this behavior for other models.
Larger draft depths, PP, EP, quantization and multimodal serving are outside
this first validation scope.

Use the [model policy](../../vllm_fl/dispatch/config/qwen3_8_flash_next_metax.yaml)
with `VLLM_FL_CONFIG`. It preserves the MetaX defaults and selects
`vendor.metax.qwen38` for five local contracts:

- `gdn_prefill`: the existing FGV MetaX V-major chunk implementation, with
  FP32 state and already-normalized Q/K. Its metadata preparation can synchronize,
  so this provider explicitly rejects graph capture.
- `qsa_select_paged_tokens`: FGV scoring, FlagGems ATen top-k with
  `sorted=False`, then Qwen block/tail expansion. The pinned FGV TLE top-k
  fails atomic pointer/value type verification with this FlagTree runtime.
  This is a policy-selected compatibility composition, not a retry after
  partial execution. The generic FGV provider remains available.
- `qwen38_small_projection`: the pinned FG linear path fails FlagTree
  `ConvertTritonGPUToLLVM` for the GR N=4 and shared-expert N=1 projections.
  Existing FG `mv` calls project each output row, then `stack` assembles the
  result. The projection stores BF16 before GR divides by `hc_count`.
  Default `F.linear` remains injectable; the shared gate changes only its
  instance's unquantized `apply`, retaining the weight loader and shared aliases.
- `qwen38_ple_gate`: BF16 `sign` hits a FlagTree `uitofp`/i16 conversion failure.
  Compute only `gate.float().sign().to(gate.dtype)` before the original
  multiplication. Sign values -1/0/1 are exact in BF16; `abs`, `clamp_min`,
  `sqrt` and `sigmoid` retain their original dtype and operation order.
- `qwen38_gdn_mixed_qkv`: the pinned MetaX FG two-dimensional
  `index_select(dim=0)` assumes tightly packed input rows. Upstream splits
  QKV from a wider QKVZ projection, so a `[42,1280]` QKV view can retain stride
  `(2048,1)`. The Qwen subclass makes this input contiguous only when native
  speculative/non-speculative mixing will gather token rows, then calls the
  unchanged upstream core. The default provider is identity; upstream already
  makes the separate `a` and `b` gates contiguous. A general stride-aware
  gather fix belongs in FlagGems; this integration only establishes the
  model's input layout boundary.

QSA expansion is model-specific plugin tensor composition. It allocates
shape-dependent intermediates and is not presented as a new generic library
kernel. FG advanced indexing can return column-major selected indices for
M>1. Its noncontiguous multiply path would materialize a Python scalar with a
host-to-device copy during capture. QSA expansion therefore creates its ratio
as an int64 zero-dimensional device tensor and uses tensor/tensor products
throughout, preserving index semantics and the configured FlagGems backend.
The model PLE path uses FGV `ple_short_conv_`.

GDN packed non-spec decode passes FP32 `b` into the existing FGV kernel to
retain FP32 sigmoid(beta), matching the plugin's corrected vLLM contract.
Speculative recurrent execution retains the vLLM 0.24 kernel compiled by
FlagTree: the pinned FGV recurrent implementation lacks the native
`NULL_BLOCK_ID=0` read/write guards. Native causal-conv also retains
accepted-token history rotation, which the pinned FGV causal-conv rejects.
These are explicit local correctness boundaries; other models and the default
backend are not replaced.

## Historical model-package ownership refactor and validation boundary

The QSA layer, its cache/backend/metadata helpers, and PLE state metadata now
live in the model package, following the model ownership in PR #573 while
retaining separate files. The three modules were relocated and their imports
updated; the old empty package was removed without compatibility forwarding
modules. All 324 retained production Python modules are text-identical to
the initial baseline after the explicit import-path substitutions. Checkpoint
prefixes, tensor arithmetic, cache layout, dispatch policy, kernel launch
parameters and communication ordering were not changed.

The refactor's CPU contracts passed 63/63 and the QSA full-forward device checks
passed 5/5, including changed-input graph replay. The PLE full-forward suite
passed 3/4: M=8 failed the exact state comparison in two runs, while its output
comparison passed. A run of the original baseline passed 4/4; importantly,
direct calls to the unchanged, pinned FGV `ple_short_conv_`, without model
code, also reproduced intermittent state mismatches (three failures in six
calls; a second process reproduced three failures in eighteen calls, with
no Qwen model modules loaded) with M8/C10240, state stride (102400,1,10240), int64 row indices
[1,3,4,5], accepted counts [2,1,2,1], dilation 3 and K1 query length 2.
These stop-after-three-failure diagnostics do not estimate a failure rate.
The M8 graph check was not reached after its eager state assertion failed.
This exposes a pre-existing operator-path issue; this structural commit does
not fix it or relax the test tolerance. Do not read the historical successful
device runs below as evidence that this issue is absent.

The serving/performance results below remain the initial
[66fa90a baseline](https://github.com/flagos-ai/vllm-plugin-FL/commit/66fa90a0638863c0bfced736b6fdc53e2a89427e).
Full-model performance and accuracy were not rerun for the directory move.

## Reproduce the CPU contracts

The current runner includes the QSA-width regressions: **76 CPU contracts**
(56 model/MTP and 20 dispatch). Historical 63-case results below retain their
original scope.

In the matching Python 3.12 / vLLM 0.24 validation environment, from the
repository root:

```bash
python tests/qwen3_8_flash_next/run_cpu_contracts.py --log-dir /tmp/qwen38-cpu
```

The runner hides accelerators, disables automatic pytest plugins, and executes
43 model/MTP contracts and 20 dispatch contracts in separate processes.
Dispatch capability/framework imports are stubbed; the model harness retains
real native metadata checks and declaration-level fixtures. This reproduces
the reported 63 CPU checks, not full runtime imports, device kernels or the
entire repository test suite. JSON summaries are written to the chosen log
directory. Device and full-model evidence are recorded separately below.

Run component tests only on an assigned idle accelerator, from the repository
root in the matching environment. Set `TEST_GPU` to that device:

```bash
export CUDA_VISIBLE_DEVICES="${TEST_GPU:?Set TEST_GPU}"
export MACA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export USE_FLAGGEMS=1 VLLM_FL_USE_FLAGGEMS_VLLM=1
export VLLM_USE_V2_MODEL_RUNNER=0
export QWEN38_RUN_DEVICE_TESTS=1
export VLLM_FL_CONFIG="$PWD/vllm_fl/dispatch/config/qwen3_8_flash_next_metax.yaml"

python -m pytest --confcutdir=tests/unit_tests/dispatch -q -s \
  tests/unit_tests/dispatch/test_qwen38_fgv_device.py \
  tests/unit_tests/dispatch/test_qwen38_native_spec_device.py \
  tests/unit_tests/dispatch/test_qwen38_projection_device.py \
  tests/unit_tests/dispatch/test_qwen38_gdn_layout_device.py
python -m pytest --confcutdir=tests/models/qwen3_8_flash_next -q -s \
  tests/models/qwen3_8_flash_next/test_ple_host_ngram_device.py \
  tests/models/qwen3_8_flash_next/test_ple_full_forward_device.py
QWEN38_QSA_EAGER_ONLY=0 python -m pytest \
  --confcutdir=tests/models/qwen3_8_flash_next -q -s \
  tests/models/qwen3_8_flash_next/test_qsa_full_forward_device.py
```

The complete PLE and QSA tests require `QWEN38_ENGINE_CONFIG` pointing to a
JSON engine configuration with an available local checkpoint. Set
`QWEN38_TEST_RESULTS_DIR` to retain optional structured results; otherwise
pytest uses its temporary directory. Prepare these fixtures before running
the opt-in tests.

Component evidence currently includes packed GDN FP32-beta regression,
QSA selection and sparse attention with changed-input graph replay, FP32
V-major GDN prefill, and the separate PLE/host N-gram device checks. Native
GDN and causal-conv K1 checks also passed eager/replay with accepted counts
1/2, changed state indices and an unchanged null state. Native GDN null outputs are intentionally
unwritten and excluded from the valid-token comparison.
The QSA selection check covers two rows and 1024 compressed columns with
512 selected blocks. A further two-row check at 17600 compressed columns and
sequence length 69632 passed eager and changed-input graph replay. This is
synthetic operator coverage, not full-model 64K capacity acceptance.
The sparse-attention check uses 20 selected positions, and GDN prefill uses
eight tokens. These small checks do not replace model-level precision or
capacity acceptance. The QSA/GDN operator tests call providers directly.

The mixed-GDN layout regression uses real FlagGems selection of two speculative
rows and forty prefill rows from `[42,1280]` with input stride `(2048,1)`.
It checks exact eager and changed-input graph results against native Torch.
The device regression passed both eager and changed-input graph replay with
bitwise agreement under FlagGems; the CPU cases verify
that only the mixed branch invokes the provider and the default keeps its view.

The full QSA layer also passed M=1/2/8/128/256 eager and changed-input graph
checks with global FlagGems enabled after the device-ratio fix. This covers
real projections, normalization, rotary embedding, indexing and attention at
TP1 with H2560/Q24/KV2/D256 and index4x128. An independent dense causal oracle
checks the short-history output, and separate oracles check raw, compressed
and main cache content. Maximum output difference was 0.0009765625, with
changed-input relative L2 at most 0.003887. These synthetic single-device
checks do not establish TP8 collectives, checkpoint accuracy or long-context
full-model acceptance.

Small-projection tests ran the complete `mv`/`stack` combination under
`flag_gems.use_gems()`, with native BF16 Torch references outside the context.
All eight M=2048/1/2/8 cases passed eager and changed-input/weight graph replay
for N=4,K=10240 and N=1,K=2560. The M=2048,N=4 maximum absolute difference was
0.0078125, with graph relative L2 error 8.76e-5; the other cases
matched exactly. The BF16-before-division boundary was checked explicitly.

Full PLE forward, including actual normalization/projections/gate/convolution
with 10240 PLE channels, passed under global FlagGems registration: M=1/2/8
decode eager and changed-input graph, plus M=2048 prefill eager. Maximum absolute output
difference was 0.00390625 eager and 0.0009765625 in graph replay. Conv state
matched the CPU update exactly when given the actual normalized input.
Isolated FP32-sign checks also covered BF16 subnormals, infinities and NaNs.
Full-model validation keeps FlagGems enabled and imports Triton from FlagTree;
this does not imply that every operator has a FlagGems implementation.

The CPU contract tests are in
[test_qwen38_fgv.py](../../tests/unit_tests/dispatch/test_qwen38_fgv.py).
Twenty isolated dispatch CPU cases passed with framework/library
capabilities stubbed; two separate CPU projection tests preserve default GR
semantics and the shared-gate weight/loader aliases. The new strided-index
regression forbids tensor/Python-scalar multiplication and host tensor
construction in expansion. Normal collection with every accelerator hidden
is currently blocked by FlagTree's driver discovery, so that isolated result is not a full runtime
pytest result. Thirty-five additional CPU contract cases passed after the
C3 metadata fix, including real vLLM 0.24 metadata/rejection preparation,
three cache slot groups, N-gram history, PLE state, lazy registration and
non-Qwen isolation. Five additional GDN CPU branch cases cover profiling,
ordinary prefill, spec-only, mixed prefill and mixed decode; only the latter
two route through the layout provider.

## Original basic-acceptance configuration

The validated engine settings are:

```json
{
  "tensor_parallel_size": 8,
  "pipeline_parallel_size": 1,
  "dtype": "bfloat16",
  "language_model_only": true,
  "max_model_len": 69632,
  "max_num_seqs": 4,
  "max_num_batched_tokens": 2048,
  "gpu_memory_utilization": 0.9,
  "enable_prefix_caching": false,
  "enable_chunked_prefill": true,
  "mamba_cache_mode": "none",
  "mamba_ssm_cache_dtype": "float32",
  "async_scheduling": false,
  "disable_custom_all_reduce": true,
  "speculative_config": {
    "method": "mtp",
    "num_speculative_tokens": 1,
    "parallel_drafting": false,
    "use_local_argmax_reduction": false,
    "disable_padded_drafter_batch": true
  },
  "compilation_config": {
    "mode": 0,
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [2, 4, 8],
    "max_cudagraph_capture_size": 8
  }
}
```

Add `model=MODEL_PATH`. Use the policy above, `USE_FLAGGEMS=1`,
`VLLM_FL_USE_FLAGGEMS_VLLM=1`, `VLLM_FL_PREFER_ENABLED=1` and
`VLLM_USE_V2_MODEL_RUNNER=0`; do not set `FLAGCX_PATH`.
The larger maximum length leaves output headroom beyond a 65536-token prompt.
The acceptance worker RPC is test-only instrumentation and is not a deployment
dependency.

## Full-model evidence and placement

The 2026-10-24 capacity run used eight MetaX C550 64 GiB devices, the fixed
versions above and the shared BF16 checkpoint. Three short chat batches
returned normal answers, including `17+25 -> 42`, English translation and
a repeated explanation of tensor parallelism. All four capacity requests
(1024/4096/16384/65536 input tokens) produced 32 output tokens without
preemption. Capacity prompts used repeated raw tokens and ended at the output
limit; they do not establish long-context answer quality.

After capture, each worker recorded 125 actual target graph replays. MTP
proposed 126 tokens and accepted 103. These counts establish active drafting
and verification, not an optimized acceptance-rate or latency claim.
The draft graph dispatcher reported `NONE`. Prefix caching remained off.

Persistent GPU parameters were exactly 33,476,362,000 bytes (31.177292 GiB)
per rank: target 30.531376 GiB plus 0.645916 GiB independent MTP parameters.
The draft shares the target embedding/head. Experts, attention projections
and MTP FC follow TP sharding; GR, routers, norms and indexer/PLE components
follow their explicit replicated layouts. The whole MTP is not replicated
on every rank. The host N-gram parameter is 12,800,061,440 bytes per rank
(11.920986 GiB, 95.367891 GiB across the host), with bounded pinned staging
and GPU preparation buffers. Each rank transfers a full 2560-element BF16
vector per requested token before the TP reduction.

Actual worker specs have 13 main QSA groups (block 400), 13 raw-index groups
(block 1600), 13 compressed-index groups (block 6400), and 37 recurrent/PLE
groups, sharing a pool of 409600-byte pages. The raw index retains full history.
Thus token-dependent cache grows by 13*(1024+256+64)=17472 bytes/token/rank
(17.0625 KiB), in addition to 28.09375 MiB logical K1 recurrent/PLE state per
request and page rounding. The older compact-ring analytical budget does not
apply to this vLLM 0.24 integration.

At the end of the capacity run, Torch allocated 55.121366 GiB/rank and reserved
55.960938–56.175781 GiB/rank. These figures include the preallocated cache pool,
graphs and runtime buffers; they are not parameter sizes or continuous peak
measurements. The engine was limited to four active requests. Full-length
concurrent capacity and formal model accuracy/performance remain unmeasured.

The existing MetaX MCCL wrapper also needed the `mcclCommAbort` binding used
by vLLM 0.24 teardown. The matching SDK exports it. A two-rank changed-input
all-reduce graph test passed, and the native bounded destroy path returned
with the abort thread completing after graph release.

The full-model harness still observes the existing executor grace-period
SIGTERM and Python resource-tracker warnings on shutdown. Workers exit and
release GPU memory; the communication binding regression alone does not
establish warning-free full-engine teardown.

The final 2026-10-25 regression used the mixed-QKV layout fix and completed
four short-chat batches of 3/3/1/4 requests: all 11 finished normally, with
arithmetic `42` and the expected English translation. Each rank recorded
114 target graph replays, 7 eager target forwards and 27 actual 8-to-6 token
unpadding events for three requests. MTP proposed 264 tokens and accepted
192, with zero preemptions. Draft execution remained eager. The harness
exited with code 0, then the workers released their GPU allocations.

The 64K capacity result belongs to the earlier capacity run. Subsequent fixes
preserve its C1 numerical/cache-layout paths: the GDN layout normalization is
strictly guarded to mixed speculative/non-speculative batches, and the final
run validates the affected multi-request cases. No formal precision score or
performance improvement is claimed.

## Optional shared lookup and MTP forward graphs (2026-10-25)

`additional_config.qwen38_ngram_mode="shared_buffer"` selects one CPU producer
per single-node TP group. TP rank zero holds the full pageable N-gram table;
other ranks retain zero-sized table parameters for loader completeness checks.
One gathered result is published through a bounded shared mmap. Consumers copy
it into private pinned staging before the producer may reuse the shared bytes,
then issue H2D into their existing fixed-address GPU buffers. Only the lookup's
GPU all-reduce is removed. `"sharded"` remains the default supported alternative.
Cross-node/shared-memory-inaccessible TP groups are explicitly rejected for the
shared mode. The large table itself is never pinned or copied into the shared
result mapping.

`additional_config.qwen38_mtp_graph=true` enables a Qwen-specific fixed-buffer
wrapper around the QSA-only MTP forward. It leaves native proposal alignment,
rejection, logits and sampling unchanged. Each graph bucket owns stable input,
block-table, sequence-length, position and slot buffers. Invalid padding slots
remain -1. QSA cache updates overwrite the same slots, allowing idempotent
capture warmup; this implementation must not be reused for recurrent/PLE draft
layers. Large prefill forwards remain eager. The native drafter dispatcher
continues to report NONE; actual scoped graph captures/replays must be audited
through this wrapper or the GPU trace, rather than inferred from that field.

On eight C550 devices, the same eight short questions completed normally with
both modes. Eager accepted 83/114 draft tokens (72.81%); graph accepted 86/117
(73.50%), with seven captured buckets and 79 graph replays per rank. This small
sample did not show a material acceptance drop; it does not establish a formal
precision score or an acceptance-rate improvement. The code remains opt-in.

The shared-table check observed exactly one 95.367891 GiB CPU table, 76 producer
lookups on rank zero and 76 result consumptions on every rank. A 0.96 GPU memory
budget failed the startup free-memory check (59.5 GiB free versus 61.05 GiB
requested); 0.93 passed loading, target capture through 128 tokens and the short
checks. This is a tested configuration, not a universal maximum safe budget.

The Qwen runner now also checks prompt progress before graph dispatch. A
2-token prefill tail may have the same shape as K1 decode, but still executes
eagerly to preserve recurrent initialization. Pure decode, dummy capture and
other models retain their prior paths. Benchmark graph coverage is checked
against scheduler phase metadata and actual replay events, with mixed batches
reported separately.

The pinned FGV PLE state writer retains the supplied index dtype when forming
`row * state_stride`. A unified BF16 cache page has a 204800-element row stride,
so int32 address arithmetic overflows above row 10485. PLE now supplies int64
indices for both speculative and non-speculative calls, preserving the native
metadata buffers and the operator implementation. Device tests cover actual
10240-channel state, rows 10485/10486/11393, prefill/speculative updates and
changed-input graph replay. The original int32 path reproduced illegal access;
all corrected cases passed. This is an address-width correction, not a reduced
cache-size workaround or a kernel-library change.


## Historical shared lookup / draft graph performance results

The current performance configuration changes the original basic-acceptance
settings to `max_num_seqs=64`, `gpu_memory_utilization=0.93`, target capture
sizes `[2,4,8,16,32,64,128]`, maximum capture size 128, and:

```json
{
  "additional_config": {
    "qwen38_ngram_mode": "shared_buffer",
    "qwen38_mtp_graph": true
  }
}
```

TP8/PP1, BF16, MTP K1, max length 69632, MBT 2048, eager prefill,
prefix off, synchronous scheduling and the fixed dependencies remain unchanged.
Random completions use O1024/C64, temperature 0, seed 0, infinite arrival rate
and ignore-EOS. Every successful request had the requested input/output length.
One complete warmup precedes one formal round per scenario.

| Formal input | Requests N | Successful / failed | Output token/s | Total token/s | Mean TTFT ms | P99 TTFT ms | Mean TPOT ms | P99 ITL ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 256 | 256 / 0 | 53.8586 | 107.7171 | 80117.66 | 505370.56 | 1069.33 | 16559.94 |
| 4096 | 128 | 128 / 0 | 26.3882 | 131.9410 | 524588.08 | 1887814.30 | 1859.70 | 17520.44 |

N changed by user request after P1K had completed. The incomplete old P4K
N256 warmup was discarded; P4K N128 restarted with a full warmup and formal
round. Different request counts change the relative tail contribution, so
these are workload-specific results, not a controlled cross-length speedup.

P1K's formal round encountered one previously unused MTP bucket and captured
it on first use; its measured duration includes that cost. No round was
replaced to select a faster result.
P1K accepted 108573/153436 draft tokens (70.76%); P4K accepted
55501/75495 (73.52%). Every rank recorded 2555/2555 and 1214/1214 pure-decode
target graph replays respectively, with no scheduler/worker phase mismatch.
P4K also recorded exactly 1214 scoped draft graph replays per rank, with no new
capture during its formal round. Mixed batches remained eager and are reported
separately: 25588 and 27446 mixed decode query tokens per rank respectively.
The 100% graph statement applies to pure-decode batches, not mixed batches.

A separate P1K/O1K/C64/N128 profile workload completed 128/128 requests,
with 131072 input and output tokens each. Every rank recorded 1392/1392 pure
decode target replays and 1392 MTP forward replays, with no new draft capture
during collection. Eight lazy draft buckets were populated by tiny requests
before collection after the profile-only service restart.

All eight ranks have 93 complete raw JSON segments each (744 files), covering
1482 worker steps per rank. Runtime launch correlation attributed every GPU
kernel event to its recorded phase; unknown kernel time was zero. Globally,
the main MCCL all-reduce represented 30.16% of accumulated GPU kernel time,
the two top-k stages 36.54%, and QSA sparse attention 11.98%. In pure decode,
the top-k stages represented 49.86%. These cumulative device-time fractions
overlap across ranks and streams and cannot establish wall-clock fractions,
network bandwidth saturation or expected optimization speedups. No tuning
was performed from these observations.

The original N256 whole-trace export exceeded the test container's 768 GiB
host-memory limit. A subsequent 32-step gzip exporter exceeded the default
300-second worker RPC timeout, without a new OOM. Successful collection used
16 worker steps per raw JSON file and a profile-only 1800-second timeout.
The production model and formal benchmark results were unchanged. Profiling
and export overhead is excluded from the formal performance table; the test
container's CPU quota was 32 cores. The profiler and phase-audit extensions
are private test instrumentation, not production model dependencies.


## MetaX parameters and QSA score-width buckets (2026-10-27)

The opt-in [MetaX tuning profile](../../examples/qwen3_8_flash_next/metax_tuning/)
packages the validated linear/MV, MoE, QSA attention, PLE, mean and GDN launch
choices. Small fingerprint-checked patches expose existing launch parameters;
no computation kernel is copied into this repository. Dense/MV use FG's
ConfigCache API and MoE uses VLLM_TUNED_CONFIG_FOLDER. Unlisted shapes retain
the existing fallback/autotuning behavior.

The fixed FGV base requires the PLE file from
[FGV #959](https://github.com/flagos-ai/FlagGems-vllm/pull/959) at
22194f0b2ee0cebb51f945a20a157b82c4e19c65 and the MetaX inplace-MoE export from
[FGV #1000](https://github.com/flagos-ai/FlagGems-vllm/pull/1000) at
e04f51c1c7176d7a356838fdfa8b3dc4fbaf6f8a before applying the parameter patches.
Preparation validates both prerequisites and every modified library file.
The tested composition is the frozen 0f3df8e runtime plus these deltas, not
the complete latest FGV main. FG and FGV keep separate default caches;
set FLAGGEMS_CACHE_DIR only for the FG seed command, not the serving process.

Enable qwen38_qsa_length_buckets in additional_config alongside the shared
N-gram and MTP graph options. The existing CPU sequence-length upper bound
selects 512/2048/8192/cache-capacity score columns. At maxlen69632, capacity
is 17600 (eleven compressed pages of 1600 groups). Scores are allocated and
computed at the selected width, with per-row causal masks and bucket padding.
Cache allocation is unchanged and no device-to-host length read is added.

Target and MTP graph keys include the width; seven target token shapes become
28 width/token combinations. Score tile32, fixed K512 and the original
two-stage TopK are preserved. No new selection algorithm, communication change
or FG/FGV exclusion is introduced. Model semantics/cache metadata remain in
models; runner, draft graphs, shared graph-key support and vendor dispatch
retain their existing directories. Parameters depend on this model's geometry,
so the profile belongs in this model PR rather than a separate operator PR.

Validation includes 76 CPU contracts; 72 strict index comparisons with three
changed-input graph replays each; real target cross-bucket graphs; and MTP
metadata guards. Earlier width validation also covered a 64K input, while
this measurement round uses P1K/P4K only. Production source matches the tested
candidate; publication only aligns setup/cache instructions and dependencies
with the measured environment.

### Serving results with the MoE fix and this profile

BF16 TP8/PP1, 8 C550, MTP K1 graph, shared CPU N-gram, FP32 recurrent state,
maxlen69632, MBT2048, max sequences64, utilization0.93, prefix off and synchronous
scheduling. Random completions: P1024/P4096, O1024, C64, N128, seed0, temperature0,
ignore EOS and infinite arrival rate; one full warmup plus one formal round.
Container limits are 32 CPU cores and 768 GiB RAM; the paired image is unchanged.

| Input | MoE-fixed baseline tok/s | Tuned output tok/s | Increase | Mean TTFT ms | Mean TPOT ms | MTP acceptance |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 94.5826 | 148.3915 | 56.89% | 34181.01 | 350.85 | 73.87% |
| 4096 | 55.5916 | 90.3198 | 62.47% | 104095.77 | 546.18 | 73.99% |

Each formal round completed 128/128 requests with exact lengths, no failures
or preemptions. All eight ranks replayed target and MTP pure-decode graphs
on every step (1372/1372 and 1312/1312), with zero eager fallback, new captures
or phase mismatch. Mixed batches remain eager. P1K uses width512 throughout;
P4K pure decode uses width2048. Warmups completed once each, with output
82.3156/57.1214 tok/s; warmup MTP captures were 2/8 and were not removed.

P1K moved from node125 to the same-model node102 after a device link error
before any benchmark requests. P4K remains node126. This is one measured round
per variant, not an interleaved same-node A/B or a model-accuracy claim.

The full P1K profile contains 744 traces, 1476 worker steps per rank and
1383 pure-decode target/MTP replays; unknown GPU kernel time is zero.
Normalized cumulative decode GPU time fell from 290.618 to 118.031 ms per
rank/step; QSA TopK fell 143.416 to 33.220 ms and sparse attention 45.934 to
9.597 ms. These overlapping device durations are not request wall time or
isolated matched-shape kernel speedups.

Remaining issue: mixed linear/MV/clear call counts rose (linear 364000 to
630219, MV 321048 to 1202412, clear 785500 to 2494746). Dynamic shapes can
still enter existing autotuning. This release does not claim to have removed
that overhead or to have attributed every additional clear. Mixed communication
residency also remains significant; no communication optimization is included.

All test GPU workers were released. Existing resource-tracker shutdown
warnings remain documented; the full accuracy and repository-wide suites
were not rerun for this parameter/profile change.
