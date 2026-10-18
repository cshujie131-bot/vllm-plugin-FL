# DeepSeek V4 Flash W8A8 on MetaX C550

This opt-in profile contains the evaluated, cleaned-up DSV4 implementation for
8 × C550 / TP8 and vLLM 0.20.2. It replaces the old private-kernel/environment
switch path. The source stays in its owning packages instead of being
rebased into a second, untested implementation.

This is an **isolated source profile**, not a change to every model using FL.
The builder copies verified dependencies into a new directory, adds the
profile's sources, and checks all 2557 source/config hashes. It does not edit
installed packages. The serving and thinking profiles reproduce the exact
source fingerprints used for validation.

## Code layout

| Under profile/src | Responsibility |
|---|---|
| vllm_fl | DS model/quantization/router adaptation, headpack communication, M/4 complete-query-row sharding, explicit operator bindings |
| flaggems_vllm | Local sparse MLA, migrated fused operators and MHC |
| flag_gems | ATen plus retained fused implementations, stable Hash routing and scoped embedding mask |
| dsv4_metax | MCTlass GEMM bindings; no vllm_metax import |
| dsv4_runtime | One-shot initialization, effective-config checks and bounded compatibility hooks |

MHC pre uses FlagGems-vllm #755 at
3a6ae4695cb63c8fe0a39774bf470e580fbbcd10. That PR is not yet merged, so the tested
source is pinned here. MHC post stays unchanged; the older split kernels are
retained only as shape/layout fallback, not as the proposed active optimization.

Decode blockdiag, staged prefill/headpack, paged logits, tree top-k and M/4
query-row sharding remain enabled. Each sharded Q row sees its complete valid
KV history; ranks gather selected indices, not partial KV-column top-k sets.
Real Q-head count is passed explicitly from FL to the local attention API.

## Configuration and compatibility

The single configuration source for each build is config/production.json
(or thinking.json for the evaluated thinking variant). Typed policies and
dispatch bindings replace the old per-optimization environment switches.
The actual ATen exclusions are checked at initialization:
topk, masked_fill and masked_fill_.

The vLLM installation is not rewritten. Necessary runtime compatibility is
explicit: version-checked serial multi-stream helpers and semantic-fingerprint
RMS launch settings. Existing FL platform/SWA compatibility remains in place;
this is not a claim of zero runtime hooks. There is no runtime source exec,
function-default rewriting, or hybrid fake operator module.

Use the validated vendor environment, not a stock NVIDIA Torch wheel:

| Component | Evaluated version |
|---|---|
| vLLM | 0.20.2 |
| Torch | 2.8.0+metax3.7.0.7 |
| Triton distribution / FlagTree | 3.0.0+metax3.7.0.7 / 0.5.1+metax3.0 |
| FlagGems | 5.0.2; source checksums are enforced |
| FlagGems-vllm base | 5f4dc53cc31b99e83967a2b19195ab1a295aeef6 |
| MCTlassEx | 0.1.1+metax3.7.0.7torch2.8 |

The dependency image is distributed separately. Its tested image ID is
sha256:d9b55ee2982c8a9727fe9510167431ab3a083747ce8f3f7a47b3fd8fe27659c7;
an image ID is not a registry pull address. The image supplies the vendor
binaries, existing tuning seed and /opt/dsv4 model checks. This repository
does not redistribute weights, closed binaries, benchmark timing databases
or compiler caches. A different image/library revision requires separate
validation; the launch helper rejects a mismatched image.

## Build and run

Obtain the pinned FlagGems-vllm checkout above and set FGV_SOURCE to its
src/flaggems_vllm directory. On an offline server, copy that checkout first.
DSV4_IMAGE is a local tag for the tested dependency image; DEPS, RUNTIME and
CACHE should be separate directories outside this checkout.

~~~sh
python examples/deepseek_v4_metax/extract_sources.py \
  --image "$DSV4_IMAGE" --output "$DEPS"

python examples/deepseek_v4_metax/build_runtime.py \
  --source vllm_fl="$DEPS/vllm_fl" \
  --source flag_gems="$DEPS/flag_gems" \
  --source dsv4_metax="$DEPS/dsv4_metax" \
  --source flaggems_vllm="$FGV_SOURCE" \
  --output "$RUNTIME"

# Prints the command. Add --execute only when the GPUs are available.
python examples/deepseek_v4_metax/run_profile.py \
  --runtime "$RUNTIME" --model "$MODEL_PATH" \
  --cache "$CACHE" --image "$DSV4_IMAGE"
~~~

The extractor creates a temporary container but never starts it. The builder
does not import torch or initialize a GPU. New output and cache directories
are required; nothing stops another workload.

Default serving: TP8, FP8 KV/block256, MBT4096, maxseq64, maxlen66560,
gpu-memory-utilization 0.85, disable-custom-all-reduce.
Keep FULL_DECODE_ONLY: prefill is outside CUDA Graph replay; decode uses FULL
graphs and torch.compile remains enabled.

For the evaluated thinking configuration, add --thinking to build_runtime.py.
It uses maxlen131072, maxseq32 and graph sizes through32. No numerical kernel
changes are made. The longer window matters: completed evaluation responses
exceeded the ordinary 66560-token serving window.

## Validation

- Standalone parity: 10 cases; two-rank headpack at16/17/128 tokens.
- TP8 graph/model smoke: 12/12 basic answers, 106/106 mixed-shape/concurrency
  requests and11/11 long-input requests; decode FULL capture/replay passed.
- Ordinary thinking/High: all198 GPQA and500 MATH responses completed with
  stop and nonempty final answers; no truncation, timeout, engine restart,
  illegal access or preemption.

| Evaluation | Result |
|---|---|
| GPQA Diamond, native strict-match | 170/198, 85.86% |
| GPQA Diamond, frozen final-answer-v1 | 171/198, 86.36% |
| MATH-500, native exact_match | 477/500, 95.40% |
| MATH-500, supplementary equivalence review | 498/500, 99.60% |

The supplementary MATH number restores21 clearly equivalent expressions
rejected by the native string-oriented scorer (e.g. exponent braces, fraction
versus decimal, root order and vector notation). It is a post-hoc review of
the23 rejected cases, **not** the native score or a new frozen automatic
metric. Original answers/results were not changed. GPQA's frozen metric is
reported separately from native strict-match.

Publication checks rebuild both profiles and compare every source/config
hash with the evaluated artifacts. This does not claim that the entire
upstream CI suite or additional APIs such as logprobs were tested.

~~~sh
python tests/unit_tests/test_deepseek_v4_profile.py -v
~~~

Measured serving throughput:64 requests, concurrency64, O1024,
one warmup + three formal rounds, last formal round:

| Input tokens | Output tok/s | Total tok/s |
|---:|---:|---:|
| 1024 | 946.18 | 1892.36 |
| 4096 | 687.18 | 3435.91 |
| 16384 | 330.72 | 5622.18 |
| 65536 | 100.31 | 6520.06 |

These performance records are retained unchanged; this source-packaging
update did not rerun a performance benchmark.
