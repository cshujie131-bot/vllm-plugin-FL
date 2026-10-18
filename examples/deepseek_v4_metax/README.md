# DeepSeek V4 Flash W8A8 on MetaX C550

This example enables the DS-specific optimizations in the **installed FL package**.
It does not assemble a second FL tree, overwrite vLLM, or require a private image ID.
Other models keep the regular FlagGems provider.

## Environment

The supported target is one node with 8 × C550 (64 GiB), TP8, and the
DeepSeek V4 Flash W8A8/INT8 checkpoint.

| Component | Tested version |
|---|---|
| vLLM | 0.20.2 |
| Torch | 2.8.0+metax3.7.0.7 |
| Triton / FlagTree | 3.0.0+metax3.7.0.7 / 0.5.1+metax3.0 |
| FlagGems | 5.0.2, commit 3123859968915e361e8452dd796dc6b27c956324 |
| FlagGems-vllm | commit 5f4dc53cc31b99e83967a2b19195ab1a295aeef6 |
| MCTlassEx | 0.1.1+metax3.7.0.7torch2.8 |

Use the matching MetaX Torch/MACA environment; do not replace it with a standard
CUDA Torch wheel. MCTlassEx supplies GEMMs, not the vllm_metax framework.
The runtime rejects importing vllm_metax. FlagTree remains enabled; FlagCX is not used.

## Prepare and launch

Run inside an isolated environment with the vendor runtime above.
Obtain the two public source checkouts at their pinned commits (or transfer them
to an offline machine). Preparation verifies their source trees, copies them to
a new directory, and applies the readable patches under dependencies/.
It never edits the supplied checkouts.

~~~sh
git clone https://github.com/flagos-ai/FlagGems.git /work/FlagGems
git -C /work/FlagGems checkout 3123859968915e361e8452dd796dc6b27c956324
git clone https://github.com/flagos-ai/FlagGems-vllm.git /work/FlagGems-vllm
git -C /work/FlagGems-vllm checkout 5f4dc53cc31b99e83967a2b19195ab1a295aeef6

python examples/deepseek_v4_metax/prepare_dependencies.py \
  --flag-gems-source /work/FlagGems \
  --fgv-source /work/FlagGems-vllm \
  --output /work/dsv4-dependencies --install

python -m pip install --no-deps --no-build-isolation -e .

# Prints the command only. Add --execute when the eight GPUs are available.
python examples/deepseek_v4_metax/serve.py \
  --model /models/DeepSeek-V4-Flash-W8A8-INT8 \
  --cache-dir /work/dsv4-cache --port 18001
~~~

No service is stopped by these helpers. Do not share a writable cache with another
experiment. A cache identity mismatch requires a new directory; caches are never
automatically deleted. Fresh caches need compilation/autotuning warmup; timing
databases from the original machine are deliberately not distributed.

config.json is the single configuration source, carried through the existing
VLLM_FL_CONFIG mechanism. serve.py sets only infrastructure/cache variables;
there is no per-optimization environment-switch collection.

Serving defaults: FP8 KV/block256, MBT4096, maxseq64, maxlen66560,
gpu-memory-utilization 0.85, disable-custom-all-reduce. FULL_DECODE_ONLY keeps
prefill outside graph replay and retains decode graphs and torch.compile.
Use --thinking for the evaluated 131072-token window and maxseq32.
The bootstrap directory is opt-in through this launcher; it is not installed as
a global sitecustomize module.

## Review map

Read the FL integration first, then only the relevant operator patch.

| Area | Main code | What to review |
|---|---|---|
| Configuration and startup | vllm_fl/dsv4_config.py; vllm_fl/patches/deepseek_v4_metax/ | Typed configuration, explicit binding, scoped compatibility |
| Model and quantization | vllm_fl/models/deepseek_v4.py; vllm_fl/quantization/w8a8/deepseek_v4.py | Accepted INT8 path, embedding/Hash-router contracts |
| Attention orchestration | vllm_fl/ops/deepseek_v4_attention.py; dsv4_headpack.py | Real-head count, headpack communication, output slicing |
| Indexer | sparse_attn_indexer.py; indexer_query_sharding.py; prefill_indexer_rowshard.py | Complete-query-row M/4, local top-k, short-row offsets |
| DS provider | vllm_fl/dispatch/backends/flaggems/deepseek_v4_metax.py; impl/deepseek_v4_metax/ | Explicit FG/FGV calls; generic provider stays unchanged |
| MCTlass bridge | vllm_fl/dispatch/backends/vendor/metax/impl/dsv4/ | GEMM binding, INT8 layouts, cache ownership; no vendor framework import |
| ATen/retained FG helpers | dependencies/flag_gems.patch | Embedding mask, Hash routing, HC head and bounded tile policy |
| Sparse MLA | dependencies/flaggems_vllm/01-sparse-mla.patch | Decode blockdiag, staged prefill, gather strides and launch policy |
| Upstream MHC source | dependencies/flaggems_vllm/02-mhc-pr755.patch | Pinned #755 source, not a new optimization claimed by this PR |
| MHC adapter | dependencies/flaggems_vllm/03-mhc-integration.patch | Graph-compatible pre wrapper; original post and shape fallback |
| Other FGV integration | dependencies/flaggems_vllm/04-fused-adapters.patch | Quant/align/activation interfaces, cache isolation, compiler-hook guard |

The dependency patches apply to the listed public commits, not arbitrary latest
versions. MHC pre comes from
[FGV #755](https://github.com/flagos-ai/FlagGems-vllm/pull/755)
at 3a6ae4695cb63c8fe0a39774bf470e580fbbcd10, which was not merged when packaged.
MHC post is not replaced by that PR. Existing Apache/MetaX attribution is retained
in the sources. No weights, experiment logs, binaries or compiler caches are included.

This branch targets the evaluated FP8 indexer-cache / MCTlass INT8 path. The
intermediate PR #372 graph-envelope, combined-compressor and INT8 indexer-cache
variants are not the measured implementation and are not silently mixed into it.

## Validation

The accepted implementation passed standalone parity, two-rank headpack checks,
TP8 decode-graph smoke, 106 mixed-shape/concurrency requests and 11 long-input
requests. Thinking/High evaluation used a 128K window and concurrency 32; all
698 responses completed without truncation, timeout or engine errors.

| Task | Reviewed accuracy |
|---|---:|
| GPQA Diamond | 171/198, 86.36% |
| MATH-500 | 498/500, 99.60% |

GPQA uses final-answer extraction; MATH-500 includes mathematical-equivalence
review of equivalent expressions rejected by the automatic scorer.

Measured throughput: 64 requests, concurrency64, O1024, one warmup plus three
formal rounds, last formal round:

| Input tokens | Output tok/s | Total tok/s |
|---:|---:|---:|
| 1024 | 946.18 | 1892.36 |
| 4096 | 687.18 | 3435.91 |
| 16384 | 330.72 | 5622.18 |
| 65536 | 100.31 | 6520.06 |

These are the retained measured results, not a new performance run.
This publication-layout update is CPU-checked; the new installed-package launch
path still needs a fresh GPU smoke before the draft is marked ready.

~~~sh
python tests/unit_tests/test_deepseek_v4_packaging.py -v
~~~
