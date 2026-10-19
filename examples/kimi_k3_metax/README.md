# Kimi K3 on MetaX C550

TP8 × PP4, DP1, EP/SP off, on four 8-GPU C550 nodes. Original MXFP4/E8M0
weights remain packed. Main GEMM/attention inputs are BF16; sensitive arithmetic
and recurrent state retain their accepted precision. This is not an INT8 conversion.

## Review map

| Responsibility | Implementation |
| --- | --- |
| Model factories, latent TP shards, head packing, vision layout | `vllm_fl/k3/{model_adapter,latent_moe,mla_adapter,kda_adapter,vision_adapter}.py` |
| One execution policy and explicit worker registration | `vllm_fl/k3/{configuration,context,registration,op_registrar,routing}.py` |
| TP8 expert layout, packed-weight GEMMs, G1/SiTU specialization, bounded G2 scratch | `vllm_fl/ops/kimi_k3_metax/{tp8_experts,tp8_expert_g1,expert_gemm,decode_experts,marlin_mxfp4}.py` |
| TP-local MLA, staged QK/PV, short/long shape selection, interleaved KV fallback | `vllm_fl/k3/{mla,mla_dispatch}.py`, `vllm_fl/ops/kimi_k3_metax/mla*.py` |
| Accepted KDA input/state compatibility | `vllm_fl/k3/kda_adapter.py`, `vllm_fl/ops/kimi_k3_metax/{kda,kda_preprocess,normalization,kda_kernels/}` |
| Runner-owned workspace and PP input ownership | `vllm_fl/k3/{model_runner,tp8_prefill_workspace}.py`, `vllm_fl/worker/pipeline_input.py` |
| Host-packed sampled-token feedback | `vllm_fl/k3/pp_feedback.py`; vLLM retains the native queue and two default-preserving extension points |
| Pure routing, SiTU and route alignment | Pinned FlagGems-vllm source and PRs below |
| ATen correctness and addmm launch compatibility | Pinned FlagGems source and PRs below |
| Existing MACA/MCCL bindings and host-only extensions | `vllm_fl/dispatch/backends/vendor/metax/`, `csrc/metax/`, `setup.py` |

This update relocates 30 previously validated source files. The framework
kernels were previously carried in an example dependency patch. TP layout,
head packing, buffer ownership and launch policy now live in FL; pure qualified
math is called from the libraries. Native model loading, processors, metadata
and model forward skeletons are reused.

The KDA compatibility path retains the previously evaluated FP32 chunk inputs
and local FP64 normalization/gate accumulation. It is not advertised as a new
performance contribution. Its wrappers retain their local kernel bindings;
replacing them with a same-named newer KDA implementation would change the
numerical contract and is outside this cleanup.

## Public source dependencies

`requirements.txt` and `dependency-lock.json` pin complete source commits. No
source patch, source snapshot or patch installer is required at installation.

| Package | Pinned source | Review |
| --- | --- | --- |
| vLLM | [cc1c8274](https://github.com/cshujie131-bot/vllm/commit/cc1c8274e47fd9148cc49a8d9e0c02fb037b09f8) | Default-preserving component/runner interfaces; explicit non-CUDA CUSTOM MLA validation; nine Python files, no scheduler algorithm change |
| FlagGems | [6efd6dab](https://github.com/cshujie131-bot/FlagGems/commit/6efd6dab4a4a8a78167262d0841ff0d728a2d1d3) | [Argmax #7300](https://github.com/flagos-ai/FlagGems/pull/7300); [addmm compatibility Draft #7302](https://github.com/flagos-ai/FlagGems/pull/7302) |
| FlagGems-vllm | [1d4b69b3](https://github.com/cshujie131-bot/FlagGems-vllm/commit/1d4b69b355a8a42deb0edd485ef9d9a94b2477b8) | [TopK #950](https://github.com/flagos-ai/FlagGems-vllm/pull/950), [SiTU #951](https://github.com/flagos-ai/FlagGems-vllm/pull/951), [alignment #952](https://github.com/flagos-ai/FlagGems-vllm/pull/952), [MLA interface #953](https://github.com/flagos-ai/FlagGems-vllm/pull/953) |

The reviews are unmerged at publication time. The integration commits backport
onto the validated bases, rather than upgrading unrelated operators. Each fork
contains `K3_INTEGRATION.md`. Argmax/addmm are correctness or compatibility work;
the MLA callable is an interface change. Only the three qualified operator
optimizations claim standalone gains against their corresponding native boundary.

TopK explicitly selects FP32 sigmoid → input-dtype materialization before bias
and selection; default FGV tanh-based sigmoid is not numerically interchangeable.
Alignment keeps FL's compact-buffer policy and imports the library's compact
kernel. SiTU uses the library math under the original FL shape guards. Small,
unqualified and non-normal-scale fallbacks remain available where previously used.

## Environment

| Component | Required version |
| --- | --- |
| Python | 3.12.13 |
| vLLM | 0.28.0, empty backend plus the pinned generic interfaces |
| FL target | `0.4.0-dev` |
| MACA SDK/runtime/MCCL | 3.8.1.3 |
| MetaX Torch | 2.10.0+metax3.8.1.0 |
| FlagTree / Triton API | 0.7.0+metax3.6 / 3.6.0 |
| Transformers / host driver | 5.15.1 / 3.8.1 |

Matching vendor Torch, MACA, MCCL, RDMA userspace and FlagTree compiler binaries
are required. Do not substitute a stock CUDA wheel. See the
[MetaX FlagTree instructions](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-metax).
No proprietary binary or weights are distributed here. The profile uses no
`vllm_metax`, `mcoplib` or FlagCX; MACA/MCCL are hardware runtime dependencies.

## Install and launch

Use a fresh environment or image with the matching vendor runtime. Force
reinstallation prevents an existing same-version package from hiding a source
change. Build-only version override identifies the empty vLLM package; Git
provenance independently checks the exact source commit.

```sh
python -I -m pip install setuptools==76.1.0 setuptools-scm==9.2.2 wheel==0.46.2
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.28.0+empty \
  python -I -m pip install --force-reinstall --no-deps --no-build-isolation \
  -r examples/kimi_k3_metax/requirements.txt
VLLM_VENDOR=metax MAX_JOBS=2 \
  python -I -m pip install --force-reinstall --no-deps --no-build-isolation .
```

For offline hosts, transfer complete Git repositories and install via
`git+file:///...@<the same commit>`. Directory installs or ZIP snapshots do not
provide the required dependency Git provenance. Preserve the original environment
and use a separate source checkout; do not overlay files onto a running service.

Every node needs a verified read-only path to the original K3 weights, tokenizer,
processor and chat template. Run once per node, changing rank 0 through 3:

```sh
python -I examples/kimi_k3_metax/launch.py \
  --model /models/Kimi-K3 --node-rank 0 \
  --master-addr MASTER_IP --master-port 29864 --host 0.0.0.0 --port 8000 \
  --cache /cache/k3/run-001/node-0 --net-interface RDMA_INTERFACE \
  --ib-hca HCA0:1,HCA1:1,HCA2:1,HCA3:1 --gid-index 3
```

The launcher checks installed dependency commits and rejects import shadowing
before model allocation. `--dry-run` prints the command without loading a model.
The policy remains `K3ExecutionConfig` plus `serve.yaml`; no new operator
environment switches are introduced. Existing transport settings use 16 channels
and 1 MiB buffers. Use a private writable compilation-cache directory per node.

Context 69,632; GPU memory 0.95; max sequences 64; MBT4096; PP layers
24/24/24/21; eager prefill; full decode Graph sizes 1/2/4/8/12/16/24/32/64;
prefix caching off. Containers require `/dev/dri`, `/dev/mxcd`,
`/dev/infiniband`, host networking, sufficient shared memory and unlimited memlock.

## Validation

Current ownership checks and installation smoke are described in
[VALIDATION.md](VALIDATION.md). Historical evaluations below are retained results
of the unified accepted arithmetic, not newly measured scores of this packaging.

| Task | Accuracy |
| --- | ---: |
| GPQA Diamond | 185/198, 93.43% |
| ERQA | 266/400, 66.50% |

Historical TP8 × PP4, MBT4096, C64, output1024 serving results. Each row retains
its successful snapshot; the long-context rows include the later memory fixes:

| Input tokens | Total tok/s | Output tok/s |
| ---: | ---: | ---: |
| 1024 | 560.057 | 280.029 |
| 4096 | 1192.596 | 238.519 |
| 16384 | 1196.806 | 70.400 |
| 65536 | 1553.939 | 23.907 |

AI assistance was used. Upstream approval and merge status remain separate from
the source pin and the reported validation.
