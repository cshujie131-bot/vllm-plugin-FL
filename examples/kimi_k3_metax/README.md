# Kimi K3 on MetaX C550

The accepted **TP8 / PP4 / DP1, EP/SP off** implementation for four 8-GPU C550 nodes.
Original MXFP4/E8M0 expert weights stay packed; tiled expert computation and the
main attention/GEMM paths use BF16 with their accepted sensitive accumulation/state
precisions. This is not an INT8 conversion.

This PR packages the unified implementation used for text and ERQA evaluation.
It adds no tuning or dependency upgrades and does not replace validated library
calls merely to change their directory ownership.

## Review map

| Function | Files / dependency patch |
| --- | --- |
| Model components, latent TP/head packing, routing | `vllm_fl/k3/model_adapter.py`, `latent_moe.py`, `mla_adapter.py`, `kda_adapter.py` |
| Vision layout, RoPE/FFN chunking, attention | `vision_adapter.py`, `vision_attention.py` |
| One execution policy and explicit registration | `configuration.py`, `context.py`, `registration.py`, `op_registrar.py`, `vllm_fl/model_registration.py` |
| Linear shape policy and existing FlagGems calls | `linear.py`, `tp8_linear.py`, `flagos_mm.py` |
| MXFP4 loading and expert binding | `moe.py` |
| Runner-owned workspace and PP receive ownership | `model_runner.py`, `tp8_prefill_workspace.py`, `vllm_fl/worker/pipeline_input.py` |
| Default-preserving vLLM component factories | patch `01` |
| PP feedback, runner hook, Graph metadata, CUSTOM MLA selection | patch `02`, `execution_backends.py` |
| ATen addmm / argmax and accepted grouped-topk | patches `03` / `04` / `05` |
| Explicit MLA dense-backend callable | patch `06` |
| Independent expert, MLA, cache and SiTU kernels | patch `07`: `flaggems_vllm.ops.kimi_k3_metax` |
| Accepted KDA kernels and dependent launch wrappers | patch `08` |
| Existing MACA/MCCL ABI wrappers and host-only extensions | `vllm_fl/dispatch/backends/vendor/metax/`, `setup.py`, `csrc/metax/uva.cpp` |
| Optional startup/OOM evidence, separate from arithmetic | `vllm_fl/worker/runtime_diagnostics.py` |

Paths without a prefix in the table are under `vllm_fl/k3/`.
The patches in `dependencies/` are incremental, grouped by function; no full
operator library, deployment tree or private binary is included.
Unchanged model loading, processors, metadata and execution skeletons are reused
from public vLLM. The model forward methods are inherited from the pinned public vLLM factories.
KDA wrappers whose globals resolve changed kernels preserve accepted semantics.
Existing SDK wrappers are amended in place, not duplicated.

## Version contract

| Component | Accepted version / public source |
| --- | --- |
| Python | 3.12.13 |
| vLLM | 0.28.0, `2cf0a6915ce544dc493a0990f2ea38d81601128a`, empty build + patches 01–02 |
| FL target / packaging base | `0.4.0-dev` / `5809ea37735de890891d6a8bab91d27c112c745b` |
| Accepted FL source | `e88e4db75ece36e482a5fcc288929d4461d53184`; unrelated target-branch changes are retained |
| MACA SDK / runtime / MCCL | 3.8.1.3 |
| MetaX Torch | `2.10.0+metax3.8.1.0` |
| FlagTree / Triton API | `0.7.0+metax3.6` / 3.6.0, matching C550 compiler binaries |
| FlagGems | `ad9f11380686edac17edc97856bc83f058329290` + patches 03–05 |
| FlagGems-vllm | `3e9e47c988bf834204b416d2a0b3db8c91e7a4de` + patches 06–08 |
| Transformers / host driver | 5.15.1 / 3.8.1 |

Install matching vendor Torch, MACA SDK/runtime, MCCL and RDMA userspace libraries
first. FlagTree requires its matching MetaX LLVM and Triton-plugin binaries:
[official MetaX instructions](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-metax).
Compiler source reference: `3693980e3c2326b2bac7ad6fbcb3c1f6e42e6595`.
Do not substitute CUDA Torch/Triton or infer the SDK version from MX-SMI.

This profile uses no `vllm_metax`, `mcoplib` or FlagCX. MACA/MCCL and vendor Torch
are hardware runtime dependencies, not model-kernel fallbacks. The two host-only
FL extensions are built from source against that Torch and SDK.

## Prepare and launch

Use a dedicated Python 3.12 vendor environment and identical sources on all nodes.
The script refuses an existing destination and checks each exact commit/patch.

```bash
python examples/kimi_k3_metax/dependencies/prepare.py /opt/k3-deps
VLLM_TARGET_DEVICE=empty python -m pip install --no-deps --no-build-isolation /opt/k3-deps/vllm
python -m pip install --no-deps --no-build-isolation /opt/k3-deps/FlagGems
python -m pip install --no-deps --no-build-isolation /opt/k3-deps/FlagGems-vllm
VLLM_VENDOR=metax MAX_JOBS=2 python -m pip install --no-deps --no-build-isolation .
```

Use original Kimi K3 weights with their tokenizer/processor, chat template and
generation config. Every node needs a validated read-only weight path, not
necessarily a separate physical copy. Preserve the vendor Torch/FlagTree stack
when installing the pinned Python packages.

Run once on each node, changing rank 0 through 3 and the private cache directory:

```bash
python examples/kimi_k3_metax/launch.py   --model /models/Kimi-K3 --node-rank 0   --master-addr MASTER_IP --master-port 29864 --host 0.0.0.0 --port 8000   --cache /cache/k3/run-001/node-0   --net-interface RDMA_INTERFACE   --ib-hca HCA0:1,HCA1:1,HCA2:1,HCA3:1 --gid-index 3
```

`--dry-run` prints the command and environment. The model policy remains the
accepted `K3ExecutionConfig`; the example uses vLLM's native YAML configuration.
No per-operator environment switch is added. Existing plugin/cache/MCCL settings
are retained, including 16 channels / 1 MiB buffers.

The profile uses context 69,632, memory utilization 0.95, max sequences 64,
MBT 4096, PP layers 24/24/24/21, eager prefill and full decode Graph sizes
1/2/4/8/12/16/24/32/64. Prefix caching is off; interleaved multimodal strings are
on, as in ERQA. Use matching RDMA libraries/interface/HCA/GID. Containers need
`/dev/dri`, `/dev/mxcd`, `/dev/infiniband`, host networking, sufficient shared
memory and unlimited memlock. Never share a writable compilation cache.

## Validation and scope

Historical validation is reused, not relabeled as a fresh run of this PR:

- Backend lowering: 31 CPU contracts, 106 default-path method comparisons,
  KDA/vision checks and 146 four-node correctness requests.
- GPQA: native strict and frozen final-answer **185/198 (93.43%)**;
  native flexible **61/198 (30.81%)**, retained separately.
- MATH-500: native **471/500 (94.20%)**, frozen supplemental **463/500 (92.60%)**.
  Later semantic review is not substituted for either formal score.
- ERQA: native and frozen supplemental **266/400 (66.50%)**;
  400 normal completions, no truncation/retries.

These evaluations used one unified implementation, including the accepted KDA
FP32 chunk inputs and historical local FP64 normalization/gate branches.
This publication does not change those precision choices.

Earlier serving measurements, client concurrency 64, 128 formal requests, O1024:

| Input | Total tok/s | Output tok/s |
| --- | ---: | ---: |
| 1K | 560.057 | 280.029 |
| 4K | 1192.596 | 238.519 |
| 16K | 1196.806 | 70.400 |
| 64K | 1553.939 | 23.907 |

1K/4K used their retained successful versions; 16K/64K included the long-context
memory fixes. These are not fresh measurements of this PR layout. Each 16K round
had four preemptions and worse TPOT than the earlier PP baseline. No isolated
operator speedup or statistical significance is inferred.

Current packaging checks and new-entry smoke are in [VALIDATION.md](VALIDATION.md).
Source equivalence does not replace runtime acceptance.

The vLLM patches change eight files and do not change the request scheduler.
Default component factories retain upstream behavior. Centralized, idempotent,
version-checked MACA/MCCL registration still installs runtime hooks; avoiding
source rewriting does not make it hook-free. Shared MetaX adapter/config changes
also affect other models using this FL platform and must be reviewed as platform
changes. No other model implementation is modified.

Remaining explicit boundaries:

- `flagos_mm.py` uses pinned FlagGems MetaX kernel/tuner internals.
- Runner-owned workspace is budgeted before KV sizing; receive storage is
  released only after the fixed-input copy. Math kernels do not own those policies.
- Historical EP/TP32 dispatch and dedicated helpers are removed. The K3 policy
  accepts TP8/DP1 with EP off; PP feedback is selected by pipeline depth.
  Shared packed-expert arithmetic used by TP8 is retained.
- Lint found a missing `_decode_half_words` import in an existing non-normal-scale
  fallback. The existing same-library helper is now imported; its math is unchanged.
- Controllers, GPU eviction guards, benchmarks, profiles, logs, weights and caches
  are excluded. Library patches are not claimed merged upstream; #540/#541 are
  organization references, not K3 dependencies.

AI assistance was used. Review the hooks, dependency diffs and validation
boundaries before merging.
