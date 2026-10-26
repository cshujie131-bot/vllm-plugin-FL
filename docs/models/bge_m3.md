# BGE-M3 selected joint outputs

The FL BGE-M3 model can return the FlagEmbedding encode result in one vLLM request. One encoder pass produces shared hidden states; only the selected dense, sparse and ColBERT heads run. Native pooling tasks remain available with their existing launch configurations. This is a mapping of the official Python result contract into vLLM's IO processor HTTP transport.

## Launch

Load the platform's validated FL/FG/FGV runtime environment first. Allow both plugins (the vLLM allowlist applies to every plugin group):

```bash
export VLLM_PLUGINS=fl,bge_m3
export VLLM_FL_CONFIG=/path/to/vllm-plugin-FL/vllm_fl/dispatch/config/bge_m3_metax.yaml
vllm serve /models/bge-m3 \
  --served-model-name bge-m3 --runner pooling \
  --hf-overrides '{"architectures":["BgeM3EmbeddingModelFL"]}' \
  --pooler-config '{"task":"embed"}' --io-processor-plugin bge_m3 \
  --dtype float32 --tensor-parallel-size 1 --enforce-eager \
  --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 8192
```

The default `embed` task keeps `/v1/embeddings` usable. `/pooling` requests with `task=plugin` select any combination in the same loaded model. The internal joint pooler is installed only when `io_processor_plugin=bge_m3` is selected; it is not exposed by ordinary launches.

## Request and response

```json
{
  "model": "bge-m3",
  "task": "plugin",
  "data": {
    "sentences": ["什么是向量检索？", "How does retrieval work?"],
    "return_dense": true,
    "return_sparse": true,
    "return_colbert_vecs": true,
    "max_length": 8192
  }
}
```

POST to `/pooling`. `input` is accepted as an alias for `sentences`. The standard vLLM response envelope has `request_id`, `created_at` and `data`; `response.data` has exactly the official keys:

| Key | List input | String input |
|---|---|---|
| dense_vecs | B × 1024 float lists | 1024 floats |
| lexical_weights | B dictionaries of string token IDs to weights | One dictionary |
| colbert_vecs | B arrays, each (L_i − 1) × 1024 | One (L − 1) × 1024 array |

JSON lists represent the NumPy arrays returned by FlagEmbedding. A disabled field is `null`; all eight flag combinations, including all disabled, are accepted. Flags default to `true/false/false`; null flags resolve to those defaults. Sparse weights filter CLS/EOS/PAD/UNK and nonpositive values and merge repeated IDs by maximum. ColBERT removes CLS and batch padding, retaining EOS and literal input tokens.

`max_length` defaults to 512, matching the fixed FlagEmbedding 1.4.2 encode default; explicitly pass 8192 for longer documents. HF tokenization truncates on the right while preserving EOS. This differs from the legacy endpoint's overlong-input rejection. `max_length` must be 2–8192 and no larger than the configured model limit. Empty strings are valid; an empty list, non-text values, unknown options or malformed flags are rejected. An input list of length one retains its batch dimension.

## Implementation and ownership

`BgeM3EmbeddingModelFL` reuses the upstream model, loaders and heads. `BgeM3JointPooler` applies the existing heads to one encoder result and packs selected tensors for vLLM's existing tensor transport. `BgeM3IOProcessor` tokenizes, sets per-request pooling options and restores the official output structure. The private transport header never appears in the public result.

Request options are isolated using ContextVar in vLLM's preprocessing thread; response options travel with each result, not a global FIFO. A private tokenizer and a bounded lock isolate mutable HF truncation settings from concurrent plugin and native requests. No request map is retained across cancellations.

This work is model-specific output semantics and framework orchestration, so it belongs in FL. It contains no kernel implementation, changes no vLLM source file and claims no kernel speedup. The MetaX FP32 path requires [FGV #995](https://github.com/flagos-ai/FlagGems-vllm/pull/995), pinned to `60f1d386c94d1084f3f5f02f71714b7bc6581e6b`. Tests cover all flag combinations, independent reference values, mixed worker batches, concurrency, shapes, errors and legacy registration.

Official contract: https://bge-model.com/bge/bge_m3.html#usage
vLLM IO extension: https://docs.vllm.ai/en/v0.24.0/models/pooling_models/specific_models/

## Ownership layout

- `models/bge_m3/`: model/position semantics, packed attention adapter, joint pooling and BGE IO contract.
- `patches/bge_m3/`: lazy architecture and IO entry-point registration.
- `dispatch/backends/flaggems/impl/bge_m3.py`: registered FGV varlen provider; no kernel implementation.
- `configs/bge_m3_metax.py`: measured Linear launch candidates and scope.
- `models/bge_m3/linear.py`: per-model-instance binding, preserving native methods and parameter ownership.
- `dispatch/backends/vendor/metax/impl/bge_m3.py`: narrow MetaX attention selection and model-owned Linear launch policy, activated by `dispatch/config/bge_m3_metax.yaml`.
- `worker/worker.py`: generic asynchronous pooling-output compatibility, with generic worker tests.
- `tests/unit_tests/models/bge_m3/` and `tests/unit_tests/dispatch/test_bge_m3_policy.py`: model and dispatch contracts.

The public FlagGems backend remains unmodified. Model-owned Linear tuning now lives in plugin configuration and uses the original FG kernel; no separate FG fork or optimization PR is required. FG's existing MetaX masked_fill fix (#6480) passed 72 GPU cases, so no model blacklist or new implementation is needed.

FP32 encoder attention calls the existing FGV flash_attn_varlen_func once per packed batch. [FGV #995](https://github.com/flagos-ai/FlagGems-vllm/pull/995) extends the already merged MetaX implementation (#808) with IEEE FP32 QK/PV, a resource bound for wider heads, and explicit unsupported-feature errors. It supports non-paged bidirectional forward attention, unequal query/key lengths, GQA, D=16/32/64/128/256, packed/strided inputs, output buffers and LSE. The FP32 support introduces no torch compute fallback. Its subsequent context-length launch configuration is included in the same FGV PR. Causal/paged/windowed attention, ALiBi, softcap and backward remain outside the new FP32 scope; existing half paths are preserved.

Upstream #759 was tested unchanged on C550: FP16 passed, but FP32 required 72 KiB shared memory and failed against the 64 KiB limit. FlagTree #1191/#1258 were tested as a minimal Python backport, but did not fix the nested-helper tiny SDPA case; the backport was reverted. The new packed path does not call that SDPA kernel or need its former AABS workaround. The required FGV dependency is pinned below; FL contains only its public-op integration.


## Model-owned Linear tuning

`configs/bge_m3_metax.py` contains the two additional M32/N64/K32/W8/S4 and M64/N64/K32/W8/S4 candidates, the M<=1024 limit, and the measured `(N,K)` pairs `(3072,1024)`, `(1024,1024)`, `(4096,1024)`, `(1024,4096)`.

The existing `bge_m3_metax.yaml` policy selects `bge_m3_linear`. The provider is resolved once at model construction, avoiding per-call policy lookup. Binding is limited to this BGE model's unquantized Linear instances, including the ColBERT projection; parameter objects, weight loading and native layer methods are retained. The vendor provider builds an independent tuner over FG's existing `linear_kernel`, copying candidate objects before adding the model settings. No FG global function, candidate list or kernel is replaced. vLLM weight wrappers are exposed to the launcher through a no-copy Tensor view; the layer retains its original Parameter and storage.

Only contiguous FP32 CUDA inference in the measured region uses these settings. M is the total flattened row count, so a packed eight-document 1K encoder batch has M=8192 and keeps its native method. Other shapes/dtypes, gradients, noncontiguous tensors and batch-invariant mode retain the original layer call. The default policy also retains that call. Training, compile/graph and additional hardware are not new validation targets.

The FGV dependency below includes the existing FP32 attention support and its context configuration. FG remains at the original baseline revision. Pure model parameter tuning is maintained in this model PR, not a separate operator PR.

## Validated dependency pins

| Component | Revision |
|---|---|
| vLLM | 0.24.0, ee0da84ab9e04ac7610e28580af62c365e898389 |
| FlagGems | 4897cc306cd83390c95b83b50df28f7edb1684c5 |
| FlagGems-vllm | [#995](https://github.com/flagos-ai/FlagGems-vllm/pull/995), [60f1d386c94d1084f3f5f02f71714b7bc6581e6b](https://github.com/flagos-ai/FlagGems-vllm/commit/60f1d386c94d1084f3f5f02f71714b7bc6581e6b) |

Validation used MetaX C550 64 GiB, MACA 3.8.1.3, Torch 2.10.0+metax3.8.1.0 and FlagTree 0.6.1+metax3.6. Preserve the matched vendor runtime. The FGV PR is not merged; install the exact public-fork revision without replacing its vendor dependencies:

```bash
python -m pip install --no-deps --no-build-isolation \
  'git+https://github.com/cshujie131-bot/FlagGems-vllm.git@60f1d386c94d1084f3f5f02f71714b7bc6581e6b'
```

## Out-of-box performance: historical unoptimized B0

These are the previously measured **unoptimized B0** results from 2026-10-23, before the joint-output interface and packed FP32 varlen integration in this PR. Each mode ran as a separate pooling task. They are a baseline reference, not throughput measurements of the current joint endpoint. Machine-readable provenance and all six rows are in [bge_m3_baseline.json](bge_m3_baseline.json).

C550 64 GiB, TP1, FP32, eager, max_num_seqs=8, max_num_batched_tokens=8192; 64 requests, concurrency 8, one document per request, one warmup round and one measured round, HTTP keepalive=600 seconds. The client read and parsed complete JSON outputs; sparse filtering/duplicate-ID max are included. All six cases completed 64/64 with no retries or failures.

| Output | 1024 tokens/document, docs/s | 4096 tokens/document, docs/s |
|---|---:|---:|
| dense | 28.698 | 5.167 |
| sparse | 28.764 | 5.166 |
| ColBERT | 1.854 | 0.426 |

docs/s measures input-document throughput. ColBERT includes its full token-vector response (about 22.15/88.63 MB per document at 1K/4K), so these HTTP numbers are not encoder-only compute throughput. These historical B0 numbers do not measure the later model-owned Linear configuration or FGV context configuration.
