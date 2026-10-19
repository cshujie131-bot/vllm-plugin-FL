# Validation of the source-only K3 integration

## Current checks

- 110 source/default-path equivalence checks; relocated function bodies and launch settings retained.
- 29 CPU policy/launcher/PP/provenance tests, including the native root-parser regression for disabled prefix caching.
- Eight real import entries; the retired FGV K3 namespace is not loaded.
- Complete TP8 expert boundary: 12 shapes from M1 to M4096 exactly match the accepted implementation. Five Graph shapes with three changing inputs each also match exactly. Paired median wall-time differences are within about 0.5%; this is component evidence, not serving throughput.
- One reserved prefill arena per registered stream, 176 MiB, unchanged.
- Interleaved/contiguous MLA fallback: 32 output/LSE exact records, including changing-input Graphs.
- The model-selected TopK provider matches accepted weights/IDs for 24 eager records and 12 changing-input Graph replays. Registration respects the existing operator configuration and invalidates cached dispatch through public manager methods.
- Compatibility checks: 48 addmm cases and four MLA callback contracts. The native C550 addmm compilation failure is retained as negative evidence.
- vLLM Ruff 0.14.0, six repository local hooks and Mypy 1.20.2 for Python 3.12 pass. Full pre-commit bootstrap timed out fetching Git dependencies; it is not reported as passed.

The clean installed-package TP8/PP4 smoke passed **18/18** requests: 1K/4K/16K/64K
text, red/blue images, mixed concurrency, repeated-query consistency and eight
exact-length concurrent outputs. All logprobs were finite; 18 distinct API
response IDs and server increments match 99,936 prompt and 208 output tokens.
Prefix caching is disabled, Mamba cache mode is `none`, prefix hits and
preemptions are zero, and the final queue is idle. All 32 rank identities and
single 176 MiB arenas match; all four model containers exited with code 0.

The negative prefix-cache option is explicit in YAML: vLLM's root CLI parser
cannot resolve a subcommand's negative boolean action while converting a plain
`false`. The first smoke passed its functional checks but used prefix caching;
it is retained separately and is not substituted for the corrected-profile run.
The runtime package is the installed `4d746d70` source; the corrected entry
configuration is the one in this PR. No inference kernel changed during this
configuration repair.

Native cold autotuning rejected some over-budget shared-memory candidates and
selected supported configurations. These handled compile trials are distinct
from allocator OOM: no allocator-OOM evidence or fatal worker error was observed.
No new full quality evaluation or serving performance benchmark was run.

## Historical results retained

The unified backend-lowering implementation passed 146 four-node correctness
requests, including text/image and mixed-length inputs. The preceding published
entry point passed its fixed 18-request smoke. GPQA and ERQA scores in README
come from the earlier full evaluations. They are not new scores of this package.

The public operator PRs contain their own native references, scope, test commands
and standalone measurements. Source equivalence alone does not prove a new
installed entry point works; the installation smoke above is tracked separately.
