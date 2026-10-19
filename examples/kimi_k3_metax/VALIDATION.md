# Validation of the source-only K3 integration

## Current checks

- 110 source/default-path equivalence checks; relocated function bodies and launch settings retained.
- 28 CPU policy/launcher/PP/provenance tests.
- Eight real import entries; the retired FGV K3 namespace is not loaded.
- Complete TP8 expert boundary: 12 shapes from M1 to M4096 exactly match the accepted implementation. Five Graph shapes with three changing inputs each also match exactly. Paired median wall-time differences are within about 0.5%; this is component evidence, not serving throughput.
- One reserved prefill arena per registered stream, 176 MiB, unchanged.
- Interleaved/contiguous MLA fallback: 32 output/LSE exact records, including changing-input Graphs.
- The model-selected TopK provider matches accepted weights/IDs for 24 eager records and 12 changing-input Graph replays. Registration respects the existing operator configuration and invalidates cached dispatch through public manager methods.
- Compatibility checks: 48 addmm cases and four MLA callback contracts. The native C550 addmm compilation failure is retained as negative evidence.
- vLLM Ruff 0.14.0, six repository local hooks and Mypy 1.20.2 for Python 3.12 pass. Full pre-commit bootstrap timed out fetching Git dependencies; it is not reported as passed.

Clean installed-package import/ABI checks and the four-node new-entry smoke are
pending for this candidate. They must be completed before claiming deployment
acceptance. No new full quality evaluation or serving performance benchmark is
part of this ownership cleanup.

## Historical results retained

The unified backend-lowering implementation passed 146 four-node correctness
requests, including text/image and mixed-length inputs. The preceding published
entry point passed its fixed 18-request smoke. GPQA and ERQA scores in README
come from the earlier full evaluations. They are not new scores of this package.

The public operator PRs contain their own native references, scope, test commands
and standalone measurements. Source equivalence alone does not prove a new
installed entry point works; the installation smoke above is tracked separately.
