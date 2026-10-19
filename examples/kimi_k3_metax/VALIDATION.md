# Publication validation

## EP/TP32 removal follow-up

2026-10-19: removed the historical EP transport/warmup and TP32-only dispatch,
AttnRes window, expert entry points and dependency kernels. TP8 uses the existing
shared packed-expert primitives. Non-TP8, DP>1 and EP configurations now fail
explicitly instead of selecting the retired paths.

| Check | Result |
| --- | --- |
| CPU configuration and launcher contracts | 15 passed |
| Incremental dependency patches | All 8 apply with strict whitespace checks |
| Critical lint / internal import closure | Passed; 121 imports resolved, no cycles |
| Retained library functions and tuning assignments | 94 function ASTs and remaining tuning assignments unchanged |
| TP8 grouped-expert fallback and D/U host paths | Equivalent after specializing the old inactive TP32 branches |
| Linear dispatch | 1,811,472 CPU metadata comparisons agree across token counts 0–69,632 and TP8 projection shapes |
| Inherited model forwards | Old non-EP/non-TP32 decoder/model ASTs equal the pinned native methods |
| Real imports on an idle C550 | 25 modules loaded, 3 native forward bindings confirmed |
| PP / workspace integration | Feedback, receive lifetime, arena, TP8 expert arithmetic and PP dependency patch unchanged |

The import check used one idle device and released it afterward, without model
requests or performance tests. The prior 18-request four-node smoke below
predates this deletion-only scope change; it is retained as historical runtime
evidence, not relabeled as a new run. Full native imports require device
capability queries, so the initial no-device attempt stopped there; source checks
and the CPU contracts passed separately.

## Original publication checks

Run: pr_publication_20261019-r1, installed-entry model smoke r2, 2026-10-19 (UTC+8).

| Check | Result |
| --- | --- |
| Eight incremental patches against exact public commits | Applied successfully; added-line whitespace checks pass |
| Changed Python syntax / critical Ruff checks | Pass |
| Example and test lint; launcher contracts | Pass; 5 launcher cases |
| Mathematical functions/methods and module tuning assignments | 188 executable ASTs unchanged, ignoring docstrings/import-path relocation |
| Library import boundaries | No new library-to-plugin imports or moved-module cycles |
| FL wheel and host-only weak-reference/UVA extensions | Clean build; 269 packaged Python files match source exactly (generated version metadata excluded) |
| Host runtime ABI | 4 checks: pinned alias, unpinned copy, empty input, weak storage alias |
| Real installed-package imports | 29 K3 modules imported; registration is idempotent |
| Configuration contracts through installed vendor package | 8 cases passed |
| Four-node source publication | Identical pinned dependency sources and FL wheel |
| New-entry TP8/PP4 smoke | 18/18 requests passed |
| Server request/token reconciliation | 18 successes, 99,936 input / 208 output tokens; exact client/server agreement |
| Runtime queues / preemptions / allocator OOM | Drained queues, 0 preemptions, 0 allocator-OOM records |
| Workspace / startup accounting | 32 rank ledgers; one 184,549,376-byte workspace per rank |
| Cleanup | Four owned model containers stopped with exit 0; GPU process lists empty |

The 18 requests comprise ten semantic checks (1K/4K/16K/64K text, red/blue
images, mixed concurrent text/images and a repeated 4K input) and eight
P1024/O16 concurrent exact-length requests. All logprobs were finite. The eight
exact-length requests intentionally ignore EOS; this is not a quality benchmark.
This smoke uses non-thinking requests. Reused full evaluations used Kimi Max
thinking mode and their separately frozen protocols/scorers.

Shutdown logs retain native TCPStore broken-pipe/resource_tracker warnings after
the completed workload. Container exits are 0 and GPU processes are gone; this
publication does not change native teardown behavior.

Non-MetaX bootstrap/extension loading remains the upstream implementation; the
MetaX open-runtime branch retains its accepted statements.

The only function-body change isolated from the 188 equivalence comparisons is
installation plumbing that locates the source-built UVA extension. A previously
missing non-normal-scale helper import was corrected; numerical helper bodies,
dtype/cast boundaries, tuning tables and workspace/Graph/PP policies remain.

The first packaging launch stopped before any generation because the example
API-server count conflicted with headless peers. The launcher now sets one API
server on rank 0 and zero on peers, with explicit regression tests. Failed
records remain separate from successful r2. Preliminary import and publication
preflight failures were retained as well.

The real deployment image was compared against public vLLM. The final default
vLLM delta is eight files; no private image modifications or old DP/engine
diagnostic copies are silently assumed. Running smoke dependency sources and
final patches have identical executable Python ASTs. Blank initializer/EOF
patch noise removed during scope review was non-executable.

Reproduce the CPU/example checks in the prepared environment:

    python -m unittest discover -s tests/k3
    python examples/kimi_k3_metax/launch.py --help
    python examples/kimi_k3_metax/dependencies/prepare.py /new/source/directory

The repository's existing Ruff exclusion of vllm_fl is unchanged. Critical lint
was additionally run over every changed Python file, including that package.
Unified-diff context whitespace is preserved: raw patches are excluded only
from rewriting/spelling hooks and checked with git apply --check
--whitespace=error-all. No GitHub diff folding or source hiding is added.

No new full accuracy evaluation, performance matrix or profile was run.
Historical results and their version boundaries are in README.md.
Other-device/model acceptance is not claimed by the C550 smoke. Shared MetaX
runtime/provider changes remain an explicit review boundary.
