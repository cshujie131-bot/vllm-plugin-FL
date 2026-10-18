# Source ownership

This is the evaluated DSV4 profile, kept in its original package namespaces.

- vllm_fl: model adaptation, TP headpacking/communication, query-row sharding, dispatch and scoped operator boundaries.
- flag_gems: ATen and retained fused implementations.
- flaggems_vllm: local MLA, MHC, quantization and migrated fused operators.
- dsv4_metax: independent MCTlass bindings and vendor-library bridge; it does not import vllm_metax.
- dsv4_runtime: explicit initialization and version-checked compatibility hooks.

Original copyright and license headers are retained. MHC pre's metax_pr755.py is
byte-identical to FlagGems-vllm PR #755 at
3a6ae4695cb63c8fe0a39774bf470e580fbbcd10. MHC post remains the previously validated
implementation. The old split implementation is only a guarded fallback, not
the active optimization being proposed.

The source delta is assembled with pinned existing dependencies. It is not a
replacement installation of FlagGems or FlagGems-vllm, and does not edit their
installed files. The full manifest verifies every assembled source/config file.
The baseline image's proprietary binaries, checkpoint, tuning databases and
compiler caches are not redistributed here.
