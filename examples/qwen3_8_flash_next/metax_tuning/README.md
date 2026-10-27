# Qwen3.8-Flash-Next MetaX tuning profile

This opt-in profile packages the reviewed BF16/TP8 C550 launch parameters.
It reuses existing FG, FGV and native vLLM kernels; there are no copied kernel
implementations. The small patches expose measured launch choices in the
existing MetaX heuristics/configuration and retain defaults on geometry misses.

The fixed dependencies are recorded in manifest.json. FGV must already contain
the exact PLE file from [FGV #959](https://github.com/flagos-ai/FlagGems-vllm/pull/959)
at 22194f0b2ee0cebb51f945a20a157b82c4e19c65 and the MetaX inplace-MoE export
from [FGV #1000](https://github.com/flagos-ai/FlagGems-vllm/pull/1000) at
e04f51c1c7176d7a356838fdfa8b3dc4fbaf6f8a. The setup checks source hashes before
writing anything. It creates a new private dependency directory and never edits
the supplied libraries. Do not reuse it for different dependency versions.

Run in a fresh, dedicated container in the matching MetaX environment, with
explicit paths to each Python package:

```bash
python prepare.py \
  --flag-gems-package /path/to/flag_gems \
  --fgv-package /path/to/flaggems_vllm \
  --vllm-package /path/to/vllm \
  --output /path/to/new-private-dependencies
source /path/to/new-private-dependencies/env.sh
FLAGGEMS_CACHE_DIR="$HOME/.flaggems" python seed_profile.py
```

The generated environment unsets shared cache overrides. Serving uses native
separate locations: FG in $HOME/.flaggems and FGV in $HOME/.flagdnn. The inline
FLAGGEMS_CACHE_DIR above applies only to the FG seed process; do not export it
for serving. seed_profile.py uses the official LibTuner ConfigCache API and
rejects conflicting existing entries. MoE uses the official
VLLM_TUNED_CONFIG_FOLDER lookup. Keep the plugin source on PYTHONPATH as usual.

Add `"qwen38_qsa_length_buckets": true` to the existing additional_config alongside
the shared N-gram and MTP graph options. QSA scores use width buckets
512/2048/8192/cache-capacity; target and MTP graphs distinguish the width.
The score tile remains 32. TopK retains its two-stage algorithm and fixed K512.
The profile contains no communication change and disables no additional FG/FGV
operators. Parameter overrides are intended only for this dedicated model process.

After setup, retain prepared.json with the serving configuration and results.
Formal performance must be measured with these prepared dependencies, not a
different exploratory source tree. Refer to docs/models/qwen3_8_flash_next.md
for the serving contract, results and validation scope.
