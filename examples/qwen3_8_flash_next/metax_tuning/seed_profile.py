# SPDX-License-Identifier: Apache-2.0
"""Seed reviewed offline choices via the existing LibTuner ConfigCache API."""

import importlib
import json
import os
from pathlib import Path

import torch
import triton
from flag_gems.utils.code_cache import config_cache_dir

assert os.environ.get("FLAGGEMS_CACHE_DIR"), "Use a dedicated FLAGGEMS_CACHE_DIR"
root = Path(__file__).parent
profile = json.loads((root / "metax-c550-bf16.json").read_text())
assert "C550" in torch.cuda.get_device_name(), torch.cuda.get_device_name()
dummy = torch.empty(1, device="cuda", dtype=torch.bfloat16)
written = []
for row in profile["shapes"]:
    operator = row["operator"]
    entry = getattr(
        importlib.import_module("flag_gems.ops." + operator), operator + "_kernel"
    )
    tuner = entry.fn
    shape = row["shape"]
    c = row["config"]
    if operator == "linear":
        args = dict(
            input_ptr=dummy, weight_ptr=dummy, bias_ptr=dummy, output_ptr=dummy, **shape
        )
    else:
        args = dict(A=dummy, B=dummy, C=dummy, N=shape["M"], M=shape["K"])
    key = tuner.get_key(args)
    cfg = triton.Config(
        {k: v for k, v in c.items() if k not in ("num_warps", "num_stages")},
        num_warps=c["num_warps"],
        num_stages=c["num_stages"],
    )
    assert any(
        x.kwargs == cfg.kwargs
        and x.num_warps == cfg.num_warps
        and x.num_stages == cfg.num_stages
        for x in tuner.configs
    ), (operator, shape, c)
    existing = tuner.cache.get(key)
    if existing is not None:
        assert (
            existing.kwargs == cfg.kwargs
            and existing.num_warps == cfg.num_warps
            and existing.num_stages == cfg.num_stages
        ), "Choose a fresh FLAGGEMS_CACHE_DIR; ConfigCache preserves existing entries."
    tuner.cache[key] = cfg
    actual = tuner.cache[key]
    assert (
        actual.kwargs == cfg.kwargs
        and actual.num_warps == cfg.num_warps
        and actual.num_stages == cfg.num_stages
    )
    written.append(
        dict(
            operator=operator,
            shape=shape,
            key=key,
            config=c,
            table=tuner.config_table_name,
        )
    )
print(
    json.dumps(
        dict(cache=str(config_cache_dir()), rows=len(written), entries=written),
        indent=2,
    )
)
