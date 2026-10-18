# SPDX-License-Identifier: Apache-2.0
"""Optional MetaX tile policy for future config queries, not an environment hook."""

import triton

_enabled = False


def configure(enabled=True):
    global _enabled
    _enabled = bool(enabled)


def adjust(op_name, configs):
    if not _enabled or op_name not in ("mm", "addmm") or not isinstance(configs, list):
        return configs
    keys = (
        ("BLOCK_M", "BLOCK_N", "BLOCK_K")
        if op_name == "mm"
        else ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K")
    )
    result = []
    for config in configs:
        old = dict(config.kwargs)
        if not set(keys).issubset(old):
            result.append(config)
            continue
        kw = dict(old)
        stages = min(int(config.num_stages), 4)
        m, n, k = keys
        while (int(kw[m]) + int(kw[n])) * int(kw[k]) * 2 * stages > 56 * 1024:
            if int(kw[m]) >= int(kw[n]) and int(kw[m]) > 64:
                kw[m] = max(64, int(kw[m]) // 2)
            elif int(kw[n]) > 64:
                kw[n] = max(64, int(kw[n]) // 2)
            elif int(kw[k]) > 16:
                kw[k] = max(16, int(kw[k]) // 2)
            elif stages > 2:
                stages = max(2, stages // 2)
            else:
                break
        if kw != old or stages != config.num_stages:
            result.append(
                triton.Config(
                    kw,
                    num_warps=config.num_warps,
                    num_stages=stages,
                    num_ctas=getattr(config, "num_ctas", 1),
                    maxnreg=getattr(config, "maxnreg", None),
                )
            )
        else:
            result.append(config)
    return result
