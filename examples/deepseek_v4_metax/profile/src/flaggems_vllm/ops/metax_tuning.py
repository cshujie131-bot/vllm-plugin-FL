# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Library-owned MetaX kernel policy. Configure once before tracing/inference.

This module has no vLLM/FL/process-environment dependency.
Input semantics (e.g. the number of real heads) remain explicit API arguments.
"""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class KernelPolicy:
    decode_split_topk: bool = True
    decode_splits: int = 4
    decode_block_k: int = 16
    decode_block_h: int = 16
    decode_warps: int = 4
    decode_stages: int = 1
    decode_max_batch: int = 64
    decode_half_d: bool = True
    decode_qkfold: bool = True
    prefill_half_d: bool = True
    prefill_blockdiag: bool = True
    prefill_staged: bool = True
    mhc_split_tail: bool = True
    mhc_small_m: bool = True
    mhc_max_m: int = 128
    mhc_splits: int = 8
    mhc_block_m: int = 16
    mhc_block_k: int = 64
    mhc_warps: int = 4
    mhc_stages: int = 1
    mhc_pad_n: int = 32
    mhc_tail_block_h: int = 8192
    mhc_tail_warps: int = 4
    mhc_tail_stages: int = 1
    enable_metax_gdn_compiler_patch: bool = False


current = KernelPolicy()
_configured = False


def configure(values: dict) -> None:
    global current, _configured
    unknown = set(values) - {f.name for f in fields(KernelPolicy)}
    if unknown:
        raise ValueError(f"Unknown kernel tuning fields: {sorted(unknown)}")
    for field in fields(KernelPolicy):
        if field.name in values and type(values[field.name]) is not field.type:
            raise TypeError(f"Invalid type for kernel field {field.name}")
    selected = KernelPolicy(**values)
    if selected.decode_block_h != 16 or selected.decode_splits != 4:
        raise ValueError("This validated profile requires decode BH16/S4")
    if selected.enable_metax_gdn_compiler_patch:
        raise ValueError("GDN compiler hook is not validated on this DS stack")
    if _configured and selected != current:
        raise RuntimeError("Kernel policy cannot change after initialization")
    current = selected
    _configured = True
