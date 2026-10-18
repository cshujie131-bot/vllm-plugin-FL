"""Deterministic one-shot startup. Explicit imports/providers; no runtime source edits."""

import importlib
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from .profile import ROOT, load
from .compat import install_vendor_guard, install_serial_execution

_initialized = False


def initialize():
    global _initialized
    if _initialized:
        return
    profile = load()
    install_vendor_guard()
    import torch

    torch._dynamo.config.ignore_logger_methods.update(
        getattr(logging.Logger, name)
        for name in (
            "debug",
            "info",
            "warning",
            "error",
            "exception",
            "critical",
            "log",
        )
    )
    # Preserve the accepted order: existing FlagGems tuners are created first;
    # the tile policy only affects subsequent config queries, as before.
    from vllm_fl import dsv4_config

    install_serial_execution(profile["compatibility"]["multi_stream_source_sha256"])
    import flag_gems
    from flag_gems.runtime import metax_launch_policy

    metax_launch_policy.configure(True)
    import triton.backends.metax.compiler as compiler

    compiler_entry = compiler.metax.translate_llvmir_to_mcfatbin
    import flaggems_vllm
    from flaggems_vllm.ops import metax_tuning

    # Decorator-time defaults are generated from this exact sealed profile.
    assert asdict(metax_tuning.KernelPolicy()) == profile["kernels"]
    metax_tuning.configure(profile["kernels"])
    dsv4_config.configure(profile["framework"], profile["moe"])
    assert flaggems_vllm.current_work_registrar is None
    assert compiler.metax.translate_llvmir_to_mcfatbin is compiler_entry
    from .rms_policy import install as install_rms

    install_rms(profile["rms"], "/var/lib/dsv4/startup_audit")
    from vllm_fl.ops.dsv4_validation import configure as configure_validation

    configure_validation(
        json.loads((ROOT / "config/hash_tables.json").read_text()),
        "/var/lib/dsv4/startup_audit",
    )
    from vllm_fl.dispatch import CachedOp, get_default_manager

    from vllm_fl.utils import get_flag_gems_whitelist_blacklist
    from vllm_fl.dispatch.config import get_oot_blacklist
    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    assert whitelist is None and blacklist == profile["dispatch"]["flagos_blacklist"], "ATen policy differs from the sealed profile"
    assert get_oot_blacklist() == profile["dispatch"]["oot_blacklist"]
    mgr = get_default_manager()
    mgr.ensure_initialized()
    modules = (
        "vllm_fl.models.deepseek_v4",
        "vllm_fl.ops.deepseek_v4_attention",
        "vllm_fl.ops.sparse_attn_indexer",
        "vllm_fl.ops.layernorm",
        "vllm_fl.ops.rotary_embedding",
        "vllm_fl.ops.activation",
        "vllm_fl.ops.fused_moe.router",
        "vllm_fl.ops.fused_moe.fused_moe",
        "vllm_fl.ops.fused_moe.fused_moe_utils",
        "vllm_fl.ops.fused_moe.activation",
        "vllm_fl.ops.fused_moe.gated_linear",
        "vllm_fl.quantization.mxfp4.mxfp4_marlin",
        "vllm_fl.platform",
    )
    bindings = {}
    for name in modules:
        mod = importlib.import_module(name)
        for attr, obj in vars(mod).items():
            if isinstance(obj, CachedOp):
                try:
                    impl = obj.bind()
                except Exception:
                    if obj._op_name == "deepseek_v4_mega_moe_experts":
                        continue
                    raise
                bindings[name + "." + attr] = impl.impl_id
    assert not any(
        n == "vllm_metax" or n.startswith("vllm_metax.") for n in sys.modules
    )
    record = {
        "release": profile["release"],
        "pid": os.getpid(),
        "bindings": bindings,
        "kernel_policy": asdict(metax_tuning.current),
        "framework_policy": asdict(dsv4_config.current),
        "aten_blacklist": blacklist,
        "vendor_imports": [],
        "compiler_entry_unchanged": True,
        "global_aten_fgv_registration": False,
    }
    folder = Path("/var/lib/dsv4/startup_audit")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"bootstrap_{os.getpid()}.json").write_text(json.dumps(record, indent=2))
    _initialized = True
    print(
        "DSV4_CLEAN_BOOTSTRAP "
        + json.dumps({"release": profile["release"], "bound_ops": len(bindings)}),
        flush=True,
    )
