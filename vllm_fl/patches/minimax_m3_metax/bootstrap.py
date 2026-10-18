# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Ordered initialization and effective-policy assertions for the sealed M3 deployment."""

import importlib, importlib.metadata, inspect, json, os, sys, hashlib
from pathlib import Path
from dataclasses import asdict
from vllm_fl.minimax_m3_config import load, thaw

_INITIALIZED = False


def _identity(fn):
    try:
        p = inspect.getsourcefile(fn)
    except TypeError:
        p = None
    return dict(
        module=getattr(fn, "__module__", None),
        qualname=getattr(fn, "__qualname__", type(fn).__qualname__),
        source=p,
        sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()
        if p and Path(p).is_file()
        else None,
    )


def initialize():
    global _INITIALIZED
    if _INITIALIZED:
        return
    profile = load()
    from .startup import configure_libraries

    configure_libraries()
    import torch
    import triton
    import triton.language
    import flag_gems
    from vllm.platforms import current_platform

    assert getattr(current_platform, "vendor_name", None) == "metax"

    assert torch.__version__ == "2.10.0+metax3.8.1.0"
    assert importlib.metadata.version("vllm").split("+", 1)[0] == "0.24.0"
    assert importlib.metadata.version("flagtree") == "0.6.1+metax3.6"
    assert (
        Path(triton.__file__)
        .resolve()
        .is_relative_to(Path(os.environ["FLAGTREE_ROOT"]).resolve())
    )
    # Correctness-sensitive registration must be fixed before WorkerFL enables ATen.
    from vllm_fl.utils import (
        get_flag_gems_whitelist_blacklist,
        get_oot_blacklist,
        is_oot_enabled,
    )
    from vllm_fl.dispatch.config import get_effective_config, get_flagos_blacklist

    assert is_oot_enabled() is True
    assert {k: get_effective_config()[k] for k in profile.dispatch} == thaw(
        profile.dispatch
    )
    assert get_flag_gems_whitelist_blacklist() == (
        None,
        list(profile.dispatch["flagos_blacklist"]),
    )
    assert get_flagos_blacklist() == list(profile.dispatch["flagos_blacklist"])
    assert get_oot_blacklist() == list(profile.dispatch["oot_blacklist"])
    generic = importlib.import_module("flag_gems.ops.masked_fill").masked_fill_
    assert flag_gems.masked_fill_ is generic
    selected = {
        row[0]: row[1]
        for row in flag_gems._FULL_CONFIG
        if row[0] in {"masked_fill_.Scalar", "masked_fill_.Tensor"}
    }
    assert len(selected) == 2 and all(fn is generic for fn in selected.values())
    align = importlib.import_module("flag_gems.fused.moe_align_block_size")
    assert (
        "allow_tle" in inspect.signature(align.moe_align_block_size_triton).parameters
    )
    assert profile.kernels.moe_alignment_tle is False
    from vllm_fl.ops.minimax_m3.attention import initialize_consumer_contract

    initialize_consumer_contract()
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        FusedTopKBiasRouter,
    )

    assert (
        FusedTopKBiasRouter._compute_routing.__module__
        == "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router"
    )
    from vllm import _custom_ops

    assert _custom_ops.moe_sum.__module__ == "vllm._custom_ops"
    import vllm.model_executor.layers.fused_moe as native_moe
    import vllm.model_executor.layers.fused_moe.oracle.unquantized as native_oracle

    assert (
        native_moe.FusedMoE.__module__ == "vllm.model_executor.layers.fused_moe.layer"
    )
    assert (
        native_oracle.select_unquantized_moe_backend.__module__
        == "vllm.model_executor.layers.fused_moe.oracle.unquantized"
    )
    assert not any(
        n == "vllm_metax" or n.startswith("vllm_metax.") for n in sys.modules
    )
    _INITIALIZED = True
    audit_effective("import")


def audit_effective(stage, runner=None):
    import torch, triton, flag_gems
    from vllm_fl.dispatch import CachedOp, get_default_manager, get_policy
    from vllm_fl.utils import get_flag_gems_whitelist_blacklist, get_oot_blacklist
    from vllm_fl.ops.minimax_m3 import layers, msa, vision

    mgr = get_default_manager()
    mgr.ensure_initialized()
    bindings = {}
    for module in (layers, msa, vision):
        for name, obj in vars(module).items():
            if isinstance(obj, CachedOp):
                impl = obj.bind() if stage == "import" else obj.inspect_binding()
                bindings[obj.op_name] = {
                    "impl_id": impl.impl_id,
                    "kind": str(impl.kind),
                    "function": _identity(impl.fn),
                }
    for name in (
        "m3_moe_align_block_size",
        "m3_invoke_fused_moe_triton_kernel",
        "m3_moe_sum",
        "m3_grouped_topk",
        "attention_backend",
    ):
        impl = mgr._resolve_impl(
            name
        )  # inspection only; CachedOp owns all mutable binding state
        bindings[name] = {
            "impl_id": impl.impl_id,
            "kind": str(impl.kind),
            "function": _identity(impl.fn),
        }
    if stage == "after_capture":
        registered = set(flag_gems.aten_lib._op_impls)
        assert {
            "aten/masked_fill_.Scalar/CUDA",
            "aten/masked_fill_.Tensor/CUDA",
        } <= registered
        assert (
            not {"aten/masked_fill.Scalar/CUDA", "aten/masked_fill.Tensor/CUDA"}
            & registered
        )
    bad = [
        k
        for k in os.environ
        if k.startswith("M3_")
        or k
        in (
            "VLLM_FL_FLAGOS_BLACKLIST",
            "VLLM_FL_FLAGOS_WHITELIST",
            "VLLM_FL_OOT_BLACKLIST",
            "VLLM_FL_OOT_WHITELIST",
        )
    ]
    assert not bad, ("Legacy project env survived", bad)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else None
    original_model_modules = [
        name
        for name in sys.modules
        if name == "vllm.models.minimax_m3"
        or name.startswith("vllm.models.minimax_m3.")
    ]
    record = dict(
        stage=stage,
        pid=os.getpid(),
        rank=rank,
        profile_sha256=load().fingerprint,
        framework=asdict(load().framework),
        kernels=asdict(load().kernels),
        dispatch=thaw(load().dispatch),
        whitelist_blacklist=get_flag_gems_whitelist_blacklist(),
        oot_blacklist=get_oot_blacklist(),
        bindings=bindings,
        policy=str(get_policy()),
        triton_module=triton.__file__,
        native_router_unchanged=True,
        native_moe_sum_unchanged=True,
        original_model_loaded=bool(original_model_modules),
        effective_env={
            k: v
            for k, v in os.environ.items()
            if k.startswith(
                (
                    "VLLM_",
                    "FLAG",
                    "GEMS",
                    "USE_FLAG",
                    "TRITON_",
                    "TORCHINDUCTOR_",
                    "MCCL_",
                    "NCCL_",
                    "GLOO_",
                    "MACA_",
                    "PYTHONPATH",
                )
            )
        },
    )
    if runner is not None:
        record["runner"] = type(runner).__module__ + "." + type(runner).__name__
    root = Path(os.environ["VLLM_CACHE_ROOT"]).parent / "audit"
    root.mkdir(parents=True, exist_ok=True)
    (
        root / (stage + "-rank" + str(rank) + "-pid" + str(os.getpid()) + ".json")
    ).write_text(json.dumps(record, indent=2) + "\n")
    print(
        "M3_EFFECTIVE_CONFIGURATION "
        + json.dumps(
            dict(
                stage=stage,
                rank=rank,
                profile=load().fingerprint,
                bindings=len(bindings),
            )
        ),
        flush=True,
    )
    return record


def audit_memory_budget(worker, returned_bytes):
    """Record already-computed public budget fields without GPU operations."""
    import torch

    record = {
        "rank": torch.distributed.get_rank(),
        "profile_sha256": load().fingerprint,
        "requested_memory": worker.requested_memory,
        "model_memory_usage": worker.model_runner.model_memory_usage,
        "peak_activation_memory": worker.peak_activation_memory,
        "non_torch_memory": worker.non_torch_memory,
        "cudagraph_memory_estimate": worker.cudagraph_memory_estimate,
        "available_kv_cache_memory_bytes": worker.available_kv_cache_memory_bytes,
        "returned_bytes": returned_bytes,
        "initial_free_memory": worker.init_snapshot.free_memory,
        "total_memory": worker.init_snapshot.total_memory,
    }
    root = Path(os.environ["VLLM_CACHE_ROOT"]).parent / "audit"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"memory-rank{record['rank']}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print("M3_MEMORY_BUDGET " + json.dumps(record), flush=True)
