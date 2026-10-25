"""CPU contract harness for a runtime whose FlagTree import requires a device.

Executes declarations from the real source files; only framework base classes
and distributed/dispatch imports are stubbed. This is not a runtime-import or
GPU-kernel validation. Device tests live separately in test_qwen38_ple_device.
"""

import __future__

import ast
import math
import sys
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]


def declarations(path, name, namespace=None, names=None):
    module = ModuleType(name)
    module.__dict__.update(
        torch=torch,
        nn=nn,
        np=np,
        math=math,
        dataclass=dataclass,
        replace=replace,
        SimpleNamespace=SimpleNamespace,
        ContextVar=ContextVar,
        wraps=wraps,
    )
    if namespace:
        module.__dict__.update(namespace)
    sys.modules[name] = module
    tree = ast.parse(Path(path).read_text())
    body = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.Assign, ast.AnnAssign))
        and (names is None or getattr(node, "name", None) in names)
    ]
    exec(
        compile(
            ast.Module(body=body, type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        module.__dict__,
    )
    return module


class GenericBase:
    @classmethod
    def __class_getitem__(cls, _):
        return cls


loading = declarations(
    ROOT / "vllm_fl/models/qwen3_8_flash_next/loading.py",
    "qwen_cpu_loading",
    names={"PLEShardOverlap", "compute_ple_shard_overlap", "copy_ple_embedding_shard_"},
)
ngram = declarations(
    ROOT / "vllm_fl/models/qwen3_8_flash_next/ngram.py",
    "qwen_cpu_ngram",
    {
        "copy_ple_embedding_shard_": loading.copy_ple_embedding_shard_,
        "get_tensor_model_parallel_world_size": lambda: 1,
        "get_tensor_model_parallel_rank": lambda: 0,
        "tensor_model_parallel_all_reduce": lambda x: x,
    },
)
# Use the exact pinned v0.24 dataclass field contract, without importing its GPU modules.
mamba_path = ROOT.parent / "vllm-0.24.0/vllm/v1/attention/backends/mamba_attn.py"
mamba = declarations(mamba_path, "qwen_cpu_mamba", names={"BaseMambaAttentionMetadata"})
ple_metadata = declarations(
    ROOT / "vllm_fl/attention/qwen3_8_flash_next/ple_metadata.py",
    "qwen_cpu_metadata",
    {
        "AttentionBackend": GenericBase,
        "BaseMambaAttentionMetadata": mamba.BaseMambaAttentionMetadata,
        "BaseMambaAttentionMetadataBuilder": GenericBase,
        "AttentionCGSupport": SimpleNamespace(UNIFORM_BATCH=2),
        "mamba_get_block_table_tensor": lambda table, *args: table,
        "NULL_BLOCK_ID": 0,
    },
)
ple = declarations(
    ROOT / "vllm_fl/models/qwen3_8_flash_next/ple.py",
    "qwen_cpu_ple",
    {
        "MambaBase": GenericBase,
        "NULL_BLOCK_ID": 0,
        "resolve_op": lambda name: None,
    },
)
worker = declarations(
    ROOT / "vllm_fl/worker/qwen3_8_flash_next.py",
    "qwen_cpu_worker",
)
