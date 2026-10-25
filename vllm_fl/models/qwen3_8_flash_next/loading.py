# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from plugin-FL PR573 at 45b24bc7748720e0022a72e47a983398ddda36d3.
from __future__ import annotations

from collections.abc import Iterable

from dataclasses import dataclass

import torch

from torch import nn

from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5Model,
)

from vllm.model_executor.models.qwen3_next import (
    Qwen3NextModel,
)

from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
)

try:
    from vllm.model_executor.layers.fused_moe import (
        fused_moe_make_expert_params_mapping,
    )
except ImportError:  # vLLM builds without the legacy fused-MoE helper
    fused_moe_make_expert_params_mapping = None


@dataclass(frozen=True)
class PLEShardOverlap:
    """Source and destination slices for one checkpoint embedding shard."""

    source_start: int
    destination_start: int
    row_count: int


def compute_ple_shard_overlap(
    *,
    checkpoint_start: int,
    checkpoint_rows: int,
    tp_start: int,
    tp_end: int,
) -> PLEShardOverlap | None:
    """Compute the overlap of a checkpoint shard and one TP vocabulary range."""

    if checkpoint_start < 0 or checkpoint_rows < 0:
        raise ValueError("checkpoint shard bounds must be non-negative")
    if tp_start < 0 or tp_end < tp_start:
        raise ValueError("invalid TP vocabulary range")
    checkpoint_end = checkpoint_start + checkpoint_rows
    overlap_start = max(checkpoint_start, tp_start)
    overlap_end = min(checkpoint_end, tp_end)
    if overlap_start >= overlap_end:
        return None
    return PLEShardOverlap(
        source_start=overlap_start - checkpoint_start,
        destination_start=overlap_start - tp_start,
        row_count=overlap_end - overlap_start,
    )


def copy_ple_embedding_shard_(
    destination: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    checkpoint_start: int,
    tp_start: int,
    tp_end: int,
) -> int:
    """Copy the overlapping rows of a PLE checkpoint shard into a TP table."""

    if destination.ndim == 0 or loaded_weight.ndim != destination.ndim:
        raise ValueError("destination and loaded weight must have matching ranks")
    if destination.shape[1:] != loaded_weight.shape[1:]:
        raise ValueError(
            "embedding shard dimensions do not match: "
            f"{tuple(destination.shape[1:])} != {tuple(loaded_weight.shape[1:])}"
        )
    if destination.shape[0] < tp_end - tp_start:
        raise ValueError("destination does not cover the requested TP range")
    overlap = compute_ple_shard_overlap(
        checkpoint_start=checkpoint_start,
        checkpoint_rows=loaded_weight.shape[0],
        tp_start=tp_start,
        tp_end=tp_end,
    )
    if overlap is None:
        return 0
    source = loaded_weight.narrow(0, overlap.source_start, overlap.row_count)
    target = destination.narrow(0, overlap.destination_start, overlap.row_count)
    with torch.no_grad():
        target.copy_(source.to(device=target.device, dtype=target.dtype))
    return overlap.row_count


def _get_qwen35_hf_to_vllm_mapper() -> WeightsMapper:
    """Return the native packed-weight mapper when the vLLM API provides it.

    The Qwen3.8 reference was authored against the post-0.24 mapper API.  The
    FlagOS Day0 base image uses vLLM 0.24, where Qwen3Next/Qwen3.5 still packs
    weights in handwritten loaders and therefore exposes no class mapper.
    """

    mapper = getattr(Qwen3_5Model, "hf_to_vllm_mapper", None)
    if mapper is None:
        mapper = getattr(Qwen3NextModel, "hf_to_vllm_mapper", None)
    return mapper if mapper is not None else WeightsMapper()


_QWEN35_HF_TO_VLLM_MAPPER = _get_qwen35_hf_to_vllm_mapper()
_HAS_NATIVE_STACKED_WEIGHTS_MAPPER = hasattr(WeightsMapper(), "orig_to_new_stacked")


_VLLM024_STACKED_WEIGHT_MAPPINGS = (
    (".q_proj", ".qkv_proj", "q"),
    (".k_proj", ".qkv_proj", "k"),
    (".v_proj", ".qkv_proj", "v"),
    (".mlp.gate_proj", ".mlp.gate_up_proj", 0),
    (".mlp.up_proj", ".mlp.gate_up_proj", 1),
    (".shared_expert.gate_proj", ".shared_expert.gate_up_proj", 0),
    (".shared_expert.up_proj", ".shared_expert.gate_up_proj", 1),
    (".in_proj_qkv", ".in_proj_qkvz", (0, 1, 2)),
    (".in_proj_z", ".in_proj_qkvz", 3),
    (".in_proj_b", ".in_proj_ba", 0),
    (".in_proj_a", ".in_proj_ba", 1),
)


def _map_vllm024_stacked_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Attach packed-linear shard metadata missing from vLLM 0.24's mapper."""

    for name, weight in weights:
        for old, new, shard_id in _VLLM024_STACKED_WEIGHT_MAPPINGS:
            if old in name:
                name = name.replace(old, new, 1)
                weight.shard_id = shard_id
                break
        yield name, weight


def _get_vllm024_expert_mappings(
    model: nn.Module,
    num_experts: int,
    num_redundant_experts: int,
) -> (
    tuple[
        list[tuple[str, str, int, str]],
        list[tuple[str, str, int, str]],
    ]
    | None
):
    """Probe for the vLLM 0.24 fused-MoE mapping ABI.

    vLLM 0.24 exposes ``fused_moe_make_expert_params_mapping`` but does not
    teach ``AutoWeightsLoader`` how to turn a Hugging Face fused 3-D
    ``experts.{gate_up_proj,down_proj}`` tensor into the
    ``experts.routed_experts.{w13,w2}_weight`` parameters.  Newer vLLM builds
    carry that knowledge through the native mapper and never enter this
    compatibility path.  Keeping the probe structural avoids coupling the
    plugin to a version string.

    The first mapping handles ordinary one-expert-at-a-time checkpoints.  The
    second mapping deliberately uses ``gate_up_proj`` for both the gate and up
    names and is reduced to generic names for fused 3-D checkpoints, matching
    the loader in vLLM's Qwen3.5 implementation.
    """

    helper = fused_moe_make_expert_params_mapping
    if not callable(helper):
        return None

    try:
        regular_mapping = list(
            helper(
                model,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=num_experts,
                num_redundant_experts=num_redundant_experts,
            )
        )
        fused_base_mapping = list(
            helper(
                model,
                ckpt_gate_proj_name="gate_up_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="gate_up_proj",
                num_experts=1,
            )
        )
    except (AttributeError, TypeError):
        # A future helper with a different ABI must stay on the native path
        # instead of being partially handled by this legacy adapter.
        return None

    fused_mapping: list[tuple[str, str, int, str]] = []
    for param_name, checkpoint_name, _, shard_id in fused_base_mapping:
        parts = checkpoint_name.split(".")
        if len(parts) < 3:
            return None
        fused_mapping.append(
            (
                f"{param_name}weight",
                f"{parts[0]}.{parts[2]}",
                0,
                shard_id,
            )
        )
    if not regular_mapping or not fused_mapping:
        return None
    return regular_mapping, fused_mapping


def _call_vllm024_expert_weight_loader(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    param_name: str,
    shard_id: str,
    expert_id: int,
) -> bool:
    """Call the legacy FusedMoE weight loader with capability probing."""

    weight_loader = getattr(param, "weight_loader", None)
    if weight_loader is None:
        raise TypeError(f"Fused expert parameter {param_name!r} has no weight_loader")
    try:
        loaded = weight_loader(
            param,
            loaded_weight,
            param_name,
            shard_id=shard_id,
            expert_id=expert_id,
            return_success=True,
        )
    except TypeError as exc:
        # Keep support for the older callable form without weakening errors
        # from inside a real loader implementation.
        if "return_success" not in str(exc):
            raise
        weight_loader(
            param,
            loaded_weight,
            param_name,
            shard_id=shard_id,
            expert_id=expert_id,
        )
        return True
    return True if loaded is None else bool(loaded)


def _load_vllm024_fused_expert_weight(
    name: str,
    loaded_weight: torch.Tensor,
    params_dict: dict[str, nn.Parameter],
    fused_mapping: list[tuple[str, str, int, str]],
    num_experts: int,
) -> tuple[bool, set[str]]:
    """Load one fused HF expert tensor using the vLLM 0.24 parameter ABI."""

    if "mlp.experts.gate_up_proj" not in name and "mlp.experts.down_proj" not in name:
        return False, set()
    if loaded_weight.ndim != 3:
        raise ValueError(
            f"Expected a fused 3-D MoE tensor for {name!r}, got "
            f"shape={tuple(loaded_weight.shape)}"
        )

    loaded_params: set[str] = set()
    for param_name, weight_name, _, shard_id in fused_mapping:
        if weight_name not in name:
            continue
        name_mapped = name.replace(weight_name, param_name, 1)
        param = params_dict.get(name_mapped)
        # A pipeline-parallel rank can legitimately not own this layer.
        if param is None:
            return True, loaded_params

        if "gate_up_proj" in name:
            split_weights = loaded_weight.chunk(2, dim=-2)
            shard_weights = (
                ("w1", split_weights[0]),
                ("w3", split_weights[1]),
            )
        else:
            shard_weights = ((shard_id, loaded_weight),)

        loaded_local_expert = False
        for actual_shard_id, shard_weight in shard_weights:
            for expert_id in range(num_experts):
                if _call_vllm024_expert_weight_loader(
                    param,
                    shard_weight[expert_id],
                    name_mapped,
                    actual_shard_id,
                    expert_id,
                ):
                    loaded_local_expert = True
        if loaded_local_expert:
            loaded_params.add(name_mapped)
        return True, loaded_params
    return True, loaded_params


def _load_vllm024_single_expert_weight(
    name: str,
    loaded_weight: torch.Tensor,
    params_dict: dict[str, nn.Parameter],
    expert_mapping: list[tuple[str, str, int, str]],
) -> tuple[bool, set[str]]:
    """Load a non-fused expert tensor when the old ABI supplies one expert."""

    loaded_params: set[str] = set()
    for param_name, weight_name, expert_id, shard_id in expert_mapping:
        if weight_name not in name:
            continue
        name_mapped = name.replace(weight_name, param_name, 1)
        param = params_dict.get(name_mapped)
        if param is None:
            return True, loaded_params
        if _call_vllm024_expert_weight_loader(
            param,
            loaded_weight,
            name_mapped,
            shard_id,
            expert_id,
        ):
            loaded_params.add(name_mapped)
        return True, loaded_params
    return False, loaded_params


class _VLLM024StackedAutoWeightsLoader(AutoWeightsLoader):
    """Teach the vLLM 0.24 auto-loader to forward packed shard identifiers."""

    def _load_param(
        self,
        base_prefix: str,
        param: nn.Parameter,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        for weight_name, weight_data in weights:
            shard_id = getattr(weight_data, "shard_id", None)
            if shard_id is None:
                yield from super()._load_param(
                    base_prefix, param, [(weight_name, weight_data)]
                )
                continue

            weight_qualname = self._get_qualname(base_prefix, weight_name)
            if self._can_skip(weight_qualname):
                continue
            if weight_name != "":
                if self._can_ignore_unexpected(weight_qualname):
                    continue
                raise ValueError(
                    f"Attempted to load nested weight {weight_qualname!r} "
                    f"into a single parameter {base_prefix!r}"
                )
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is None:
                raise TypeError(
                    f"Packed parameter {weight_qualname!r} has no weight_loader"
                )
            weight_loader(param, weight_data, shard_id)
            yield weight_qualname


def without_modelopt_fp4(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    """Return ``None`` for weights excluded from Qwen3.8-Flash-Next ModelOpt-FP4."""

    if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
        return None
    return quant_config


def _remap_qsa_cache_scale_name(
    name: str,
    qsa_layer_ids: frozenset[int],
) -> str:
    """Map serialized main-cache scales onto the merged QSA owner.

    Regular attention keeps cache scales below its ``attn`` child. QSA owns
    that cache directly, so only QSA layers need the final path component
    moved to the owner's persistent ``_k_scale``/``_v_scale`` buffers.
    """

    scale_suffixes = {
        "k_proj.k_scale": "_k_scale",
        "k_proj.output_scale": "_k_scale",
        "attn.k_scale": "_k_scale",
        "attn._k_scale": "_k_scale",
        "k_scale": "_k_scale",
        "_k_scale": "_k_scale",
        "v_proj.v_scale": "_v_scale",
        "v_proj.output_scale": "_v_scale",
        "attn.v_scale": "_v_scale",
        "attn._v_scale": "_v_scale",
        "v_scale": "_v_scale",
        "_v_scale": "_v_scale",
    }
    for layer_id in qsa_layer_ids:
        marker = f"layers.{layer_id}.self_attn."
        marker_start = name.find(marker)
        if marker_start < 0 or (marker_start > 0 and name[marker_start - 1] != "."):
            continue
        suffix = name[marker_start + len(marker) :]
        mapped_suffix = scale_suffixes.get(suffix)
        if mapped_suffix is not None:
            return f"{name[: marker_start + len(marker)]}{mapped_suffix}"
    return name


_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES = [
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    "_weight_scale",
    "_input_scale",
]
