# Copyright (c) 2026 BAAI. All rights reserved.
"""Runtime registration for the Qwen3.8-Flash-Next / Qwen4Exp basic integration model.

The checkpoint keeps its original ``qwen4_exp`` model types and architecture
name.  This module maps those public names to the plugin-owned implementation
without modifying either the checkpoint or the installed vLLM package.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vllm.model_executor.models.config import Qwen3_5ForConditionalGenerationConfig

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = logging.getLogger(__name__)


def _strip_mrope(model_config: ModelConfig) -> None:
    configs = {
        id(config): config
        for config in (
            getattr(model_config, "hf_config", None),
            model_config.hf_text_config,
        )
        if config is not None
    }
    for config in configs.values():
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_parameters.pop("mrope_section", None)
            rope_parameters.pop("mrope_interleaved", None)


class Qwen3_8FlashNextForConditionalGenerationConfig(
    Qwen3_5ForConditionalGenerationConfig
):
    """Apply the hybrid-cache and unsupported-feature contract."""

    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
        text_config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config

        # vLLM 0.24's Qwen3.5 verifier is intentionally empty. Preserve the
        # checkpoint's FP32 recurrent-state contract here.
        mamba_ssm_dtype = getattr(text_config, "mamba_ssm_dtype", None)
        if cache_config.mamba_ssm_cache_dtype == "auto":
            if mamba_ssm_dtype is not None:
                cache_config.mamba_ssm_cache_dtype = mamba_ssm_dtype
        elif (
            mamba_ssm_dtype is not None
            and cache_config.mamba_ssm_cache_dtype != mamba_ssm_dtype
        ):
            logger.warning(
                "Qwen4Exp config requests mamba_ssm_dtype=%s, but the runtime "
                "override is %s; preserving the explicit runtime value.",
                mamba_ssm_dtype,
                cache_config.mamba_ssm_cache_dtype,
            )

        if int(text_config.hc_count) <= 1:
            raise ValueError("Qwen4Exp requires hc_count > 1")

        parallel_config = vllm_config.parallel_config
        uses_ple_or_qsa = bool(text_config.ple_layer_ids) or (
            getattr(text_config, "indexer_n_heads", None) is not None
        )
        if uses_ple_or_qsa and (
            parallel_config.enable_dbo or parallel_config.ubatch_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp PLE/QSA does not support dual-batch overlap or "
                "microbatching in the basic integration path"
            )
        if (
            bool(text_config.ple_layer_ids)
            and parallel_config.pipeline_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp PLE requires pipeline_parallel_size=1 because raw "
                "token n-gram context is not broadcast between PP stages"
            )

        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is not None and multimodal_config.language_model_only:
            _strip_mrope(vllm_config.model_config)

        import torch
        from vllm import envs

        if envs.VLLM_USE_V2_MODEL_RUNNER:
            raise NotImplementedError("Qwen3.8 basic integration requires ModelRunnerFL/MRV1")
        if (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.num_speculative_tokens != 1
        ):
            raise NotImplementedError("Qwen3.8 basic integration currently requires MTP K1")
        if (
            vllm_config.speculative_config is not None
            and not vllm_config.speculative_config.disable_padded_drafter_batch
        ):
            raise NotImplementedError(
                "Qwen3.8 vLLM 0.24 MTP requires disable_padded_drafter_batch=True "
                "for chunked-prefill token alignment"
            )
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("Qwen3.8 basic integration requires BF16 weights")
        if cache_config.mamba_ssm_cache_dtype != "float32":
            raise ValueError("Qwen3.8 recurrent state requires float32")
        if cache_config.mamba_cache_mode != "none":
            raise NotImplementedError("Qwen3.8 requires mamba_cache_mode=none")
        if cache_config.enable_prefix_caching:
            raise NotImplementedError("Qwen3.8 prefix caching is not supported")
        if vllm_config.scheduler_config.async_scheduling:
            raise NotImplementedError(
                "CPU N-gram history requires synchronous scheduling"
            )
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError("Qwen3.8 basic integration requires PP1")
        if multimodal_config is not None and not multimodal_config.language_model_only:
            raise NotImplementedError(
                "Qwen3.8 basic integration currently supports text-only serving"
            )


class Qwen3_8FlashNextForCausalLMConfig(Qwen3_8FlashNextForConditionalGenerationConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        Qwen3_8FlashNextForConditionalGenerationConfig.verify_and_update_config(
            vllm_config
        )
        _strip_mrope(vllm_config.model_config)


_ARCHITECTURES = {
    "Qwen3_8FlashNextForCausalLM": (
        "Qwen3_8FlashNextForCausalLM",
        Qwen3_8FlashNextForCausalLMConfig,
    ),
    "Qwen3_8FlashNextForConditionalGeneration": (
        "Qwen3_8FlashNextForConditionalGeneration",
        Qwen3_8FlashNextForConditionalGenerationConfig,
    ),
    "Qwen4ExpForCausalLM": (
        "Qwen4ExpForCausalLM",
        Qwen3_8FlashNextForCausalLMConfig,
    ),
    "Qwen4ExpForConditionalGeneration": (
        "Qwen4ExpForConditionalGeneration",
        Qwen3_8FlashNextForConditionalGenerationConfig,
    ),
}


def _register_compilation_boundaries() -> None:
    from vllm.config.compilation import CompilationConfig

    for op in (
        "vllm::qwen3_8_flash_next_ple_short_conv",
        "vllm::qwen3_8_flash_next_qsa_with_output",
    ):
        if op not in CompilationConfig._attention_ops:
            CompilationConfig._attention_ops.append(op)


def apply_qwen3_8_flash_next_patches() -> bool:
    """Install config/model registrations and opaque compilation boundaries."""
    from vllm.model_executor.models import (
        config as model_config,
        registry as model_registry,
    )
    from vllm.transformers_utils import config as transformers_config

    from vllm_fl.models.qwen3_8_flash_next.config import (
        Qwen3_8FlashNextConfig,
        Qwen3_8FlashNextTextConfig,
    )

    config_registry = transformers_config._CONFIG_REGISTRY
    config_registry.setdefault("qwen3_8_flash_next", Qwen3_8FlashNextConfig)
    config_registry.setdefault("qwen3_8_flash_next_text", Qwen3_8FlashNextTextConfig)
    # Reuse the concrete base classes for checkpoint aliases. Transformers 5
    # serializes inherited composite sub-configs back to dictionaries when an
    # alias subclass changes ``sub_configs``; the base classes preserve typed
    # text/vision configs while retaining the checkpoint's instance
    # ``model_type`` values.
    config_registry.setdefault("qwen4_exp", Qwen3_8FlashNextConfig)
    config_registry.setdefault("qwen4_exp_text", Qwen3_8FlashNextTextConfig)

    module = "vllm_fl.models.qwen3_8_flash_next.model"
    supported_architectures = set(model_registry.ModelRegistry.get_supported_archs())
    for architecture, (class_name, verifier) in _ARCHITECTURES.items():
        model_config.MODELS_CONFIG_MAP.setdefault(architecture, verifier)
        if architecture not in supported_architectures:
            model_registry.ModelRegistry.register_model(
                architecture, f"{module}:{class_name}"
            )
            supported_architectures.add(architecture)

    from .mtp import install_qwen38_mtp_compat

    install_qwen38_mtp_compat()
    _register_compilation_boundaries()
    logger.info(
        "Installed Qwen3.8-Flash-Next / Qwen4Exp basic integration runtime support"
    )
    return True


__all__ = [
    "Qwen3_8FlashNextForCausalLMConfig",
    "Qwen3_8FlashNextForConditionalGenerationConfig",
    "apply_qwen3_8_flash_next_patches",
]
