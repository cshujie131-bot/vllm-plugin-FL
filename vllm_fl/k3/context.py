"""Resolve execution policy at model/worker construction, never in GPU kernels."""

from vllm.config import get_current_vllm_config
from vllm.platforms import current_platform
from .configuration import K3ExecutionConfig


def execution_config(config=None):
    if config is None:
        config = get_current_vllm_config()
    return K3ExecutionConfig.from_vllm(
        config, vendor=getattr(current_platform, "vendor_name", None)
    )
