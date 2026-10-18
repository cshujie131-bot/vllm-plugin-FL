"""
FlagGems deepseekV4 operator implementations.
"""

from __future__ import annotations

import torch


from ..deepseek_v4_ops import (
    gather_k_cache_flaggems,
    fused_indexer_q_rope_flaggems,
    fused_inv_rope_flaggems,
    gather_bf16_kv_from_pages_flaggems,
    bf16_mqa_logits_flaggems,
)
