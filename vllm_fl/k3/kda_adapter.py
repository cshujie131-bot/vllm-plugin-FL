# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KDA model integration using the native construction hook."""

from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention

from vllm_fl.ops.kimi_k3_metax.kda import chunk_kda_with_fused_gate


class K3DeltaAttention(KimiK3DeltaAttention):
    @staticmethod
    def _get_chunk_prefill_backend():
        return chunk_kda_with_fused_gate
