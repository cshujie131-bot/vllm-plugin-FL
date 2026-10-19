from vllm.model_executor.layers.activation import SituAndMul
from vllm_fl.ops.kimi_k3_metax.activations import _can_fuse_situ_input, situ_out


class K3SituAndMul(SituAndMul):
    """K3's explicit OOT entry; no global platform override or extra registration."""

    def forward_oot(self, x):
        if _can_fuse_situ_input(x, self.beta, self.linear_beta):
            # The inherited CUDA-alike entry calls the one registered situ_out.
            return self.forward_cuda(x)
        # Preserve the actual old OOT fallback, without adding an out-copy.
        return self.forward_native(x)
