# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded K3 vision execution through native component factories."""

import torch

from vllm.model_executor.models.kimi_k25_vit import (
    MLP2,
    MoonViT3dEncoder,
    MoonViT3dPretrainedModel,
    MoonViTEncoderLayer,
    Rope2DPosEmbRepeated,
    _apply_rope_input_validation,
)


def _apply_rope_block(xq, xk, freqs_cis):
    freqs_cis = freqs_cis.unsqueeze(-2)
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_rope(xq, xk, freqs_cis):
    """Preserve the original complex64 rotation while bounding temporary rows."""
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)
    if xq.ndim != 3 or xq.shape[0] < 16384:
        return _apply_rope_block(xq, xk, freqs_cis)
    out_q, out_k = torch.empty_like(xq), torch.empty_like(xk)
    freqs_real = torch.view_as_real(freqs_cis)
    for start in range(0, xq.shape[0], 8192):
        stop = min(start + 8192, xq.shape[0])
        q, k = _apply_rope_block(
            xq[start:stop],
            xk[start:stop],
            torch.view_as_complex(freqs_real[start:stop]),
        )
        out_q[start:stop].copy_(q)
        out_k[start:stop].copy_(k)
        del q, k
    return out_q, out_k


class K3VisionRope(Rope2DPosEmbRepeated):
    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid."""
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = (
            torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)
        )  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        # Move the real/imaginary pairs; the complex RoPE values are unchanged.
        pairs = torch.cat(
            [
                torch.view_as_real(x_cis).unsqueeze(-2),
                torch.view_as_real(y_cis).unsqueeze(-2),
            ],
            dim=-2,
        )
        freqs_cis = torch.view_as_complex(pairs)
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(
        self, grid_thws: torch.Tensor | list[list[int]], device: torch.device
    ) -> torch.Tensor:
        """
        Args:
            grid_thws (torch.Tensor): grid time, height and width

        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        if not hasattr(self, "freqs_cis"):
            self.register_buffer(
                "freqs_cis", self._precompute_freqs_cis(device), persistent=False
            )

        shapes = grid_thws if isinstance(grid_thws, list) else grid_thws.tolist()
        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width for t, h, w in shapes
        ), (
            shapes,
            self.max_height,
            self.max_width,
        )
        pairs = torch.view_as_real(self.freqs_cis)
        selected = torch.cat(
            [
                pairs[:h, :w].reshape(-1, self.dim // 2, 2).repeat(t, 1, 1)
                for t, h, w in shapes
            ],
            dim=0,
        )
        freqs_cis = torch.view_as_complex(selected)
        return freqs_cis


class K3VisionMLP(MLP2):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_data_parallel or x.ndim != 2 or x.shape[0] < 16384:
            return super().forward(x)
        output = torch.empty(
            (x.shape[0], self.fc1.output_size), dtype=x.dtype, device=x.device
        )
        for start in range(0, x.shape[0], 8192):
            stop = min(start + 8192, x.shape[0])
            block = super().forward(x[start:stop])
            output[start:stop].copy_(block)
            del block
        return output


class K3VisionEncoderLayer(MoonViTEncoderLayer):
    mlp_cls = K3VisionMLP
    rotary_embedding = staticmethod(apply_rope)


class K3VisionEncoder(MoonViT3dEncoder):
    rope_cls = K3VisionRope
    encoder_layer_cls = K3VisionEncoderLayer


class K3VisionModel(MoonViT3dPretrainedModel):
    encoder_cls = K3VisionEncoder
