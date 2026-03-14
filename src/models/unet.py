"""Small UNet for image-scale flow matching, trainable on a free Colab T4."""

import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for the time variable."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) time values in [0, 1]
        Returns:
            (batch, dim) embedding
        """
        # TODO: standard sinusoidal embedding (same idea as in Transformers)
        # half_dim = self.dim // 2
        # frequencies = exp(-log(10000) * arange(half_dim) / half_dim)
        # args = t[:, None] * frequencies[None, :]
        # return cat([sin(args), cos(args)], dim=-1)
        raise NotImplementedError


class ResBlock(nn.Module):
    """Residual block with time conditioning: Conv -> GroupNorm -> SiLU -> Conv + time projection."""

    def __init__(self, channels: int, time_emb_dim: int):
        super().__init__()
        # TODO:
        # self.conv1 = ...
        # self.conv2 = ...
        # self.norm1 = GroupNorm(...)
        # self.norm2 = GroupNorm(...)
        # self.time_proj = nn.Linear(time_emb_dim, channels)
        raise NotImplementedError

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # TODO: h = conv1(norm1(x)) + time_proj(t_emb), then conv2(norm2(h)) + x
        raise NotImplementedError


class SmallUNet(nn.Module):
    """Minimal UNet (2-3 levels) for 28x28 or 32x32 single-channel images.

    ~1-2M parameters, enough to learn on MNIST/FMNIST on a T4 GPU.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32, time_emb_dim: int = 64):
        super().__init__()
        # TODO: build a small UNet
        # Encoder: 2-3 downsampling blocks (ResBlock + downsample)
        # Bottleneck: ResBlock
        # Decoder: 2-3 upsampling blocks (ResBlock + upsample + skip connections)
        # Final conv to in_channels
        raise NotImplementedError

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, C, H, W) — noisy image x_t
            t: (batch,) — time in [0, 1]
        Returns:
            velocity: (batch, C, H, W)
        """
        raise NotImplementedError
