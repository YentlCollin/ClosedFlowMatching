"""Simple MLP velocity network for 2D toy experiments."""

import torch
import torch.nn as nn


class VelocityMLP(nn.Module):
    """MLP that predicts u_theta(x_t, t) for 2D data.

    Input: concatenation of x_t (dim d) and t (dim 1) -> output: velocity (dim d).
    Architecture: several hidden layers with ReLU (or SiLU), no fancy stuff.
    """

    def __init__(self, data_dim: int = 2, hidden_dim: int = 128, n_layers: int = 4):
        super().__init__()
        # TODO: build a sequential MLP
        # Input size = data_dim + 1 (for time)
        # n_layers hidden layers of size hidden_dim with activation
        # Output size = data_dim
        raise NotImplementedError

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, data_dim) — noisy sample x_t
            t: (batch,) or (batch, 1) — time

        Returns:
            velocity: (batch, data_dim)
        """
        # TODO: concatenate x and t, pass through network
        raise NotImplementedError
