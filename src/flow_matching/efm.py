"""Empirical Flow Matching — Algorithm 2 of the paper."""

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import trange


def compute_efm_target(
    x_t: torch.Tensor, x1: torch.Tensor, data: torch.Tensor, t: torch.Tensor, M: int
) -> torch.Tensor:
    """Compute the EFM target û★_M(x_t, t) using M samples.

    Key trick: b^(1) = x1 (the point that generated x_t), and b^(2)..b^(M) are
    sampled uniformly from the training set.

    This is equation (8) of the paper.

    Args:
        x_t: (batch, d) — interpolated points
        x1: (batch, d) — the data points used to construct x_t
        data: (n, d) — full training set
        t: (batch, 1) — time values
        M: number of samples for the Monte Carlo estimate

    Returns:
        target: (batch, d) — the EFM velocity target
    """
    # TODO:
    # 1. b^(1) = x1
    # 2. Sample M-1 random indices from data -> b^(2)..b^(M)
    # 3. Stack all M samples: b of shape (batch, M, d)
    # 4. Compute softmax logits: -||x_t - t * b^(j)||^2 / (2*(1-t)^2) for j=1..M
    # 5. weights = softmax(logits)  -> (batch, M)
    # 6. directions = (b - x_t.unsqueeze(1)) / (1 - t).unsqueeze(-1)  -> (batch, M, d)
    # 7. target = (weights.unsqueeze(-1) * directions).sum(dim=1)  -> (batch, d)
    raise NotImplementedError


class EFMTrainer:
    """Trains a velocity network u_theta with the Empirical Flow Matching loss.

    Same as CFM, but the target is û★_M instead of u^cond = x1 - x0.
    """

    def __init__(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        M: int = 128,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        """
        Args:
            model: velocity network
            train_data: (n, d) — full training set, kept on device for EFM target computation
            M: number of samples for Monte Carlo estimate of û★
            lr: learning rate
        """
        self.model = model.to(device)
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.device = device
        self.train_data = train_data.to(device)
        self.M = M

    def train_step(self, x1: torch.Tensor) -> float:
        """One training step with EFM loss.

        Args:
            x1: (batch, d) — batch of training data

        Returns:
            loss value (float)
        """
        # TODO:
        # 1. Sample t ~ Uniform([0, 1])
        # 2. Sample x0 ~ N(0, I)
        # 3. x_t = (1 - t) * x0 + t * x1
        # 4. target = compute_efm_target(x_t, x1, self.train_data, t, self.M)
        # 5. prediction = self.model(x_t, t)
        # 6. Loss = MSE(prediction, target)
        # 7. Backward + step
        raise NotImplementedError

    def train(self, dataset, n_steps: int = 10000, batch_size: int = 256, log_every: int = 1000):
        """Full training loop."""
        # TODO: same structure as CFMTrainer.train
        raise NotImplementedError
