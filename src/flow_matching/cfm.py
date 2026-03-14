"""Vanilla Conditional Flow Matching — Algorithm 1 of the paper."""

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import trange


class CFMTrainer:
    """Trains a velocity network u_theta with the standard CFM loss.

    Loss: E_{t, x0, x1} || u_theta(x_t, t) - (x1 - x0) ||^2
    where x_t = (1-t)*x0 + t*x1.
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3, device: str = "cpu"):
        self.model = model.to(device)
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.device = device

    def train_step(self, x1: torch.Tensor) -> float:
        """One training step.

        Args:
            x1: (batch, d) or (batch, C, H, W) — batch of training data

        Returns:
            loss value (float)
        """
        # TODO:
        # 1. Sample t ~ Uniform([0, 1]), shape (batch, 1) or (batch,)
        # 2. Sample x0 ~ N(0, I), same shape as x1
        # 3. Compute x_t = (1 - t) * x0 + t * x1
        # 4. Target velocity = x1 - x0
        # 5. Predicted velocity = self.model(x_t, t)
        # 6. Loss = MSE between prediction and target
        # 7. Backward + optimizer step
        # 8. Return loss.item()
        raise NotImplementedError

    def train(self, dataset, n_steps: int = 10000, batch_size: int = 256, log_every: int = 1000):
        """Full training loop.

        Args:
            dataset: object with a .sample(batch_size) method (toy) or a DataLoader (images)
            n_steps: number of gradient steps
            batch_size: batch size (used for toy datasets)
            log_every: print loss every this many steps
        """
        # TODO: training loop
        raise NotImplementedError
