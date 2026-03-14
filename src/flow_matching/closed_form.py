"""Closed-form optimal velocity field û★ (Proposition 1 of the paper)."""

import torch


def softmax_weights(x_t: torch.Tensor, data: torch.Tensor, t: float) -> torch.Tensor:
    """Compute the softmax weights λ_i(x_t, t) from Proposition 1.

    λ(x, t) = softmax( -||x - t * x^(j)||^2 / (2*(1-t)^2) )  for j = 1..n

    Args:
        x_t: (batch, d) — current positions along the flow
        data: (n, d) — all training samples x^(1), ..., x^(n)
        t: scalar time in [0, 1)

    Returns:
        weights: (batch, n) — softmax weights, each row sums to 1
    """
    # TODO:
    # 1. Compute logits: -||x_t[i] - t * data[j]||^2 / (2*(1-t)^2) for all (i, j)
    #    Hint: use broadcasting. x_t is (B, d), data is (n, d).
    #    diffs = x_t[:, None, :] - t * data[None, :, :]  -> (B, n, d)
    #    logits = -||diffs||^2 / (2*(1-t)^2)              -> (B, n)
    # 2. Return softmax over dimension 1
    # Note: use the log-sum-exp trick for numerical stability (torch.softmax does this)
    raise NotImplementedError


def optimal_velocity(x_t: torch.Tensor, data: torch.Tensor, t: float) -> torch.Tensor:
    """Compute û★(x_t, t) = Σ_i λ_i(x_t, t) * (x^(i) - x_t) / (1 - t).

    This is equation (6) of the paper.

    Args:
        x_t: (batch, d) — current positions
        data: (n, d) — training data
        t: scalar time in [0, 1)

    Returns:
        velocity: (batch, d)
    """
    # TODO:
    # 1. Compute weights = softmax_weights(x_t, data, t)  -> (B, n)
    # 2. Compute directions = (data[None, :, :] - x_t[:, None, :]) / (1 - t)  -> (B, n, d)
    # 3. Return weighted sum: (weights[:, :, None] * directions).sum(dim=1)  -> (B, d)
    raise NotImplementedError


def cosine_sim_u_star_vs_ucond(
    x0: torch.Tensor, x1: torch.Tensor, data: torch.Tensor, t: float
) -> torch.Tensor:
    """Cosine similarity between û★(x_t, t) and u^cond(x_t, t) = x1 - x0.

    This is the key measurement for Figure 1 of the paper.

    Args:
        x0: (batch, d) — noise samples from p0
        x1: (batch, d) — data samples (the ones used to build x_t)
        data: (n, d) — full training set
        t: scalar time

    Returns:
        similarities: (batch,) — cosine similarities in [-1, 1]
    """
    # TODO:
    # 1. x_t = (1 - t) * x0 + t * x1
    # 2. u_star = optimal_velocity(x_t, data, t)
    # 3. u_cond = x1 - x0   (this equals (x1 - x_t) / (1 - t) for linear interpolation)
    # 4. Return cosine similarity between u_star and u_cond along dim=-1
    raise NotImplementedError
