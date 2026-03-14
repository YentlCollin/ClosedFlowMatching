"""Evaluation metrics for flow matching experiments."""

import torch
import torch.nn.functional as F


def cosine_similarity_batch(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity between rows of u and v.

    Args:
        u: (batch, d)
        v: (batch, d)

    Returns:
        similarities: (batch,)
    """
    # TODO: use F.cosine_similarity(u, v, dim=-1)
    raise NotImplementedError


def nearest_neighbor_distance(
    generated: torch.Tensor, train_data: torch.Tensor
) -> torch.Tensor:
    """For each generated sample, compute L2 distance to its nearest neighbor in train_data.

    This is the memorization metric from Figure 2 (right panel).

    Args:
        generated: (m, d) — generated samples
        train_data: (n, d) — training set

    Returns:
        distances: (m,) — distance to nearest training point
    """
    # TODO:
    # Compute pairwise distances (m, n) using torch.cdist
    # Return min over training set dimension
    raise NotImplementedError


def velocity_approximation_error(
    model: torch.nn.Module,
    data: torch.Tensor,
    t_values: list[float],
    n_eval: int = 256,
    device: str = "cpu",
) -> dict[float, float]:
    """Measure ||u_theta(x_t, t) - û★(x_t, t)||^2 for multiple time values.

    This is the key metric for Figure 2 (left panel).

    Args:
        model: trained velocity network
        data: (n, d) — training set
        t_values: list of time values to evaluate at
        n_eval: number of (x0, x1) pairs to average over

    Returns:
        dict mapping t -> average squared error
    """
    # TODO:
    # For each t in t_values:
    #   1. Sample n_eval pairs (x0 ~ N(0,I), x1 ~ data)
    #   2. x_t = (1-t)*x0 + t*x1
    #   3. u_star = optimal_velocity(x_t, data, t)  (from closed_form.py)
    #   4. u_theta = model(x_t, t)
    #   5. error = mean ||u_theta - u_star||^2
    raise NotImplementedError
