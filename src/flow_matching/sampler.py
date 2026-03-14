"""ODE sampler: generate new data by integrating the learned velocity field."""

import torch
import torch.nn as nn


def ode_sample(
    model: nn.Module,
    n_samples: int,
    data_shape: tuple,
    n_steps: int = 100,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate samples by solving dx/dt = u_theta(x, t) from t=0 to t=1.

    Uses simple Euler integration. For better quality, you can switch to
    torchdiffeq.odeint or a higher-order method.

    Args:
        model: trained velocity network u_theta(x, t)
        n_samples: number of samples to generate
        data_shape: shape of one sample, e.g., (2,) for 2D or (1, 28, 28) for images
        n_steps: number of Euler steps
        device: device

    Returns:
        x: (n_samples, *data_shape) — generated samples at t=1
    """
    # TODO:
    # 1. x = torch.randn(n_samples, *data_shape)  (sample from p0 = N(0, I))
    # 2. dt = 1.0 / n_steps
    # 3. For k in range(n_steps):
    #        t = k * dt
    #        velocity = model(x, t * torch.ones(n_samples, device=device))
    #        x = x + velocity * dt
    # 4. Return x
    raise NotImplementedError


def ode_sample_hybrid(
    model: nn.Module,
    data: torch.Tensor,
    x0: torch.Tensor,
    tau: float,
    n_steps: int = 100,
    device: str = "cpu",
) -> torch.Tensor:
    """Hybrid sampling for Figure 3: follow û★ for t in [0, tau], then u_theta for t in [tau, 1].

    Args:
        model: trained velocity network
        data: (n, d) — training set (needed for closed-form û★)
        x0: (batch, d) — initial noise
        tau: switching time
        n_steps: total Euler steps

    Returns:
        x: (batch, d) — generated samples
    """
    # TODO:
    # 1. Euler integration from t=0 to t=tau using optimal_velocity(x, data, t)
    # 2. Continue Euler integration from t=tau to t=1 using model(x, t)
    # 3. Return x
    raise NotImplementedError
