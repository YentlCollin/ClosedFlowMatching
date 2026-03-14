"""Toy 2D datasets for flow matching experiments."""

import torch
from sklearn.datasets import make_moons


class ToyDataset:
    """Generates samples from toy 2D distributions.

    Supported distributions: "moons", "rings", "gaussian_mixture", "pinwheel".
    """

    def __init__(self, name: str, n_samples: int = 2000, seed: int = 42):
        self.name = name
        self.n_samples = n_samples
        self.seed = seed
        self.data = self._generate()

    def _generate(self) -> torch.Tensor:
        """Generate the dataset. Returns tensor of shape (n_samples, 2)."""
        torch.manual_seed(self.seed)
        if self.name == "moons":
            return self._make_moons()
        elif self.name == "rings":
            return self._make_rings()
        elif self.name == "gaussian_mixture":
            return self._make_gaussian_mixture()
        else:
            raise ValueError(f"Unknown dataset: {self.name}")

    def _make_moons(self) -> torch.Tensor:
        # TODO: use sklearn make_moons, normalize, return as tensor
        raise NotImplementedError

    def _make_rings(self) -> torch.Tensor:
        # TODO: concentric rings with some noise
        raise NotImplementedError

    def _make_gaussian_mixture(self, n_components: int = 8) -> torch.Tensor:
        # TODO: equally spaced Gaussian centers on a circle
        raise NotImplementedError

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample a random batch from the dataset (with replacement)."""
        # TODO: random indices into self.data
        raise NotImplementedError
