"""
Arrival time noise model configuration.
"""
from typing import Dict, Literal, Optional
import numpy as np
from attrs import frozen, field


@frozen
class NoiseModelConfig:
    """
    Configuration for stochastic arrival time perturbations.

    Supports multiple noise distributions to model different
    disruption scenarios (e.g., normal weather delays, uniform jitter).
    """

    distribution: Literal['normal', 'uniform', 'erlang'] = 'normal'
    params: Dict[str, float] = field(factory=dict)

    def sample(self, rng: np.random.Generator, size: Optional[int] = None) -> np.ndarray:
        """
        Sample noise from configured distribution.

        Args:
            rng: Numpy random generator
            size: Number of samples (None for single sample)

        Returns:
            Noise values (delays/advances in time units)
        """
        if self.distribution == 'normal':
            mean = self.params.get('mean', 0)
            std = self.params.get('std', 1)
            return rng.normal(mean, std, size=size)

        elif self.distribution == 'uniform':
            low = self.params.get('low', -5)
            high = self.params.get('high', 5)
            return rng.uniform(low, high, size=size)

        elif self.distribution == 'erlang':
            # Erlang models cascading delays (always positive)
            # Shifted to have approximately zero mean
            shape = self.params.get('shape', 2)
            scale = self.params.get('scale', 1)
            samples = rng.gamma(shape, scale, size=size)
            return samples - shape * scale  # Center around 0

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def __str__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"NoiseModel({self.distribution}, {param_str})"