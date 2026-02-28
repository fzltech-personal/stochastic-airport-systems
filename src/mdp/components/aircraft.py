"""
Aircraft type configuration.
"""
import numpy as np
from attrs import frozen, field, validators


@frozen
class AircraftTypeConfig:
    """Configuration for a single aircraft type."""

    name: str
    base_service_mean: float = field(validator=validators.gt(0))
    base_service_std: float = field(validator=validators.ge(0))
    probability: float = field(validator=[validators.ge(0), validators.le(1)])
    description: str = ""
    typical_routes: str = ""

    def sample_base_service_time(self, rng: np.random.Generator) -> float:
        """
        Sample service time from configured distribution.

        Args:
            rng: Numpy random generator

        Returns:
            Sampled base service time (before taxiing penalty)
        """
        return rng.normal(self.base_service_mean, self.base_service_std)

    def __str__(self) -> str:
        return (f"AircraftType(name={self.name}, "
                f"service={self.base_service_mean}±{self.base_service_std}, "
                f"prob={self.probability:.2f})")