"""
Airport topology configuration.
"""
import numpy as np
from attrs import frozen, field, validators


@frozen
class AirportTopologyConfig:
    """
    Airport physical layout configuration.

    Defines the number of gates and runways, and the taxiing time
    matrix representing spatial heterogeneity.
    """

    num_gates: int = field(validator=validators.gt(0))
    num_runways: int = field(validator=validators.gt(0))
    delta_matrix: np.ndarray = field()  # Shape: (num_runways, num_gates)

    @delta_matrix.validator
    def _validate_delta_shape(self, attribute, value):
        """Ensure delta matrix has correct dimensions."""
        expected_shape = (self.num_runways, self.num_gates)
        if value.shape != expected_shape:
            raise ValueError(
                f"delta_matrix shape {value.shape} != expected {expected_shape}"
            )

    @delta_matrix.validator
    def _validate_delta_nonnegative(self, attribute, value):
        """Ensure all taxiing times are non-negative."""
        if np.any(value < 0):
            raise ValueError("All taxiing times must be non-negative")

    def get_taxiing_time(self, runway_idx: int, gate_idx: int) -> float:
        """
        Get taxiing time from runway to gate.

        Args:
            runway_idx: Runway index (0-indexed)
            gate_idx: Gate index (0-indexed)

        Returns:
            Taxiing time in minutes
        """
        return float(self.delta_matrix[runway_idx, gate_idx])

    def __str__(self) -> str:
        return (f"Airport(gates={self.num_gates}, runways={self.num_runways}, "
                f"max_taxi={self.delta_matrix.max():.1f}min)")