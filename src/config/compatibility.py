"""
Aircraft-gate compatibility and preference configuration.
"""
import numpy as np
from attrs import frozen, field


@frozen
class CompatibilityConfig:
    """
    Aircraft-gate compatibility and preferences.

    Distinguishes between hard constraints (physical compatibility)
    and soft preferences (operational efficiency).
    """

    compatibility_matrix: np.ndarray = field()  # Shape: (num_types, num_gates), binary
    preference_matrix: np.ndarray = field()  # Shape: (num_types, num_gates), non-negative

    @compatibility_matrix.validator
    def _validate_compatibility_binary(self, attribute, value):
        """Ensure compatibility matrix is binary."""
        if not np.all(np.isin(value, [0, 1])):
            raise ValueError("Compatibility matrix must be binary (0 or 1)")

    @preference_matrix.validator
    def _validate_preference_nonnegative(self, attribute, value):
        """Ensure preference scores are non-negative."""
        if np.any(value < 0):
            raise ValueError("Preference matrix must be non-negative")

    @compatibility_matrix.validator
    def _validate_shapes_match(self, attribute, value):
        """Ensure compatibility and preference matrices have same shape."""
        if value.shape != self.preference_matrix.shape:
            raise ValueError(
                f"Compatibility {value.shape} and preference {self.preference_matrix.shape} "
                f"matrices must have same shape"
            )

    def is_compatible(self, aircraft_type_idx: int, gate_idx: int) -> bool:
        """
        Check if aircraft type can physically use gate.

        Args:
            aircraft_type_idx: Aircraft type index
            gate_idx: Gate index

        Returns:
            True if compatible (hard constraint satisfied)
        """
        return bool(self.compatibility_matrix[aircraft_type_idx, gate_idx])

    def get_preference(self, aircraft_type_idx: int, gate_idx: int) -> float:
        """
        Get preference score for assignment.

        Args:
            aircraft_type_idx: Aircraft type index
            gate_idx: Gate index

        Returns:
            Preference score (0 = no preference, higher = more preferred)
        """
        return float(self.preference_matrix[aircraft_type_idx, gate_idx])

    def get_compatible_gates(self, aircraft_type_idx: int) -> np.ndarray:
        """
        Get all gates compatible with aircraft type.

        Args:
            aircraft_type_idx: Aircraft type index

        Returns:
            Array of compatible gate indices
        """
        return np.where(self.compatibility_matrix[aircraft_type_idx, :] == 1)[0]