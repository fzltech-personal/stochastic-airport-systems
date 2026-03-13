"""
Complete scenario configuration (aggregates all components).
"""
from typing import Dict, List
import numpy as np
from attrs import frozen, field

from .airport import AirportTopologyConfig
from .compatibility import CompatibilityConfig
from .noise import NoiseModelConfig
from .reward import RewardConfig
from .time import TimeConfig
from .schedule import ScheduleConfig
from ..mdp.components.aircraft import AircraftTypeConfig


@frozen
class ScenarioConfig:
    """
    Complete scenario configuration.

    Aggregates all sub-configurations into a single immutable object
    representing a complete MDP problem instance.
    """

    name: str
    airport: AirportTopologyConfig
    aircraft_types: List[AircraftTypeConfig]
    compatibility: CompatibilityConfig
    schedule: ScheduleConfig
    noise_model: NoiseModelConfig
    rewards: RewardConfig
    time: TimeConfig

    # Computed properties for convenience

    @property
    def type_name_to_idx(self) -> Dict[str, int]:
        """Mapping from aircraft type name to index."""
        return {t.name: i for i, t in enumerate(self.aircraft_types)}

    @property
    def num_aircraft_types(self) -> int:
        """Number of aircraft types in scenario."""
        return len(self.aircraft_types)

    @property
    def num_gates(self) -> int:
        """Number of gates (convenience accessor)."""
        return self.airport.num_gates

    @property
    def num_runways(self) -> int:
        """Number of runways (convenience accessor)."""
        return self.airport.num_runways

    def validate_consistency(self):
        """
        Cross-validate configuration components.

        Raises:
            ValueError: If configuration is internally inconsistent
        """
        # Check compatibility matrix dimensions
        expected_shape = (self.num_aircraft_types, self.num_gates)
        if self.compatibility.compatibility_matrix.shape != expected_shape:
            raise ValueError(
                f"Compatibility matrix shape mismatch: "
                f"{self.compatibility.compatibility_matrix.shape} != {expected_shape}"
            )

        if self.compatibility.preference_matrix.shape != expected_shape:
            raise ValueError(
                f"Preference matrix shape mismatch: "
                f"{self.compatibility.preference_matrix.shape} != {expected_shape}"
            )

        # Check that probabilities sum to ~1
        total_prob = sum(t.probability for t in self.aircraft_types)
        if not np.isclose(total_prob, 1.0, atol=1e-3):
            raise ValueError(
                f"Aircraft type probabilities sum to {total_prob:.4f}, not 1.0"
            )

        # Check that every type can use at least one gate
        # Use aircraft type name instead of integer index
        for aircraft_type in self.aircraft_types:
            compatible_gates = self.compatibility.get_compatible_gates(aircraft_type.name)
            if len(compatible_gates) == 0:
                raise ValueError(
                    f"Aircraft type '{aircraft_type.name}' has no compatible gates"
                )

    def __str__(self) -> str:
        return (f"Scenario(name='{self.name}', "
                f"gates={self.num_gates}, runways={self.num_runways}, "
                f"types={self.num_aircraft_types}, flights={self.schedule.num_flights})")