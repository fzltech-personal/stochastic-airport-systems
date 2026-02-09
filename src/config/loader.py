"""
Factory for loading scenarios from YAML files.
"""
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any

from .scenario import ScenarioConfig
from .airport import AirportTopologyConfig
from .compatibility import CompatibilityConfig
from .noise import NoiseModelConfig
from .reward import RewardConfig
from .time import TimeConfig
from .schedule import ScheduleConfig
from ..mdp.components.aircraft import AircraftTypeConfig


class ScenarioLoader:
    """Load and construct ScenarioConfig from YAML files."""

    @staticmethod
    def from_yaml(filepath: Path) -> ScenarioConfig:
        """
        Load scenario from YAML file.

        Args:
            filepath: Path to YAML configuration file

        Returns:
            Validated ScenarioConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If file is malformed
            ValueError: If configuration is invalid
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        scenario = ScenarioLoader._build_scenario(data)
        scenario.validate_consistency()

        return scenario

    @staticmethod
    def _build_scenario(data: Dict[str, Any]) -> ScenarioConfig:
        """Construct ScenarioConfig from parsed YAML dict."""

        # Parse airport topology
        airport = ScenarioLoader._build_airport(data['airport'])

        # Parse aircraft types
        aircraft_types = [
            AircraftTypeConfig(**type_data)
            for type_data in data['aircraft_types']
        ]

        # Parse compatibility
        compatibility = ScenarioLoader._build_compatibility(data['compatibility'])

        # Parse schedule
        schedule = ScheduleConfig(
            scenario_name=data['schedule']['scenario_name'],
            num_flights=data['schedule']['num_flights'],
            flights=data['schedule'].get('flights'),
            generation_params=data['schedule'].get('generation_params')
        )

        # Parse noise model
        noise_model = NoiseModelConfig(
            distribution=data['noise_model']['distribution'],
            params=data['noise_model']['params']
        )

        # Parse rewards
        rewards = RewardConfig(**data['rewards'])

        # Parse time config
        time = TimeConfig(**data['time'])

        # Construct complete scenario
        return ScenarioConfig(
            name=data['name'],
            airport=airport,
            aircraft_types=aircraft_types,
            compatibility=compatibility,
            schedule=schedule,
            noise_model=noise_model,
            rewards=rewards,
            time=time
        )

    @staticmethod
    def _build_airport(data: Dict[str, Any]) -> AirportTopologyConfig:
        """Build airport topology config."""
        return AirportTopologyConfig(
            num_gates=data['num_gates'],
            num_runways=data['num_runways'],
            delta_matrix=np.array(data['delta_matrix'])
        )

    @staticmethod
    def _build_compatibility(data: Dict[str, Any]) -> CompatibilityConfig:
        """Build compatibility config."""
        return CompatibilityConfig(
            compatibility_matrix=np.array(data['matrix']),
            preference_matrix=np.array(data['preferences'])
        )