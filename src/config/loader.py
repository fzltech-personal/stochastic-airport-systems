"""
Factory for loading scenarios from YAML files.
"""
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

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

        scenario = ScenarioLoader._build_scenario(data, filepath.parent)
        scenario.validate_consistency()

        return scenario

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Helper to load a YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    @staticmethod
    def _build_scenario(data: Dict[str, Any], base_path: Path) -> ScenarioConfig:
        """Construct ScenarioConfig from parsed YAML dict."""

        # Determine project root (assuming src/config/loader.py structure)
        project_root = Path(__file__).parent.parent.parent

        # 1. Load Airport Configuration
        airport_path_str = data.get('airport')
        if not airport_path_str:
            raise ValueError("Scenario must specify 'airport' path.")

        airport_path = Path(airport_path_str)
        if not airport_path.is_absolute():
            airport_path = project_root / airport_path

        airport_data = ScenarioLoader._load_yaml(airport_path)
        airport = ScenarioLoader._build_airport(airport_data)

        # 2. Load Aircraft Types (Fleet Mix)
        aircraft_types_path = project_root / "configs/components/aircraft_types.yaml"
        aircraft_types_data = ScenarioLoader._load_yaml(aircraft_types_path)

        fleet_mix_name = data.get('fleet_mix')
        if not fleet_mix_name:
            raise ValueError("Scenario must specify 'fleet_mix'.")

        # Extract the base definitions and the specific mix
        base_types = aircraft_types_data.get('aircraft_types', {})
        fleet_mixes = aircraft_types_data.get('fleet_mixes', {})

        if fleet_mix_name not in fleet_mixes:
            raise ValueError(f"Fleet mix '{fleet_mix_name}' not found in {aircraft_types_path}")

        mix_probabilities = fleet_mixes[fleet_mix_name]
        aircraft_types = []

        for type_name, probability in mix_probabilities.items():
            if type_name not in base_types:
                raise ValueError(f"Aircraft type '{type_name}' found in mix but missing from definitions.")

            # Copy base data so we don't mutate the loaded dictionary
            type_data = base_types[type_name].copy()
            type_data['probability'] = float(probability)

            # Construct the config object
            aircraft_types.append(AircraftTypeConfig(**type_data))

        # Validate accepted types and determine row indices for matrix slicing
        accepted_types = airport_data.get('accepted_aircraft_types')
        active_indices = []

        if accepted_types:
            for at in aircraft_types:
                if at.name not in accepted_types:
                    raise ValueError(
                        f"Aircraft type '{at.name}' is not accepted by the airport "
                        f"(accepted: {accepted_types})"
                    )
                # Keep track of the index for slicing later
                active_indices.append(accepted_types.index(at.name))
        else:
            # If no accepted_types list exists, we assume a 1:1 mapping (legacy support)
            active_indices = list(range(len(aircraft_types)))

        # 3. Load Compatibility (Smart Slicing)
        if 'compatibility' in airport_data:
            comp_data = airport_data['compatibility']

            # Convert to numpy arrays to allow multi-row slicing
            full_matrix = np.array(comp_data['matrix'])
            full_prefs = np.array(comp_data['preferences'])

            # Slice the matrices to only include active aircraft types
            sliced_matrix = full_matrix[active_indices, :]
            sliced_prefs = full_prefs[active_indices, :]

            compatibility = CompatibilityConfig(
                compatibility_matrix=sliced_matrix,
                preference_matrix=sliced_prefs
            )
        else:
            # Fallback if defined directly in scenario (old style)
            if 'compatibility' in data:
                compatibility = ScenarioLoader._build_compatibility(data['compatibility'])
            else:
                raise ValueError("Compatibility matrix must be defined in airport config or scenario.")

        # 4. Parse Schedule
        schedule_data = data['schedule']
        schedule_file = schedule_data.get('schedule_file')
        if schedule_file:
            schedule_file = Path(schedule_file)

        schedule = ScheduleConfig(
            scenario_name=schedule_data['scenario_name'],
            num_flights=schedule_data['num_flights'],
            flights=schedule_data.get('flights'),
            generation_params=schedule_data.get('generation_params'),
            schedule_file=schedule_file
        )

        # 5. Parse Noise Model
        noise_model = NoiseModelConfig(
            distribution=data['noise_model']['distribution'],
            params=data['noise_model']['params']
        )

        # 6. Load Rewards
        rewards_path_str = data.get('rewards')
        if isinstance(rewards_path_str, str):
            # It's a path to a rewards file
            rewards_path = Path(rewards_path_str)
            if not rewards_path.is_absolute():
                rewards_path = project_root / rewards_path

            rewards_data_full = ScenarioLoader._load_yaml(rewards_path)
            reward_profile = data.get('reward_profile')
            if not reward_profile:
                raise ValueError("When using external rewards file, 'reward_profile' must be specified.")

            if reward_profile not in rewards_data_full:
                raise ValueError(f"Reward profile '{reward_profile}' not found in {rewards_path}")

            rewards_data = rewards_data_full[reward_profile]

            # Filter out non-field keys like 'description', 'rationale', 'inherit'
            valid_fields = {'c_wait', 'c_overflow', 'c_assign', 'beta', 'Q_max'}
            filtered_rewards = {k: v for k, v in rewards_data.items() if k in valid_fields}
            rewards = RewardConfig(**filtered_rewards)

        else:
            # Inline definition (old style)
            rewards = RewardConfig(**data['rewards'])

        # 7. Parse Time Config
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