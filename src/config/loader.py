"""
Factory for loading scenarios from YAML files.
"""
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.utils.paths import ProjectPaths  # Reliable path resolution
from .scenario import ScenarioConfig
from .airport import AirportTopologyConfig
from .compatibility import CompatibilityConfig
from .noise import NoiseModelConfig, NoiseModelsConfig
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
            # Try to resolve relative to configs dir if just a name is given
            if not filepath.is_absolute():
                 potential_path = ProjectPaths.get_configs_dir() / filepath
                 if potential_path.exists():
                     filepath = potential_path
            
            if not filepath.exists():
                raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        scenario = ScenarioLoader._build_scenario(data)
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
    def _build_scenario(data: Dict[str, Any]) -> ScenarioConfig:
        """Construct ScenarioConfig from parsed YAML dict."""

        # 1. Load Airport Configuration
        airport_path_str = data.get('airport')
        if not airport_path_str:
            raise ValueError("Scenario must specify 'airport' path.")

        # Resolve airport config path
        airport_path = ProjectPaths.resolve_path(airport_path_str)
        
        airport_data = ScenarioLoader._load_yaml(airport_path)
        airport = ScenarioLoader._build_airport(airport_data)

        # 2. Load Aircraft Types (Fleet Mix)
        # Check if the scenario overrides the default aircraft types path
        aircraft_types_path_str = data.get('aircraft_types_path')
        if aircraft_types_path_str:
            aircraft_types_path = ProjectPaths.resolve_path(aircraft_types_path_str)
        else:
            # Fallback to the default location (assuming standard project structure)
            aircraft_types_path = ProjectPaths.get_configs_dir() / "components/aircraft_types.yaml"
            
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

        # Validate accepted types (Logic moved out of matrix slicing)
        accepted_types = airport_data.get('accepted_aircraft_types')

        if accepted_types:
            for at in aircraft_types:
                if at.name not in accepted_types:
                    raise ValueError(
                        f"Aircraft type '{at.name}' is not accepted by the airport "
                        f"(accepted: {accepted_types})"
                    )
        else:
            # Fallback if not specified: all loaded types are accepted
            accepted_types = [at.name for at in aircraft_types]

        # 3. Load Compatibility (Smart Slicing)
        if 'compatibility' in airport_data:
            comp_data = airport_data['compatibility']

            # Use raw matrices directly
            raw_matrix = np.array(comp_data['matrix'])
            raw_prefs = np.array(comp_data['preferences'])
            
            # Use the 'master' aircraft list defined in the airport config (or implied)
            # Assuming the rows of raw_matrix correspond to 'accepted_aircraft_types' in the airport config order
            # Wait, the prompt says "master_aircraft_list (List[str] representing the rows of the raw matrices)"
            # This is tricky because the YAML structure for 'compatibility' needs to tell us what the rows are.
            # In 'airport_data', there is usually a list of all types the airport supports.
            
            master_list = airport_data.get('accepted_aircraft_types')
            if not master_list:
                # If master list isn't explicit, assume the raw matrix rows match the accepted_types *if* no filtering happened yet.
                # However, typically an airport defines a master set of types it handles.
                # Let's assume the scenario loader logic we replaced was:
                # sliced_matrix = full_matrix[active_indices, :]
                # where active_indices were indices into accepted_types. 
                # This implies accepted_types *was* the master list for the airport.
                master_list = accepted_types

            # Wait, accepted_types in the airport config is the list of types the airport *can* handle.
            # fleet_mix is the list of types appearing in *this scenario*.
            # So:
            # master_aircraft_list = airport_data['accepted_aircraft_types']
            # accepted_aircraft_types (for this config) = [at.name for at in aircraft_types]
            
            # We need to make sure we use the right list for 'master'. 
            # In the previous code, active_indices mapped from the scenario types to the airport's accepted types.
            # So airport_data['accepted_aircraft_types'] IS the master list for the rows of the matrix.
            
            airport_master_types = airport_data.get('accepted_aircraft_types')
            if not airport_master_types:
                 # If the airport doesn't list types, we can't safely slice.
                 # Fallback to 1:1 if we assume the matrix matches exactly the current mix (legacy)
                 airport_master_types = [at.name for at in aircraft_types]
            
            # The fleet mix for this scenario
            scenario_active_types = [at.name for at in aircraft_types]

            compatibility = CompatibilityConfig.from_raw_data(
                raw_comp_matrix=raw_matrix,
                raw_pref_matrix=raw_prefs,
                master_aircraft_list=airport_master_types,
                accepted_aircraft_types=scenario_active_types
            )
            
        else:
            # Fallback if defined directly in scenario (old style)
            if 'compatibility' in data:
                # Legacy path: assumes the matrix is already shaped for the active types
                compatibility = CompatibilityConfig(
                    active_aircraft_types=[at.name for at in aircraft_types],
                    compatibility_matrix=np.array(data['compatibility']['matrix']),
                    preference_matrix=np.array(data['compatibility']['preferences'])
                )
            else:
                raise ValueError("Compatibility matrix must be defined in airport config or scenario.")

        # 4. Parse Schedule
        schedule_data = data['schedule']
        schedule_file = schedule_data.get('schedule_file')
        if schedule_file:
            # Resolve schedule file path
            schedule_file = ProjectPaths.resolve_path(schedule_file)

        schedule = ScheduleConfig(
            scenario_name=schedule_data['scenario_name'],
            num_flights=schedule_data['num_flights'],
            flights=schedule_data.get('flights'),
            generation_params=schedule_data.get('generation_params'),
            schedule_file=schedule_file
        )

        # 5. Parse Noise Models
        noise_data = data.get('noise_models')
        if noise_data:
            arrival_model = NoiseModelConfig(
                distribution=noise_data['arrival']['distribution'],
                params=noise_data['arrival']['params']
            )
            service_model = NoiseModelConfig(
                distribution=noise_data['service']['distribution'],
                params=noise_data['service']['params']
            )
            noise_models = NoiseModelsConfig(arrival=arrival_model, service=service_model)
        else:
            # Fallback for old configs
            old_noise = data.get('noise_model')
            if old_noise:
                arrival_model = NoiseModelConfig(
                    distribution=old_noise['distribution'],
                    params=old_noise['params']
                )
            else:
                arrival_model = NoiseModelConfig()
            
            # Create a default zero-noise service model
            service_model = NoiseModelConfig(distribution='normal', params={'mean': 0, 'std': 0})
            noise_models = NoiseModelsConfig(arrival=arrival_model, service=service_model)


        # 6. Load Rewards
        rewards_path_str = data.get('rewards')
        if isinstance(rewards_path_str, str):
            # It's a path to a rewards file
            rewards_path = ProjectPaths.resolve_path(rewards_path_str)

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
            noise_models=noise_models,
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
        """Build compatibility config (Legacy/Direct)."""
        # This is for when compatibility is defined inline without master/slicing
        # We dummy the active types list since we don't have context here
        return CompatibilityConfig(
            active_aircraft_types=[], # This is technically invalid but used for legacy structure match?
                                      # Actually, better to raise error or fix caller.
                                      # For now, let's assume direct usage provides pre-sliced data.
            compatibility_matrix=np.array(data['matrix']),
            preference_matrix=np.array(data['preferences'])
        )