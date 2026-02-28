from pathlib import Path
import numpy as np

from src.config.loader import ScenarioLoader
from src.mdp.components.schedule_generator import ScheduleGenerator
from src.utils.visualization import plot_runway_schedule


def generate_and_visualize_schedule(scenario_config_path: str, seed: int = 42):
    """
    Generate a synthetic schedule and visualize it.
    """
    print(f"Loading scenario: {scenario_config_path}")
    config_path = Path(scenario_config_path)
    scenario = ScenarioLoader.from_yaml(config_path)
    print(f"✓ Loaded scenario: {scenario.name}\n")

    if scenario.schedule.generation_params is None:
        print("ERROR: Scenario does not specify generation_params.")
        return

    rng = np.random.default_rng(seed)

    expected_flights = scenario.schedule.num_flights
    actual_num_flights = rng.poisson(lam=expected_flights)
    print(f"Sampled {actual_num_flights} total flights from Poisson(lambda={expected_flights}).")

    flights = ScheduleGenerator.generate(
        scenario_name=scenario.schedule.scenario_name,
        num_flights=actual_num_flights,  # Pass the randomized count here
        generation_params=scenario.schedule.generation_params,
        num_runways=scenario.airport.num_runways,
        aircraft_types=scenario.aircraft_types,
        rng=rng
    )

    print(f"Generated {len(flights)} total movements.")

    # Get occupancy time from config for visualization
    occupancy_time = scenario.schedule.generation_params.get('runway_occupancy_time', 1)

    plot_runway_schedule(
        flights,
        num_runways=scenario.airport.num_runways,
        title=f"Generated Runway Schedule: {scenario.schedule.scenario_name}",
        block_duration=occupancy_time
    )