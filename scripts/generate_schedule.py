from pathlib import Path
from collections import Counter
import numpy as np

from src.config.loader import ScenarioLoader
from src.mdp.components.schedule_generator import ScheduleGenerator


def generate_schedule_from_scenario(scenario_config_path: str, seed: int = 42):
    """
    Generate a synthetic schedule based on a scenario configuration file.

    Args:
        scenario_config_path: Path to scenario YAML file
        seed: Random seed for reproducibility
    """
    print(f"Loading scenario: {scenario_config_path}")

    # Load scenario configuration
    config_path = Path(scenario_config_path)
    scenario = ScenarioLoader.from_yaml(config_path)

    print(f"✓ Loaded scenario: {scenario.name}\n")

    # Check if schedule uses generation_params
    if scenario.schedule.generation_params is None:
        print("ERROR: Scenario does not specify generation_params.")
        print("This scenario likely references an existing schedule file.")
        print(f"Schedule config: {scenario.schedule}")
        return

    print("Generating schedule...")
    print(f"  Scenario: {scenario.schedule.scenario_name}")
    print(f"  Flights: {scenario.schedule.num_flights} (arrivals)")
    print(f"  Pattern: {scenario.schedule.generation_params.get('arrival_pattern', 'unknown')}")
    print(f"  Seed: {seed}\n")

    # Initialize RNG
    rng = np.random.default_rng(seed)

    expected_flights = scenario.schedule.num_flights
    actual_num_flights = rng.poisson(lam=expected_flights)

    # Generate flights (arrivals and departures)
    flights = ScheduleGenerator.generate(
        scenario_name=scenario.schedule.scenario_name,
        num_flights=actual_num_flights,  # Pass the randomized count here
        generation_params=scenario.schedule.generation_params,
        num_runways=scenario.airport.num_runways,
        aircraft_types=scenario.aircraft_types,
        rng=rng
    )

    # Print summary
    print("Generated flights:")
    print(f"  Total movements: {len(flights)}")

    direction_counts = Counter(f.direction for f in flights)
    print(f"  By direction: {dict(direction_counts)}")

    if flights:
        print(f"  Time range: {min(f.scheduled_time for f in flights)} - {max(f.scheduled_time for f in flights)} minutes")

    # Count by type
    type_counts = Counter(f.aircraft_type for f in flights)
    print(f"  By type: {dict(type_counts)}")

    # Count by runway
    runway_counts = Counter(f.runway for f in flights)
    print(f"  By runway: {dict(runway_counts)}")
    print()

    # Determine output path
    safe_name = scenario.schedule.scenario_name.lower().replace(' ', '_').replace('-', '_')
    output_path = Path(f"data/schedules/synthetic/{safe_name}_{scenario.schedule.num_flights}_full.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = {
        'scenario_name': scenario.schedule.scenario_name,
        'description': f"Generated from scenario: {scenario.name} (arrivals and departures)",
        'num_flights': scenario.schedule.num_flights,
        'num_movements': len(flights),
        'time_window': scenario.schedule.generation_params.get('time_window'),
        'num_runways': scenario.airport.num_runways,
        'seed': seed,
        'generation_params': scenario.schedule.generation_params,
        'source_config': str(config_path)
    }

    ScheduleGenerator.save_schedule(flights, output_path, metadata)
    print(f"✓ Saved schedule to: {output_path}")

    # Show first few flights
    print("\nFirst 10 movements:")
    for flight in flights[:10]:
        print(f"  {flight}")

    print("\n" + "="*70)
    print("NEXT STEP: Update your scenario YAML to reference this schedule file:")
    print(f"  schedule_file: \"{output_path}\"")
    print("="*70)