"""
Main entry point for the airport MDP project.

Uses configuration files to define scenarios and generate schedules.
"""
import sys
from pathlib import Path
import numpy as np
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config.loader import ScenarioLoader
from src.mdp.components.schedule_generator import ScheduleGenerator
from src.utils.visualization import plot_runway_schedule


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

    # Generate flights (arrivals and departures)
    flights = ScheduleGenerator.generate(
        scenario_name=scenario.schedule.scenario_name,
        num_flights=scenario.schedule.num_flights,
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


def inspect_scenario(scenario_config_path: str):
    """
    Load and inspect a scenario configuration.

    Args:
        scenario_config_path: Path to scenario YAML file
    """
    print(f"Loading scenario: {scenario_config_path}\n")

    config_path = Path(scenario_config_path)
    scenario = ScenarioLoader.from_yaml(config_path)

    print("="*70)
    print(f"SCENARIO: {scenario.name}")
    print("="*70)
    print()

    # Airport
    print("AIRPORT:")
    print(f"  Gates: {scenario.airport.num_gates}")
    print(f"  Runways: {scenario.airport.num_runways}")
    print(f"  Delta matrix shape: {scenario.airport.delta_matrix.shape}")
    print(f"  Taxiing time range: {scenario.airport.delta_matrix.min():.1f} - {scenario.airport.delta_matrix.max():.1f} min")
    print()

    # Aircraft types
    print("AIRCRAFT TYPES:")
    for i, aircraft_type in enumerate(scenario.aircraft_types):
        print(f"  [{i}] {aircraft_type}")
    print()

    # Compatibility
    print("COMPATIBILITY:")
    print(f"  Matrix shape: {scenario.compatibility.compatibility_matrix.shape}")
    for i, aircraft_type in enumerate(scenario.aircraft_types):
        compatible_gates = scenario.compatibility.get_compatible_gates(i)
        print(f"  {aircraft_type.name}: gates {list(compatible_gates)}")
    print()

    # Schedule
    print("SCHEDULE:")
    print(f"  Scenario: {scenario.schedule.scenario_name}")
    print(f"  Flights: {scenario.schedule.num_flights}")
    if scenario.schedule.schedule_file:
        print(f"  Source: {scenario.schedule.schedule_file}")
        if Path(scenario.schedule.schedule_file).exists():
            print("  Status: ✓ File exists")
        else:
            print("  Status: ✗ File not found (needs generation)")
    elif scenario.schedule.generation_params:
        print(f"  Source: Generated ({scenario.schedule.generation_params.get('arrival_pattern', 'unknown')} pattern)")
        print("  Status: ⚠ Not yet generated")
    print()

    # Noise model
    print("NOISE MODEL:")
    print(f"  Distribution: {scenario.noise_model.distribution}")
    print(f"  Parameters: {scenario.noise_model.params}")
    print()

    # Rewards
    print("REWARDS:")
    print(f"  c_wait: {scenario.rewards.c_wait}")
    print(f"  c_overflow: {scenario.rewards.c_overflow}")
    print(f"  c_assign: {scenario.rewards.c_assign}")
    print(f"  beta (preference): {scenario.rewards.beta}")
    print(f"  Q_max: {scenario.rewards.Q_max}")
    print()

    # Time
    print("TIME:")
    print(f"  Horizon: {scenario.time.horizon} minutes")
    print(f"  Timestep: {scenario.time.timestep} minute(s)")
    print(f"  Total steps: {scenario.time.num_timesteps}")
    print(f"  Max service time: {scenario.time.S_max} minutes")
    print()

    print("="*70)


def load_and_show_flights(scenario_config_path: str, num_flights: int = 10):
    """
    Load a scenario and display its flights.

    Args:
        scenario_config_path: Path to scenario YAML file
        num_flights: Number of flights to display
    """
    print(f"Loading scenario: {scenario_config_path}\n")

    config_path = Path(scenario_config_path)
    scenario = ScenarioLoader.from_yaml(config_path)

    print(f"✓ Loaded scenario: {scenario.name}\n")

    try:
        flights = scenario.schedule.get_flights()

        print(f"FLIGHTS: {len(flights)} total")
        if flights:
            print(f"  Time range: {min(f.scheduled_time for f in flights)} - {max(f.scheduled_time for f in flights)} minutes")

        # Count by type
        type_counts = Counter(f.aircraft_type for f in flights)
        print(f"  By type: {dict(type_counts)}")

        # Count by runway
        runway_counts = Counter(f.runway for f in flights)
        print(f"  By runway: {dict(runway_counts)}")
        print()

        print(f"First {num_flights} flights:")
        for flight in flights[:num_flights]:
            print(f"  {flight}")

    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: Could not load flights: {e}")
        print("\nIf this scenario uses generation_params, run:")
        print(f"  python main.py --generate-schedule")


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

    flights = ScheduleGenerator.generate(
        scenario_name=scenario.schedule.scenario_name,
        num_flights=scenario.schedule.num_flights,
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


def main():
    """Main entry point."""

    # =========================================================================
    # CONFIGURATION: Specify which scenario to work with
    # =========================================================================

    # Choose one of:
    # scenario_config = "configs/scenarios/toy_problem.yaml"
    scenario_config = r"D:\_programming\uni\own-projects\stochastic-airport-systems\configs\scenario\morning_rush.yaml"
    # scenario_config = "configs/scenarios/disrupted_evening.yaml"

    # Random seed for reproducibility
    seed = 42

    # =========================================================================
    # CHOOSE ACTION: Uncomment the action you want to perform
    # =========================================================================

    # Action 1: Inspect the scenario configuration
    # inspect_scenario(scenario_config)

    # Action 2: Generate a synthetic schedule from generation_params
    # generate_schedule_from_scenario(scenario_config, seed=seed)

    # Action 3: Load and display flights from existing schedule file
    # load_and_show_flights(scenario_config, num_flights=10)

    # Action 4: Generate a schedule and visualize it
    generate_and_visualize_schedule(scenario_config, seed=seed)


if __name__ == '__main__':
    main()