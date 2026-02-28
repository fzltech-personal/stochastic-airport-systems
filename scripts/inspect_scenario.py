from pathlib import Path
from src.config.loader import ScenarioLoader


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