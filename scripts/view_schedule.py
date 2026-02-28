from pathlib import Path
from collections import Counter
from src.config.loader import ScenarioLoader


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