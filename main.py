"""
Main entry point for the airport MDP project.

This script provides a command-line interface to run various parts of the simulation,
such as inspecting scenarios, generating flight schedules, and visualizing data.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.inspect_scenario import inspect_scenario
from scripts.generate_schedule import generate_schedule_from_scenario
from scripts.view_schedule import load_and_show_flights
from scripts.visualize_schedule import generate_and_visualize_schedule

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Airport MDP Simulation CLI")

    # Common arguments
    parser.add_argument("scenario", type=str, help="Path to the scenario YAML file (e.g., configs/scenario/morning_rush.yaml)")

    # Mutually exclusive actions
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--inspect", action="store_true", help="Inspect the scenario configuration.")
    action_group.add_argument("--generate", action="store_true", help="Generate a synthetic flight schedule.")
    action_group.add_argument("--view-schedule", action="store_true", help="Load and display flights from the schedule.")
    action_group.add_argument("--visualize-schedule", action="store_true", help="Generate and visualize a runway schedule.")

    # Optional arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation tasks.")
    parser.add_argument("--num-flights", type=int, default=10, help="Number of flights to display for --view-schedule.")

    args = parser.parse_args()

    # Execute the chosen action
    if args.inspect:
        inspect_scenario(args.scenario)
    elif args.generate:
        generate_schedule_from_scenario(args.scenario, seed=args.seed)
    elif args.view_schedule:
        load_and_show_flights(args.scenario, num_flights=args.num_flights)
    elif args.visualize_schedule:
        generate_and_visualize_schedule(args.scenario, seed=args.seed)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()