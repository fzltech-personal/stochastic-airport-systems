"""
Main entry point for the Stochastic Airport Systems project.

This script provides a command-line interface to run various parts of the simulation,
or execute the end-to-end ADP Machine Learning pipeline.
"""

import sys
import argparse
import subprocess
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.inspect_scenario import inspect_scenario
from scripts.generate_schedule import generate_schedule_from_scenario
from scripts.view_schedule import load_and_show_flights
from scripts.visualize_schedule import generate_and_visualize_schedule


def create_dynamic_scenario(json_path: Path, base_yaml: str) -> str:
    """Wraps a raw schedule.json into a temporary YAML scenario for the pipeline."""
    config_dir = Path(__file__).resolve().parent / "configs" / "scenarios"
    base_path = config_dir / base_yaml

    if not base_path.exists():
        print(f"❌ ERROR: Base config '{base_yaml}' not found in configs/scenarios/.")
        sys.exit(1)

    with open(base_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Overwrite the schedule block to point directly to the real-time JSON
    base_config['name'] = f"Real-Time Injection: {json_path.stem}"
    base_config['schedule'] = {
        'scenario_name': json_path.stem,
        'num_flights': 2,  # Dummy value to pass validation
        'schedule_file': str(json_path.resolve())
    }

    # Save as a temporary scenario
    temp_filename = f"auto_{json_path.stem}.yaml"
    temp_path = config_dir / temp_filename

    with open(temp_path, 'w') as f:
        yaml.dump(base_config, f, sort_keys=False)

    return temp_filename


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Airport MDP Simulation & ML CLI")

    # Core Input
    parser.add_argument("input", type=str, help="Path to scenario (.yaml) OR raw real-time schedule (.json)")
    parser.add_argument("--base-config", type=str, default="master_training.yaml",
                        help="Base physics config to use if a .json schedule is provided.")

    # Optional Actions (Defaults to running the ML pipeline)
    action_group = parser.add_mutually_exclusive_group(required=False)
    action_group.add_argument("--inspect", action="store_true", help="Inspect the scenario configuration.")
    action_group.add_argument("--generate", action="store_true", help="Generate a synthetic flight schedule.")
    action_group.add_argument("--view-schedule", action="store_true", help="Load and display flights from the schedule.")
    action_group.add_argument("--visualize-schedule", action="store_true", help="Generate and visualize a runway schedule.")

    # ML Pipeline arguments
    parser.add_argument("--train", action="store_true", help="Run the training steps (Builds graph & learns weights).")
    parser.add_argument("-c", "--continue-training", action="store_true",
                        help="Continue training from existing checkpoint (Used with --train).")
    parser.add_argument("--model", type=str, default=None, help="Specific trained model prefix to load.")

    # Data args
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation tasks.")
    parser.add_argument("--num-flights", type=int, default=10, help="Number of flights to display.")

    args = parser.parse_args()

    input_path = Path(args.input)

    # 1. Handle JSON Real-Time Injection
    if input_path.suffix == '.json':
        if not input_path.exists():
            print(f"❌ ERROR: JSON file '{input_path}' not found.")
            sys.exit(1)
        print(f"🔄 Detected raw JSON schedule. Wrapping with physics from {args.base_config}...")
        scenario_filename = create_dynamic_scenario(input_path, args.base_config)
    else:
        # Assume it's a standard YAML scenario in the configs/scenarios folder
        scenario_filename = input_path.name

    # 2. Execute Action
    if args.inspect:
        inspect_scenario(scenario_filename)
    elif args.generate:
        generate_schedule_from_scenario(scenario_filename, seed=args.seed)
    elif args.view_schedule:
        load_and_show_flights(scenario_filename, num_flights=args.num_flights)
    elif args.visualize_schedule:
        generate_and_visualize_schedule(scenario_filename, seed=args.seed)
    else:
        # DEFAULT BEHAVIOR: Run the Orchestrator Pipeline
        print(f"\n🚀 Routing to AI Orchestrator...")
        cmd = [sys.executable, "experiments/run_pipeline.py", scenario_filename]

        if args.train:
            cmd.append("--train")
            if args.continue_training:
                cmd.append("-c")

        # If a model is specified, use it.
        if args.model:
            cmd.extend(["--model", args.model])
        # If no model is specified but we injected a JSON, default to using the base_config's brain!
        elif input_path.suffix == '.json':
            cmd.extend(["--model", Path(args.base_config).stem])

        subprocess.run(cmd)

if __name__ == '__main__':
    main()