"""
Master Orchestrator Pipeline.
Executes the Operations Research pipeline, allowing for decoupled training and evaluation.
"""
import subprocess
import sys
from pathlib import Path
import time
import argparse


def run_script(script_path: Path, args_list: list):
    """Runs a Python script as a subprocess with a list of arguments."""
    print(f"\n{'=' * 60}")
    print(f"🚀 EXECUTING: {script_path.name}")
    print(f"   Args:      {' '.join(args_list)}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    # Run the script and pass the arguments
    command = [sys.executable, str(script_path)] + args_list
    result = subprocess.run(command)

    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\n❌ PIPELINE HALTED: {script_path.name} failed with exit code {result.returncode}.")
        sys.exit(1)

    print(f"\n✅ SUCCESS: {script_path.name} completed in {elapsed:.2f} seconds.")


def main():
    parser = argparse.ArgumentParser(description="Run the ADP Airport Pipeline.")
    parser.add_argument("scenario", type=str, help="The YAML scenario file to run (e.g., stress_test.yaml)")
    parser.add_argument("--train", action="store_true", help="Include this flag to run training steps (01, 02, 03)")
    parser.add_argument("-c", "--continue-training", action="store_true",
                        help="Continue training from existing checkpoint")
    parser.add_argument("--model", type=str, default=None,
                        help="The prefix of the trained model to load. Defaults to the scenario name.")

    args = parser.parse_args()
    scenario_filename = args.scenario

    # If no specific model is provided, assume the model matches the scenario name
    model_prefix = args.model if args.model else Path(scenario_filename).stem

    experiments_dir = Path(__file__).resolve().parent

    training_steps = [
        "01_generate_features.py",
        "02_visualize_graph.py",
        "03_train_agent.py"
    ]

    evaluation_steps = [
        "01b_visualize_runways.py",
        "04_evaluate_policy.py",
        "05_visualize_schedule.py"
    ]

    total_start_time = time.time()
    print(f"🛫 INITIATING AIRPORT MDP PIPELINE 🛫")
    print(f"   Scenario Traffic: {scenario_filename}")
    print(f"   AI Brain (Model): {model_prefix}")
    print(f"   Mode: {'TRAIN + EVALUATE' if args.train else 'EVALUATE ONLY'}")

    # 1. Run Training Steps (Only if --train flag is present)
    if args.train:
        for step in training_steps:
            script_path = experiments_dir / step
            if script_path.exists():
                args_list = [scenario_filename]
                if step == "03_train_agent.py" and args.continue_training:
                    args_list.append("-c")
                # Training scripts just need the scenario name to generate the artifacts
                run_script(script_path, args_list)
            else:
                print(f"\n⚠️ WARNING: Could not find {step}. Skipping...")

    # 2. Run Evaluation Steps (Always run these)
    for step in evaluation_steps:
        script_path = experiments_dir / step
        if script_path.exists():
            # Evaluation scripts need BOTH the scenario traffic and the specific model brain to load
            run_script(script_path, [scenario_filename, model_prefix])
        else:
            print(f"\n⚠️ WARNING: Could not find {step}. Skipping...")

    total_elapsed = time.time() - total_start_time
    minutes = int(total_elapsed // 60)
    seconds = int(total_elapsed % 60)

    print(f"\n{'=' * 60}")
    print(f"🎉 PIPELINE COMPLETE in {minutes}m {seconds}s! 🎉")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
