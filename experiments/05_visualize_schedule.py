"""
Experiment: Visualize the Schedule and Gate Assignments (Gantt Chart).
Runs a single episode with the trained ADP Agent and plots a timeline.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TABLEAU_COLORS
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.paths import ProjectPaths
from src.config.loader import ScenarioLoader
from src.mdp.environment import AirportEnvironment
from src.simulation.simulator import Simulator
from src.adp.features import PVFFeatureExtractor
from src.adp.value_function import LinearVFA
from src.adp.policies import ADPPolicy
from attr import evolve


def main(scenario_filename: str, model_prefix: str, timestamp: str = None):
    config_path = ProjectPaths.get_configs_dir() / f"scenarios/{scenario_filename}"
    scenario_prefix = config_path.stem

    if not timestamp:
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")

    print(f"Loading Scenario '{model_prefix}'...")
    scenario = ScenarioLoader.from_yaml(config_path)

    # Load the schedule from the saved artifact
    json_path = ProjectPaths.get_data_dir() / f"schedules/synthetic/{scenario_prefix}_eval_schedule.json"
    if not json_path.exists():
        # Fallback to processed dir if missing in synthetic
        json_path = ProjectPaths.get_data_dir() / f"processed/{scenario_prefix}_eval_schedule.json"
        
    if not json_path.exists():
        print(f"!!! ERROR: Schedule artifact '{json_path}' not found.")
        print(f"Please run 01b_visualize_runways.py with '{scenario_filename}' first!")
        return

    new_schedule = evolve(
        scenario.schedule,
        schedule_file=str(json_path),
        generation_params=None,
        flights=None
    )
    scenario = evolve(scenario, schedule=new_schedule)
    env = AirportEnvironment(scenario)

    print("Loading Trained Brain...")
    data_dir = ProjectPaths.get_data_dir() / "processed"

    basis_path = data_dir / f"{model_prefix}_basis_functions.npy"
    mapping_path = data_dir / f"{model_prefix}_state_mapping.pkl"
    theta_path = data_dir / f"{model_prefix}_learned_theta.npy"

    if not theta_path.exists():
        print(f"!!! ERROR: Learned weights for '{model_prefix}' not found.")
        return

    extractor = PVFFeatureExtractor(str(basis_path), str(mapping_path))
    vfa = LinearVFA(num_features=extractor.num_features)
    vfa.theta = np.load(theta_path)

    # Pure exploitation - we want to see its best possible performance
    adp_policy = ADPPolicy(vfa=vfa, extractor=extractor, epsilon=0.0, gamma=0.99)
    simulator = Simulator(env, adp_policy)

    print("Running simulation...")
    trajectory = simulator.run_episode()

    # --- EXTRACT DATA FOR GANTT CHART ---
    assignments = []
    queue_lengths = []
    times = []
    step_counter = 0

    # Get the flights correctly
    flights = env.scenario.schedule.get_flights()

    for state, action, reward, next_state in trajectory:
        # Extract queue data for every step
        queue_lengths.append(len(state.runway_queue))
        times.append(state.t)
        
        if not getattr(action, 'is_noop', True):

            # 1. Search the list for the matching flight object
            flight = next((f for f in flights if f.flight_id == action.flight_id), None)

            # 2. Extract attributes
            if flight:
                ac_type = flight.aircraft_type
            else:
                ac_type = 'Unknown'

            # Calculate exact duration by looking at what the environment actually locked in!
            # The next_state reflects the timer exactly 1 minute *after* the assignment.
            # So the total duration is next_state.gates[gate] + 1
            duration = next_state.gates[action.gate_idx] + 1

            assignment_time = getattr(state, 'time',
                                      getattr(state, 'current_time',
                                              getattr(state, 'step', step_counter)))

            assignments.append({
                "flight_id": action.flight_id,
                "gate": action.gate_idx,
                "start": assignment_time,
                "duration": duration,
                "type": ac_type
            })

        step_counter += 1

    if not assignments:
        print("No assignments made during the episode! Nothing to plot.")
        return

    # --- PLOT THE GANTT CHART ---
    print("Plotting Gantt Chart...")

    # Map aircraft types to colors for easy reading
    unique_types = list(set(a["type"] for a in assignments))
    color_palette = list(TABLEAU_COLORS.values())
    type_colors = {ac_type: color_palette[i % len(color_palette)] for i, ac_type in enumerate(unique_types)}

    # Figure size scales with the number of unique gates used
    used_gates = sorted(list(set(a["gate"] for a in assignments)))
    fig, ax = plt.subplots(figsize=(16, max(6, len(used_gates) * 0.4)))

    for task in assignments:
        gate = task["gate"]
        start = task["start"]
        duration = task["duration"]
        color = type_colors[task["type"]]

        # Plot the bar
        ax.broken_barh([(start, duration)], (gate - 0.4, 0.8), facecolors=color, edgecolor='black', alpha=0.8)

        # Add the Flight ID text in the middle of the bar
        ax.text(start + duration / 2, gate, str(task["flight_id"]),
                ha='center', va='center', color='white', fontsize=8, fontweight='bold',
                clip_on=True)

    # Formatting the chart
    ax.set_yticks(used_gates)
    ax.set_yticklabels([f"Gate {g}" for g in used_gates])
    ax.set_xlabel("Time (Minutes from start of simulation)")
    ax.set_title(f"Airport Gate Schedule ({model_prefix.upper()})\nTrained ADP Policy Execution")
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)

    # Create a custom legend for Aircraft Types
    legend_patches = [mpatches.Patch(color=color, label=ac_type) for ac_type, color in type_colors.items()]
    ax.legend(handles=legend_patches, title="Aircraft Type", loc="upper right")

    plt.tight_layout()

    # Save output
    output_dir = ProjectPaths.get_root() / f"experiments/results/plots/{scenario_prefix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    gantt_path = output_dir / f"{timestamp}_{scenario_prefix}_gantt_chart.png"

    plt.savefig(gantt_path)
    print(f"Schedule visualized and saved to: {gantt_path}")
    plt.close(fig) # Close the figure to free memory
    
    # --- PLOT THE QUEUE LENGTH ---
    print("Plotting Queue Length...")
    
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    
    ax2.plot(times, queue_lengths, color='red', linewidth=2, label="Queue Length")
    ax2.fill_between(times, queue_lengths, 0, color='red', alpha=0.3)
    
    # Formatting the queue chart
    ax2.set_xlabel("Time (Minutes)")
    ax2.set_ylabel("Number of Planes Waiting")
    ax2.set_title(f"Taxiway Queue Over Time ({model_prefix.upper()})")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlim(min(times), max(times))
    ax2.set_ylim(bottom=0) # Queue can't go below 0
    ax2.legend()
    
    plt.tight_layout()
    
    queue_path = output_dir / f"{timestamp}_{scenario_prefix}_queue_length.png"
    plt.savefig(queue_path)
    print(f"Queue length visualized and saved to: {queue_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", nargs="?", default="morning_rush.yaml")
    parser.add_argument("model", nargs="?", default=None)
    parser.add_argument("--timestamp", type=str, default=None)
    
    args = parser.parse_args()
    
    scenario_filename = args.scenario
    model_prefix = args.model if args.model else Path(scenario_filename).stem
    
    main(scenario_filename, model_prefix, timestamp=args.timestamp)