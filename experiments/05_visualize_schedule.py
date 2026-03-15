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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.paths import ProjectPaths
from src.config.loader import ScenarioLoader
from src.mdp.components.schedule_generator import ScheduleGenerator
from src.mdp.environment import AirportEnvironment
from src.simulation.simulator import Simulator
from src.adp.features import PVFFeatureExtractor
from src.adp.value_function import LinearVFA
from src.adp.policies import ADPPolicy


def main(scenario_filename: str, model_prefix: str):
    # --- EASILY SWITCH SCENARIOS HERE ---
    # scenario_filename = "greedy_trap.yaml"
    # ------------------------------------

    config_path = ProjectPaths.get_configs_dir() / f"scenarios/{scenario_filename}"
    # scenario_prefix = config_path.stem

    print(f"Loading Scenario '{model_prefix}'...")
    scenario = ScenarioLoader.from_yaml(config_path)

    # Use a fixed seed so we get a reproducible, clean schedule
    rng = np.random.default_rng(999)
    flights = ScheduleGenerator.generate(
        scenario_name=scenario.schedule.scenario_name,
        num_flights=scenario.schedule.num_flights,
        generation_params=scenario.schedule.generation_params,
        num_runways=scenario.airport.num_runways,
        aircraft_types=scenario.aircraft_types,
        rng=rng
    )

    from attr import evolve
    new_schedule = evolve(
        scenario.schedule,
        flights=flights,
        generation_params=None,
        schedule_file=None,
        num_flights=len(flights)
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
    step_counter = 0

    for state, action, reward, next_state in trajectory:
        if not getattr(action, 'is_noop', True):

            # 1. Search the list for the matching flight object
            flight = next((f for f in env.scenario.schedule.flights if f.flight_id == action.flight_id), None)

            # 2. Safely extract attributes if the flight was found
            if flight:
                ac_type = flight.aircraft_type
                
                # Check for linked flight to calculate true duration
                if flight.linked_flight_id:
                    linked_flight = next((f for f in env.scenario.schedule.flights if f.flight_id == flight.linked_flight_id), None)
                    if linked_flight:
                        # Assuming departure time > arrival time
                        duration = abs(linked_flight.scheduled_time - flight.scheduled_time)
                    else:
                        print(f"!!! WARNING: Linked flight '{flight.linked_flight_id}' not found!")
                        duration = 45 # Fallback if linked flight not found
                else:
                    duration = 45 # Fallback if no linked_flight_id
            else:
                ac_type = 'Unknown'
                duration = 45

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
    output_dir = ProjectPaths.get_root() / "experiments/results/plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{scenario_filename}_gantt_chart.png"

    plt.savefig(output_path)
    print(f"Schedule visualized and saved to: {output_path}")


if __name__ == "__main__":
    import sys

    scenario_arg = sys.argv[1] if len(sys.argv) > 1 else "morning_rush.yaml"

    # Grab the model prefix if provided, otherwise assume it matches the scenario
    model_arg = sys.argv[2] if len(sys.argv) > 2 else Path(scenario_arg).stem

    # Pass BOTH into main
    main(scenario_arg, model_arg)