"""
Experiment 01b: Visualize the Input Runway Schedule.
Generates a Gantt chart of the raw demand before the MDP processes it.
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
from src.mdp.components.schedule_generator import ScheduleGenerator


def main(scenario_filename: str, timestamp: str = None):
    config_path = ProjectPaths.get_configs_dir() / f"scenarios/{scenario_filename}"
    scenario_prefix = config_path.stem
    
    if not timestamp:
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")

    print(f"Loading Scenario '{scenario_prefix}' for runway visualization...")
    scenario = ScenarioLoader.from_yaml(config_path)

    # Use the EXACT same seed as the evaluation scripts (04 and 05)
    # so the runway plot perfectly matches the final Gate Gantt chart!
    rng = np.random.default_rng(999)

    flights = ScheduleGenerator.generate(
        scenario_name=scenario.schedule.scenario_name,
        num_flights=scenario.schedule.num_flights,
        generation_params=scenario.schedule.generation_params,
        num_runways=scenario.airport.num_runways,
        aircraft_types=scenario.aircraft_types,
        rng=rng
    )

    print(f"Generated {len(flights)} total runway movements.")

    if not flights:
        print("No flights generated! Check your time_window or generation_params.")
        return

    # Save the generated schedule as an artifact for reproducible evaluation
    json_path = ProjectPaths.get_data_dir() / f"schedules/synthetic/{scenario_prefix}_eval_schedule.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    ScheduleGenerator.save_schedule(flights, json_path)
    print(f"Saved evaluation schedule to {json_path}")

    # Get occupancy time for block width
    occ_time = scenario.schedule.generation_params.get('runway_occupancy_time', 2)

    # --- PLOT THE RUNWAY GANTT CHART ---
    print("Plotting Runway Gantt Chart...")

    unique_types = list(set(f.aircraft_type for f in flights))
    color_palette = list(TABLEAU_COLORS.values())
    type_colors = {ac_type: color_palette[i % len(color_palette)] for i, ac_type in enumerate(unique_types)}

    num_runways = scenario.airport.num_runways
    fig, ax = plt.subplots(figsize=(16, max(4, num_runways * 1.5)))

    for flight in flights:
        runway = flight.runway
        start = flight.scheduled_time
        color = type_colors[flight.aircraft_type]

        # Draw the block for runway occupancy
        ax.broken_barh([(start, occ_time)], (runway - 0.4, 0.8), facecolors=color, edgecolor='black', alpha=0.8)

        # Optional: Add an 'A' for Arrival and 'D' for Departure text
        label = "A" if flight.direction == "arrival" else "D"
        ax.text(start + occ_time / 2, runway, label,
                ha='center', va='center', color='white', fontsize=7, fontweight='bold', clip_on=True)

    # Formatting
    ax.set_yticks(range(num_runways))
    ax.set_yticklabels([f"Runway {r}" for r in range(num_runways)])
    ax.set_xlabel("Time (Minutes from start of simulation)")
    ax.set_title(f"Raw Runway Demand ({scenario_prefix.upper()})\nBefore MDP Gate Assignment")
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)

    # Legend
    legend_patches = [mpatches.Patch(color=color, label=ac_type) for ac_type, color in type_colors.items()]
    ax.legend(handles=legend_patches, title="Aircraft Type", loc="upper right")

    plt.tight_layout()

    # Save output with timestamp
    output_dir = ProjectPaths.get_root() / f"experiments/results/plots/{scenario_prefix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{timestamp}_{scenario_prefix}_runway_schedule.png"

    plt.savefig(output_path)
    print(f"Runway schedule visualized and saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", nargs="?", default="morning_rush.yaml")
    parser.add_argument("--timestamp", type=str, default=None)
    
    args = parser.parse_args()
    main(args.scenario, timestamp=args.timestamp)