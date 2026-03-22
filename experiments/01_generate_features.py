"""
Pipeline Step 1: Generate and Save Proto-Value Functions.

This script runs the random exploration, builds the state-transition graph,
computes the spectral basis functions, and saves the artifacts to disk
for downstream training and visualization.
"""
import sys
import pickle
import argparse
from pathlib import Path
import networkx as nx
import numpy as np
from tqdm import tqdm
from attr import evolve

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.paths import ProjectPaths
from src.config.loader import ScenarioLoader
from src.mdp.components.schedule_generator import ScheduleGenerator
from src.mdp.environment import AirportEnvironment
from src.adp.policies import RandomPolicy
from src.simulation.simulator import Simulator
from src.representation.graph_builder import StateGraph
from src.representation.spectral import PVFCreator


def main(scenario_filename: str, force_rebuild: bool = False):
    config_path = ProjectPaths.get_configs_dir() / f"scenarios/{scenario_filename}"
    scenario_prefix = config_path.stem

    # --- CACHE CHECK ---
    data_dir = ProjectPaths.get_data_dir() / "processed"
    pvf_path = data_dir / f"{scenario_prefix}_basis_functions.npy"
    
    if not force_rebuild and pvf_path.exists():
        print(f"⚡ Cache hit! Graph and PVF artifacts already exist for '{scenario_prefix}'.")
        print("Skipping generation. Run with --force to rebuild.")
        return
    # -------------------

    scenario = ScenarioLoader.from_yaml(config_path)

    print(f"Generating synthetic schedule for: {scenario_prefix}...")
    rng = np.random.default_rng(42)  # Use a fixed seed for reproducible graphs
    flights = ScheduleGenerator.generate(
        scenario_name=scenario.schedule.scenario_name,
        num_flights=scenario.schedule.num_flights,
        generation_params=scenario.schedule.generation_params,
        num_runways=scenario.airport.num_runways,
        aircraft_types=scenario.aircraft_types,
        rng=rng
    )

    new_schedule = evolve(
        scenario.schedule,
        flights=flights,
        generation_params=None,
        schedule_file=None,
        num_flights=len(flights)
    )
    scenario = evolve(scenario, schedule=new_schedule)

    # 2. Setup Environment
    env = AirportEnvironment(scenario)
    simulator = Simulator(env, RandomPolicy())
    graph_builder = StateGraph()

    # 3. Collect Trajectories
    num_episodes = 100
    print(f"Running {num_episodes} episodes for graph construction...")
    for _ in tqdm(range(num_episodes)):
        trajectory = simulator.run_episode()
        graph_builder.add_trajectory(trajectory)

    print(f"Graph built: {graph_builder.num_nodes} nodes.")

    # 4. Extract Largest Connected Component
    G = graph_builder.graph

    components = list(nx.connected_components(G))
    print(f"  [Graph] Total nodes: {G.number_of_nodes()}")
    print(f"  [Graph] Number of disconnected islands: {len(components)}")

    # Extract only the biggest island
    largest_cc = max(components, key=len)
    G_sub = G.subgraph(largest_cc).copy()
    print(f"  [Graph] Largest component size: {G_sub.number_of_nodes()} nodes.")

    if G_sub.number_of_nodes() < 100:
        print("!!! ERROR: Your simulation isn't finding enough recurring states.")
        print("Your graph is just a bunch of disconnected lines. We can't do math on this.")
        return

    sub_nodes = list(G_sub.nodes())
    sub_adj = nx.to_scipy_sparse_array(G_sub, nodelist=sub_nodes, weight='weight', format='csr')

    # 5. Compute Basis Functions
    num_features = min(20, G_sub.number_of_nodes() - 2)
    if num_features < 1:
        print("Graph too small to compute features. Exiting.")
        return

    print("Computing Proto-Value Functions...")
    basis_functions = PVFCreator.compute_basis(sub_adj, num_features)

    # 6. Save Artifacts with Scenario Prefix!
    data_dir.mkdir(parents=True, exist_ok=True)

    np.save(data_dir / f"{scenario_prefix}_basis_functions.npy", basis_functions)

    with open(data_dir / f"{scenario_prefix}_state_mapping.pkl", "wb") as f:
        pickle.dump(sub_nodes, f)

    # Optional: Save the graph itself for the visualizer
    if hasattr(nx, 'write_gpickle'):
        nx.write_gpickle(G_sub, data_dir / f"{scenario_prefix}_state_graph.gpickle")
    else:
        pickle.dump(G_sub, open(data_dir / f"{scenario_prefix}_state_graph.pkl", "wb"))

    print(f"SUCCESS: Pipeline Step 1 complete. Artifacts saved to {data_dir} with prefix '{scenario_prefix}_'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Proto-Value Functions for a scenario.")
    parser.add_argument(
        "scenario", 
        nargs="?", 
        default="morning_rush.yaml", 
        help="The scenario YAML file to use (default: morning_rush.yaml)"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force rebuild of the graph and PVFs even if cached artifacts exist."
    )
    
    args = parser.parse_args()
    main(args.scenario, force_rebuild=args.force)