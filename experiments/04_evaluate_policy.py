"""
Experiment: Evaluate trained ADP Policy vs. Baselines.
Translates MDP rewards into real-world airport metrics (Total Delay Minutes).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.paths import ProjectPaths
from src.config.loader import ScenarioLoader
from src.mdp.components.schedule_generator import ScheduleGenerator
from src.mdp.environment import AirportEnvironment
from src.simulation.simulator import Simulator
from src.adp.features import PVFFeatureExtractor
from src.adp.value_function import LinearVFA
from src.adp.policies import ADPPolicy, RandomPolicy, GreedyPolicy


def run_evaluation(env, policy, num_episodes=10):
    simulator = Simulator(env, policy)
    total_rewards = []

    for _ in range(num_episodes):
        trajectory = simulator.run_episode()
        # Sum the rewards
        episode_reward = sum(r for _, _, r, _ in trajectory)
        total_rewards.append(episode_reward)

    return np.mean(total_rewards), np.std(total_rewards)


def main(scenario_filename: str, model_prefix: str):
    # --- EASILY SWITCH SCENARIOS HERE ---
    # scenario_filename = "greedy_trap.yaml" # Change to "greedy_trap.yaml" when ready!
    # ------------------------------------

    config_path = ProjectPaths.get_configs_dir() / f"scenarios/{scenario_filename}"
    scenario_prefix = config_path.stem

    print(f"Loading Scenario '{scenario_prefix}' and generated schedule...")
    scenario = ScenarioLoader.from_yaml(config_path)

    # Use a fixed seed for the schedule so all policies face the exact same traffic
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

    print(f"Loading Trained ADP Agent for '{model_prefix}'...")
    data_dir = ProjectPaths.get_data_dir() / "processed"

    basis_path = data_dir / f"{model_prefix}_basis_functions.npy"
    mapping_path = data_dir / f"{model_prefix}_state_mapping.pkl"
    theta_path = data_dir / f"{model_prefix}_learned_theta.npy"

    if not theta_path.exists():
        print(f"!!! ERROR: Learned weights for '{model_prefix}' not found.")
        print(f"Please run 03_train_agent.py with '{scenario_filename}' first!")
        return

    extractor = PVFFeatureExtractor(str(basis_path), str(mapping_path))
    vfa = LinearVFA(num_features=extractor.num_features)

    # Load the brain you just trained!
    vfa.theta = np.load(theta_path)

    # Set epsilon to 0.0 (Pure Exploitation - no random moves!)
    adp_policy = ADPPolicy(vfa=vfa, extractor=extractor, epsilon=0.0, gamma=0.99)

    policies = {
        "Random (Baseline)": RandomPolicy(),
        "Greedy (Myopic)": GreedyPolicy(),
        "ADP (Trained AI)": adp_policy
    }

    results = []
    num_test_episodes = 10

    print(f"\nRunning {num_test_episodes} evaluation episodes per policy...")
    for name, policy in policies.items():
        print(f"Testing {name}...")
        mean_reward, std_reward = run_evaluation(env, policy, num_episodes=num_test_episodes)
        results.append({"Policy": name, "Mean Reward": mean_reward, "Std Dev": std_reward})

    df = pd.DataFrame(results)
    print(f"\n=== FINAL EVALUATION RESULTS ({model_prefix.upper()}) ===")
    print(df.to_string(index=False))

    # Save to CSV with scenario prefix
    output_dir = ProjectPaths.get_root() / "experiments/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{scenario_prefix}_evaluation_metrics.csv"

    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")


if __name__ == "__main__":
    import sys

    scenario_arg = sys.argv[1] if len(sys.argv) > 1 else "morning_rush.yaml"

    # Grab the model prefix if provided, otherwise assume it matches the scenario
    model_arg = sys.argv[2] if len(sys.argv) > 2 else Path(scenario_arg).stem

    # Pass BOTH into main
    main(scenario_arg, model_arg)