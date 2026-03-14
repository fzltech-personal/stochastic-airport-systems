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


def main():
    print("Loading Scenario and generated schedule...")
    config_path = ProjectPaths.get_configs_dir() / "scenarios/morning_rush.yaml"
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
    scenario = evolve(scenario, schedule=evolve(scenario.schedule, flights=flights))
    env = AirportEnvironment(scenario)

    print("Loading Trained ADP Agent...")
    data_dir = ProjectPaths.get_data_dir() / "processed"
    extractor = PVFFeatureExtractor(str(data_dir / "basis_functions.npy"), str(data_dir / "state_mapping.pkl"))

    vfa = LinearVFA(num_features=extractor.num_features)
    # Load the brain you just trained!
    vfa.theta = np.load(data_dir / "learned_theta.npy")

    # Set epsilon to 0.0 (Pure Exploitation - no random moves!)
    adp_policy = ADPPolicy(vfa=vfa, extractor=extractor, epsilon=0.0)

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
    print("\n=== FINAL EVALUATION RESULTS ===")
    print(df.to_string(index=False))

    # Save to CSV
    df.to_csv(ProjectPaths.get_root() / "experiments/results/evaluation_metrics.csv", index=False)


if __name__ == "__main__":
    main()