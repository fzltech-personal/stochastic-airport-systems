"""
Experiment: Train the ADP Agent using TD(0) Learning.
"""
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from attr import evolve

# Add project root to path robustly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.paths import ProjectPaths
from src.config.loader import ScenarioLoader
from src.mdp.components.schedule_generator import ScheduleGenerator
from src.mdp.environment import AirportEnvironment
from src.adp.policies import RandomPolicy, ADPPolicy
from src.simulation.simulator import Simulator
from src.adp.features import PVFFeatureExtractor
from src.adp.value_function import LinearVFA
from src.adp.agent import TD0Learner


def main():
    # 1. Load configuration and generate schedule
    config_path = ProjectPaths.get_configs_dir() / "scenarios/morning_rush.yaml"
    scenario = ScenarioLoader.from_yaml(config_path)

    rng = np.random.default_rng(42)
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

    # 2. Setup Data Paths
    data_dir = ProjectPaths.get_data_dir() / "processed"
    basis_path = data_dir / "basis_functions.npy"
    mapping_path = data_dir / "state_mapping.pkl"

    if not basis_path.exists():
        print("Error: PVF data not found! Run 01_generate_features.py first.")
        return

    # 3. Initialize ADP Components
    extractor = PVFFeatureExtractor(str(basis_path), str(mapping_path))
    vfa = LinearVFA(num_features=extractor.num_features)

    # Hyperparameters: Gamma = Discount factor, Alpha = Learning rate
    learner = TD0Learner(vfa=vfa, extractor=extractor, gamma=0.99, alpha=0.01)

    # 4. Setup Environment and Simulator
    env = AirportEnvironment(scenario)
    # policy = RandomPolicy()  # Train using random policy to explore everything
    policy = ADPPolicy(vfa=vfa, extractor=extractor, epsilon=0.5, gamma=0.99)
    simulator = Simulator(env, policy)

    # 5. Training Loop
    num_episodes = 100
    episode_rewards = []

    print(f"Training ADP Agent for {num_episodes} episodes...")
    for episode in tqdm(range(num_episodes)):
        # The environment must be reset under the hood by simulator.run_episode()
        trajectory = simulator.run_episode()

        # The agent learns from the experience!
        learner.learn_from_trajectory(trajectory)

        policy.epsilon = max(0.01, policy.epsilon * 0.95)

        # Track total reward for the episode
        total_reward = sum(reward for _, _, reward, _ in trajectory)
        episode_rewards.append(total_reward)

    # 6. Plot Learning Curve
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label="Total Reward per Episode", alpha=0.3, color='blue')

    # Plot moving average
    window = max(1, num_episodes // 10)
    moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
    plt.plot(range(window - 1, len(episode_rewards)), moving_avg, color='red', linewidth=2,
             label=f"{window}-Episode Moving Avg")

    plt.title("ADP Training Progress (TD(0) with Proto-Value Functions)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_dir = ProjectPaths.get_root() / "experiments/results/plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "training_curve.png"
    plt.savefig(out_path)
    print(f"Training complete! Curve saved to {out_path}")

    # Save the learned weights!
    np.save(data_dir / "learned_theta.npy", vfa.theta)
    print("Learned weights saved.")


if __name__ == "__main__":
    main()