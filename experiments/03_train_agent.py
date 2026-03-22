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


def main(scenario_filename: str, continue_training: bool = False):
    # 1. Load configuration and generate schedule
    config_path = ProjectPaths.get_configs_dir() / f"scenarios/{scenario_filename}"
    scenario_prefix = config_path.stem

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
    basis_path = data_dir / f"{scenario_prefix}_basis_functions.npy"
    mapping_path = data_dir / f"{scenario_prefix}_state_mapping.pkl"
    theta_path = data_dir / f"{scenario_prefix}_learned_theta.npy"
    rewards_path = data_dir / f"{scenario_prefix}_reward_history.npy"

    if not basis_path.exists():
        print(f"Error: PVF data for '{scenario_prefix}' not found!")
        print(f"Run 01_generate_features.py with '{scenario_filename}' first.")
        return

    # 3. Initialize ADP Components
    extractor = PVFFeatureExtractor(str(basis_path), str(mapping_path))
    vfa = LinearVFA(num_features=extractor.num_features)
    learner = TD0Learner(vfa=vfa, extractor=extractor, gamma=0.99, alpha=0.001)
    policy = ADPPolicy(vfa=vfa, extractor=extractor, epsilon=0.5, gamma=0.99)
    
    # 4. Checkpointing Logic
    start_episode = 0
    episode_rewards = []
    if continue_training:
        print("--- Attempting to continue training ---")
        if theta_path.exists():
            print(f"Loading learned weights from: {theta_path}")
            loaded_theta = np.load(theta_path)
            if loaded_theta.shape != vfa.theta.shape:
                print(f"WARNING: saved theta shape {loaded_theta.shape} != current {vfa.theta.shape}. Starting fresh.")
            else:
                vfa.theta = loaded_theta
        
        if rewards_path.exists():
            print(f"Loading reward history from: {rewards_path}")
            episode_rewards = list(np.load(rewards_path))
            start_episode = len(episode_rewards)
            
            # Recalculate epsilon based on the number of completed episodes
            # This ensures the exploration rate continues its decay schedule
            initial_epsilon = 0.5
            decay_rate = 0.995
            policy.epsilon = max(0.05, initial_epsilon * (decay_rate ** start_episode))
            print(f"Resuming from episode {start_episode}. New epsilon: {policy.epsilon:.4f}")
        else:
            print("No reward history found. Starting from scratch.")
    
    # 5. Setup Environment and Simulator
    env = AirportEnvironment(scenario)
    simulator = Simulator(env, policy)

    # 6. Training Loop
    num_episodes = 1000
    print(f"Training ADP Agent on '{scenario_prefix}' from episode {start_episode} to {num_episodes}...")
    pbar = tqdm(range(start_episode, num_episodes), initial=start_episode, total=num_episodes)

    try:
        for _ in pbar:
            trajectory = simulator.run_episode()
            learner.learn_from_trajectory(trajectory)
            policy.epsilon = max(0.05, policy.epsilon * 0.995)

            total_reward = sum(reward for _, _, reward, _ in trajectory)
            episode_rewards.append(total_reward)

            pbar.set_postfix({
                "Reward": f"{total_reward:.1f}",
                "Epsilon": f"{policy.epsilon:.3f}"
            })
            
            # --- SAVE CHECKPOINT AT THE END OF EACH EPISODE ---
            np.save(theta_path, vfa.theta)
            np.save(rewards_path, np.array(episode_rewards))
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user! Gracefully exiting and generating training curve...")

    # 7. Plot Learning Curve
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label="Total Reward per Episode", alpha=0.3, color='blue')

    window = 10
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(episode_rewards)), moving_avg, color='red', linewidth=2,
                 label=f"{window}-Episode Moving Avg")

    plt.title(f"ADP Training Progress - {scenario_prefix}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_dir = ProjectPaths.get_root() / "experiments/results/plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{scenario_prefix}_training_curve.png"
    plt.savefig(out_path)
    print(f"Training complete! Curve saved to {out_path}")
    print(f"Final learned weights saved to {theta_path}.")


if __name__ == "__main__":
    import sys

    scenario_arg = "morning_rush.yaml"
    continue_flag = False

    if len(sys.argv) > 1:
        # Check for flags before positional arguments
        if '-c' in sys.argv or '--continue' in sys.argv:
            continue_flag = True
            # Remove the flag to not confuse the scenario argument parsing
            if '-c' in sys.argv: sys.argv.remove('-c')
            if '--continue' in sys.argv: sys.argv.remove('--continue')

        # The remaining argument should be the scenario file
        if len(sys.argv) > 1:
            scenario_arg = sys.argv[1]

    main(scenario_arg, continue_training=continue_flag)