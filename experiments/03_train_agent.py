"""
Experiment: Train the ADP Agent using TD(lambda) with eligibility traces.
"""
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from attr import evolve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.paths import ProjectPaths
from src.config.loader import ScenarioLoader
from src.mdp.components.schedule_generator import ScheduleGenerator
from src.mdp.environment import AirportEnvironment
from src.adp.policies import RandomPolicy, ADPPolicy
from src.simulation.simulator import Simulator
from src.adp.features import PVFFeatureExtractor
from src.adp.value_function import LinearVFA
from src.adp.agent import TDLambdaLearner


def run_eval(env, policy, n_episodes: int = 10, epsilon_override: float = 0.0) -> list:
    """Run evaluation episodes with a fixed epsilon, return list of total rewards."""
    saved_eps = policy.epsilon
    policy.epsilon = epsilon_override
    rewards = []
    for _ in range(n_episodes):
        traj = Simulator(env, policy).run_episode()
        rewards.append(sum(r for _, _, r, _ in traj))
    policy.epsilon = saved_eps
    return rewards


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
    eval_path = data_dir / f"{scenario_prefix}_eval_history.npy"

    if not basis_path.exists():
        print(f"Error: PVF data for '{scenario_prefix}' not found!")
        print(f"Run 01_generate_features.py with '{scenario_filename}' first.")
        return

    # 3. Initialize ADP Components
    extractor = PVFFeatureExtractor(str(basis_path), str(mapping_path))
    vfa = LinearVFA(num_features=extractor.num_features)
    learner = TDLambdaLearner(vfa=vfa, extractor=extractor, gamma=0.99, alpha=0.001, lambda_=0.9)
    policy = ADPPolicy(vfa=vfa, extractor=extractor, epsilon=0.5, gamma=0.99)

    # 4. Checkpointing Logic
    start_episode = 0
    episode_rewards = []
    eval_history = []   # list of (episode_idx, mean_eval_reward)

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

            initial_epsilon = 0.5
            decay_rate = 0.995
            policy.epsilon = max(0.05, initial_epsilon * (decay_rate ** start_episode))
            print(f"Resuming from episode {start_episode}. New epsilon: {policy.epsilon:.4f}")
        else:
            print("No reward history found. Starting from scratch.")

        if eval_path.exists():
            eval_history = np.load(eval_path, allow_pickle=True).tolist()

    # 5. Setup Environment and Simulator
    env = AirportEnvironment(scenario)
    eval_env = AirportEnvironment(scenario)
    simulator = Simulator(env, policy)

    # Per-step running reward statistics for normalization
    # Maintained as (count, mean, M2) for Welford online algorithm
    _rn_count = 0
    _rn_mean = 0.0
    _rn_M2 = 0.0

    def update_reward_stats(r: float):
        nonlocal _rn_count, _rn_mean, _rn_M2
        _rn_count += 1
        delta = r - _rn_mean
        _rn_mean += delta / _rn_count
        _rn_M2 += delta * (r - _rn_mean)

    def normalized_reward(r: float) -> float:
        """Normalize reward to zero mean, unit std using running statistics."""
        if _rn_count < 2:
            return r
        std = (_rn_M2 / (_rn_count - 1)) ** 0.5
        return (r - _rn_mean) / (std + 1e-8)

    # 6. Training Loop
    num_episodes = 1000
    eval_interval = 50   # run evaluation every N training episodes
    eval_n = 10          # episodes per evaluation checkpoint

    print(f"Training ADP Agent on '{scenario_prefix}' from episode {start_episode} to {num_episodes}...")
    print(f"  lambda=0.9, alpha=0.001, gamma=0.99, eval every {eval_interval} episodes")
    pbar = tqdm(range(start_episode, num_episodes), initial=start_episode, total=num_episodes)

    try:
        for ep_idx in pbar:
            trajectory = simulator.run_episode()

            # Update running reward statistics and build normalized trajectory
            normalized_traj = []
            for s, a, r, ns in trajectory:
                update_reward_stats(r)
                normalized_traj.append((s, a, normalized_reward(r), ns))

            learner.learn_from_trajectory(normalized_traj)
            policy.epsilon = max(0.05, policy.epsilon * 0.995)

            total_reward = sum(r for _, _, r, _ in trajectory)
            episode_rewards.append(total_reward)

            pbar.set_postfix({
                "Reward": f"{total_reward:.0f}",
                "Eps": f"{policy.epsilon:.3f}",
            })

            # Periodic evaluation checkpoint (epsilon=0, clean signal)
            abs_ep = ep_idx + 1
            if abs_ep % eval_interval == 0:
                eval_rewards = run_eval(eval_env, policy, n_episodes=eval_n, epsilon_override=0.0)
                eval_history.append((abs_ep, float(np.mean(eval_rewards))))
                np.save(eval_path, np.array(eval_history, dtype=object))

            # Checkpoint weights + training curve
            np.save(theta_path, vfa.theta)
            np.save(rewards_path, np.array(episode_rewards))

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving and plotting...")

    # 7. Plot: training curve + eval overlay
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(episode_rewards, alpha=0.2, color='blue', linewidth=0.8, label="Training reward")

    window = 20
    if len(episode_rewards) >= window:
        ma = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(episode_rewards)), ma, color='blue', linewidth=1.5,
                label=f"{window}-ep moving avg")

    if eval_history:
        eval_eps = [e for e, _ in eval_history]
        eval_means = [m for _, m in eval_history]
        ax.plot(eval_eps, eval_means, 'o-', color='red', linewidth=2, markersize=5,
                label=f"Eval mean (eps=0, {eval_n} ep)")

        # t-test: first half vs second half of eval points
        if len(eval_means) >= 4:
            from scipy.stats import ttest_ind
            half = len(eval_means) // 2
            stat, pval = ttest_ind(eval_means[half:], eval_means[:half])
            ax.set_title(
                f"ADP Training — {scenario_prefix}  |  "
                f"Eval improvement t-test: p={pval:.3f} ({'significant' if pval < 0.05 else 'not significant'})"
            )
        else:
            ax.set_title(f"ADP Training — {scenario_prefix}")
    else:
        ax.set_title(f"ADP Training — {scenario_prefix}")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_dir = ProjectPaths.get_root() / "experiments/results/plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{scenario_prefix}_training_curve.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Training curve saved to {out_path}")
    print(f"Final weights saved to {theta_path}")

    if eval_history:
        eval_means = [m for _, m in eval_history]
        print(f"Eval rewards: first checkpoint={eval_means[0]:.0f}, last={eval_means[-1]:.0f}")
        if len(eval_means) >= 4:
            from scipy.stats import ttest_ind
            half = len(eval_means) // 2
            _, pval = ttest_ind(eval_means[half:], eval_means[:half])
            print(f"t-test (later half vs earlier half): p={pval:.4f}")


if __name__ == "__main__":
    scenario_arg = "morning_rush.yaml"
    continue_flag = False

    if len(sys.argv) > 1:
        if '-c' in sys.argv or '--continue' in sys.argv:
            continue_flag = True
            if '-c' in sys.argv: sys.argv.remove('-c')
            if '--continue' in sys.argv: sys.argv.remove('--continue')

        if len(sys.argv) > 1:
            scenario_arg = sys.argv[1]

    main(scenario_arg, continue_training=continue_flag)
