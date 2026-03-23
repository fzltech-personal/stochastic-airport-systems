"""
Experiment: Train the ADP Agent using TD(lambda) with eligibility traces.
"""
import sys
import csv
import time
from typing import Optional
from pathlib import Path
from datetime import datetime
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
from src.adp.agent import TDLambdaLearner, LSTDLearner


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


def main(scenario_filename: str, continue_training: bool = False, extra_epochs: int = 0):
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

    # ── Learner selection ────────────────────────────────────────────────────
    # Switch by changing LEARNER_TYPE. All learners share the BaseLearner
    # interface (learn_from_trajectory), so nothing else in this script changes.
    #
    #   "td_lambda"  — semi-gradient TD(λ), needs careful alpha tuning
    #   "lstd"       — LSTD(λ), solves exactly, no alpha needed
    LEARNER_TYPE = "lstd"

    if LEARNER_TYPE == "td_lambda":
        learner = TDLambdaLearner(vfa=vfa, extractor=extractor, gamma=0.99, alpha=0.001, lambda_=0.9)
    elif LEARNER_TYPE == "lstd":
        learner = LSTDLearner(vfa=vfa, extractor=extractor, gamma=0.99, lambda_=0.9, reg=1e-4)
    else:
        raise ValueError(f"Unknown LEARNER_TYPE: {LEARNER_TYPE!r}")

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

        # When continuing LSTD training, reset the accumulated A/b matrices so
        # the new run re-estimates the fixed point from fresh trajectories at
        # the current (lower) epsilon, rather than mixing in stale exploratory
        # data from the previous run. vfa.theta is kept as the warm start.
        if isinstance(learner, LSTDLearner):
            learner.reset_matrices()
            print("LSTD matrices reset — theta preserved, A/b start fresh.")

    # Open a new timestamped log file for this training run.
    # Each --continue starts a fresh file but resumes from start_episode,
    # so the episode column always reflects the true absolute episode number.
    log_dir = ProjectPaths.get_root() / "experiments/results/training_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
    log_path = log_dir / f"{timestamp}_{scenario_prefix}_training_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["episode", "duration_s", "reward", "moving_avg_10", "ema_reward", "epsilon"])
    print(f"Training log: {log_path}")

    # 5. Setup Environment and Simulator
    env = AirportEnvironment(scenario)
    eval_env = AirportEnvironment(scenario)
    simulator = Simulator(env, policy)

    # EMA state — alpha=0.05 gives a ~20-episode effective window, smoothing
    # enough to see trends without lagging too far behind structural shifts.
    _ema_alpha = 0.05

    def _compute_ema(rewards: list, alpha: float) -> list:
        result, val = [], None
        for r in rewards:
            val = r if val is None else alpha * r + (1 - alpha) * val
            result.append(val)
        return result

    # Seed EMA from loaded history so the continued curve stays continuous.
    ema_rewards: list = _compute_ema(episode_rewards, _ema_alpha)
    _ema_reward: Optional[float] = ema_rewards[-1] if ema_rewards else None

    # 6. Training Loop
    base_episodes = 1000
    num_episodes = start_episode + extra_epochs if extra_epochs > 0 else max(base_episodes, start_episode + 1)
    eval_interval = 50   # run evaluation every N training episodes
    eval_n = 30          # episodes per evaluation checkpoint

    print(f"Training ADP Agent on '{scenario_prefix}' from episode {start_episode} to {num_episodes}...")
    print(f"  learner={LEARNER_TYPE}, gamma=0.99, eval every {eval_interval} episodes")
    pbar = tqdm(range(start_episode, num_episodes), initial=start_episode, total=num_episodes)

    try:
        for ep_idx in pbar:
            ep_start = time.perf_counter()
            trajectory = simulator.run_episode()

            learner.learn_from_trajectory(trajectory)
            policy.epsilon = max(0.05, policy.epsilon * 0.995)

            total_reward = sum(r for _, _, r, _ in trajectory)
            episode_rewards.append(total_reward)
            duration = time.perf_counter() - ep_start

            window_rewards = episode_rewards[-10:]
            moving_avg = sum(window_rewards) / len(window_rewards)

            _ema_reward = total_reward if _ema_reward is None else (
                _ema_alpha * total_reward + (1 - _ema_alpha) * _ema_reward
            )
            ema_rewards.append(_ema_reward)

            log_writer.writerow([ep_idx + 1, f"{duration:.3f}", f"{total_reward:.1f}",
                                  f"{moving_avg:.1f}", f"{_ema_reward:.1f}", f"{policy.epsilon:.4f}"])
            log_file.flush()

            pbar.set_postfix({
                "Reward": f"{total_reward:.0f}",
                "Eps": f"{policy.epsilon:.3f}",
                "s/ep": f"{duration:.1f}",
            })

            # Periodic evaluation checkpoint (epsilon=0, clean signal)
            if (ep_idx + 1) % eval_interval == 0:
                eval_rewards = run_eval(eval_env, policy, n_episodes=eval_n, epsilon_override=0.0)
                eval_history.append((ep_idx + 1, float(np.mean(eval_rewards))))
                np.save(eval_path, np.array(eval_history, dtype=object))

            # Checkpoint weights + training curve
            np.save(theta_path, vfa.theta)
            np.save(rewards_path, np.array(episode_rewards))

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving and plotting...")
    finally:
        log_file.close()

    # 7. Plot: training curve + eval overlay
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(episode_rewards, alpha=0.2, color='blue', linewidth=0.8, label="Training reward")

    window = 20
    if len(episode_rewards) >= window:
        ma = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(episode_rewards)), ma, color='blue', linewidth=1.5,
                label=f"{window}-ep moving avg")

    if ema_rewards:
        ax.plot(ema_rewards, color='green', linewidth=1.5, label=f"EMA (α={_ema_alpha})")

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
    import argparse as _argparse
    _parser = _argparse.ArgumentParser()
    _parser.add_argument("scenario", nargs="?", default="morning_rush.yaml")
    _parser.add_argument("-c", "--continue-training", action="store_true")
    _parser.add_argument("--extra-epochs", type=int, default=0,
                         help="Train for this many additional episodes beyond the current checkpoint.")
    _args = _parser.parse_args()
    main(_args.scenario, continue_training=_args.continue_training, extra_epochs=_args.extra_epochs)
