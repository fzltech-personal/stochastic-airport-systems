"""
Experiment: Evaluate trained ADP Policy vs. Baselines.
Translates MDP rewards into real-world airport metrics (Total Delay Minutes).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.paths import ProjectPaths
from src.config.loader import ScenarioLoader
from src.mdp.environment import AirportEnvironment
from src.simulation.simulator import Simulator
from src.adp.features import PVFFeatureExtractor
from src.adp.value_function import LinearVFA
from src.adp.policies import ADPPolicy, RandomPolicy, GreedyPolicy
from attr import evolve


def run_evaluation(env, policy, num_episodes=10):
    simulator = Simulator(env, policy)
    total_rewards = []
    
    # Store trajectory metrics for the last episode to show the flow check
    last_trajectory = None

    for _ in range(num_episodes):
        trajectory = simulator.run_episode()
        last_trajectory = trajectory
        # Sum the rewards
        episode_reward = sum(r for _, _, r, _ in trajectory)
        total_rewards.append(episode_reward)

    return np.mean(total_rewards), np.std(total_rewards), last_trajectory


def main(scenario_filename: str, model_prefix: str, timestamp: str = None):
    config_path = ProjectPaths.get_configs_dir() / f"scenarios/{scenario_filename}"
    scenario_prefix = config_path.stem
    
    if not timestamp:
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")

    print(f"Loading Scenario '{scenario_prefix}' and generated schedule...")
    scenario = ScenarioLoader.from_yaml(config_path)

    # Load the schedule from the saved artifact
    json_path = ProjectPaths.get_data_dir() / f"schedules/synthetic/{scenario_prefix}_eval_schedule.json"
    if not json_path.exists():
        # Fallback to the processed directory if the artifact is not in schedules/synthetic
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

    # Total scheduled arrivals
    total_scheduled = sum(1 for f in env.scenario.schedule.get_flights() if f.direction == "arrival")

    print(f"\nRunning {num_test_episodes} evaluation episodes per policy...")
    for name, policy in policies.items():
        print(f"Testing {name}...")
        mean_reward, std_reward, last_trajectory = run_evaluation(env, policy, num_episodes=num_test_episodes)
        results.append({"Policy": name, "Mean Reward": mean_reward, "Std Dev": std_reward})
        
        # --- CONSERVATION OF FLOW CHECK ---
        gated_count = 0
        for state, action, reward, next_state in last_trajectory:
            # Check if action is NOT a NO_OP (it assigned a flight to a gate)
            if not getattr(action, 'is_noop', True) and action.flight_id is not None and action.gate_idx != -1:
                gated_count += 1
                
        # Look at the final state in the trajectory to see who is left waiting
        last_state = last_trajectory[-1][0]
        queue_count = len(last_state.runway_queue)
        
        print(f"  ↳ Flow Check: Scheduled ({total_scheduled}) = Gated ({gated_count}) + In Queue ({queue_count})")
        
        if (gated_count + queue_count) != total_scheduled:
            print(f"  !!! WARNING: Conservation of flow violated! Lost {total_scheduled - (gated_count + queue_count)} flights.")
        # -----------------------------------

    df = pd.DataFrame(results)
    print(f"\n=== FINAL EVALUATION RESULTS ({model_prefix.upper()}) ===")
    print(df.to_string(index=False))

    # Save to CSV with scenario prefix and timestamp
    output_dir = ProjectPaths.get_root() / f"experiments/results/{scenario_prefix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{timestamp}_{scenario_prefix}_evaluation_metrics.csv"

    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", nargs="?", default="morning_rush.yaml")
    parser.add_argument("model", nargs="?", default=None)
    parser.add_argument("--timestamp", type=str, default=None)
    
    args = parser.parse_args()
    
    scenario_filename = args.scenario
    model_prefix = args.model if args.model else Path(scenario_filename).stem
    
    main(scenario_filename, model_prefix, timestamp=args.timestamp)