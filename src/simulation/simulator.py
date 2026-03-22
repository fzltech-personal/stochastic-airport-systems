"""
Simulation Driver.

Runs Monte Carlo episodes by orchestrating the interaction between an 
AirportEnvironment and a Policy. Generates trajectories for offline learning.
"""
from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.mdp.state import AirportState
    from src.mdp.action import Action
    from src.mdp.environment import AirportEnvironment
    from src.adp.policies import BasePolicy


class Simulator:
    """
    Drives the execution of simulation episodes.
    
    Responsible for generating (state, action, reward, next_state) trajectories
    used in graph construction and TD learning.
    """

    def __init__(self, env: AirportEnvironment, policy: BasePolicy):
        """
        Initialize the simulator.

        Args:
            env: The AirportEnvironment instance.
            policy: The decision-making policy to execute.
        """
        self.env = env
        self.policy = policy

    def run_episode(self) -> List[Tuple[AirportState, Action, float, AirportState]]:
        """
        Run a single simulation episode from start to termination.

        Returns:
            A list of experience tuples (S_t, A_t, R_{t+1}, S_{t+1}).
            This format supports offline Temporal Difference (TD) learning.
        """
        trajectory: List[Tuple[AirportState, Action, float, AirportState]] = []
        
        # Initialize episode
        state = self.env.reset()
        done = False
        
        # Standard RL Loop
        while not done:
            # 1. Select Action
            action = self.policy.get_action(state, self.env)
            
            # 2. Step Environment
            next_state, reward, done, info = self.env.step(action)
            
            # 3. Record Experience (S_t, A_t, R_{t+1}, S_{t+1} or None if terminal)
            # Store None for terminal next_state so TD learner doesn't bootstrap past end.
            trajectory.append((state, action, reward, None if done else next_state))
            
            # 4. Advance State
            state = next_state
            
        return trajectory
