"""
Decision-making policies for the airport gate assignment MDP.

Defines the abstract interface and concrete implementations for
selecting actions based on the current state.
"""
from __future__ import annotations
import random
import numpy as np
from typing import Protocol, List, TYPE_CHECKING, Optional

from src.mdp.action import Action, ActionSpace, NO_OP

if TYPE_CHECKING:
    from src.mdp.state import AirportState
    from src.mdp.environment import AirportEnvironment
    from src.adp.features import PVFFeatureExtractor
    from src.adp.value_function import LinearVFA


class BasePolicy(Protocol):
    """
    Abstract interface for all decision-making policies.
    """

    def get_action(self, state: AirportState, env: AirportEnvironment) -> Action:
        """
        Select an action based on the current state and environment context.

        Args:
            state: The current immutable state.
            env: The environment instance (for accessing config/active_flights).

        Returns:
            The chosen Action.
        """
        ...


class RandomPolicy:
    """
    A policy that selects a valid action uniformly at random.
    Useful for exploration and baseline comparison.
    """

    def get_action(self, state: AirportState, env: AirportEnvironment) -> Action:
        valid_actions = env.get_valid_actions(state)
        return random.choice(valid_actions)


class GreedyPolicy:
    """
    A policy that greedily selects the assignment with the highest preference score.
    
    It filters for valid assignments first. If multiple assignments share the 
    highest preference score, it breaks ties randomly to aid exploration.
    If no assignments are possible, it returns NO_OP.
    """

    def get_action(self, state: AirportState, env: AirportEnvironment) -> Action:
        valid_actions = env.get_valid_actions(state)

        # Filter out NO_OP to prioritize assignments
        assignment_actions = [a for a in valid_actions if not a.is_noop]

        # If no assignments are possible, we must wait
        if not assignment_actions:
            return NO_OP

        # Calculate scores for all possible assignments
        scored_actions = []
        for action in assignment_actions:
            score = self._get_score(action, env)
            scored_actions.append((score, action))

        # Find the maximum score
        max_score = max(scored_actions, key=lambda x: x[0])[0]

        # Identify all actions that share the maximum score
        best_actions = [action for score, action in scored_actions if score == max_score]

        # Break ties randomly
        return random.choice(best_actions)

    def _get_score(self, action: Action, env: AirportEnvironment) -> float:
        """Helper to calculate preference score for a specific action."""
        flight = env.active_flights[action.flight_id]
        try:
            ac_idx = env.scenario.compatibility.type_to_idx[flight.aircraft_type]
            return env.scenario.compatibility.get_preference_idx(ac_idx, action.gate_idx)
        except KeyError:
            return -1.0


class ADPPolicy:
    """
    Approximate Dynamic Programming Policy.
    Uses a learned Value Function Approximator to make epsilon-greedy decisions.
    """

    def __init__(self, vfa: 'LinearVFA', extractor: 'PVFFeatureExtractor', epsilon: float = 0.2, gamma: float = 0.99):
        self.vfa = vfa
        self.extractor = extractor
        self.epsilon = epsilon
        self.gamma = gamma

    def get_action(self, state: 'AirportState', env: 'AirportEnvironment') -> Action:
        valid_actions = env.get_valid_actions(state)

        if not valid_actions:
            return NO_OP

        # 1. EXPLORATION SHORT-CIRCUIT: With probability epsilon, pick a random action immediately.
        if np.random.random() < self.epsilon:
            return random.choice(valid_actions)

        # 2. EXPLOITATION: Pick the action that maximizes the Bellman equation.
        # Short-circuit: if there is only one valid action there is nothing to compare.
        if len(valid_actions) == 1:
            return valid_actions[0]

        # Simulate all actions in one call (shared gate-tick + queue work done once).
        # Returns (resource_state, reward, done) tuples — no AirportState objects created.
        simulations = env.simulate_actions_batch(state, valid_actions)

        # Batch KNN over resource_states for all non-terminal next states (one call).
        non_terminal = [(i, rs) for i, (rs, _, done) in enumerate(simulations) if not done]
        phi_matrix = None
        if non_terminal:
            phi_matrix = self.extractor.extract_resource_states_batch(
                [rs for _, rs in non_terminal]
            )

        best_action = valid_actions[0]
        best_value = -float('inf')
        phi_row = 0

        for i, (action, (resource_state, expected_reward, done)) in enumerate(
            zip(valid_actions, simulations)
        ):
            if done:
                action_value = expected_reward
            else:
                action_value = expected_reward + self.gamma * self.vfa.predict(phi_matrix[phi_row])
                phi_row += 1

            if action_value > best_value:
                best_value = action_value
                best_action = action
        return best_action
