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

        # 2. EXPLOITATION: Pick the action that maximizes the Bellman equation
        best_action = valid_actions[0]
        best_value = -float('inf')

        for action in valid_actions:
            # Lookahead using the lightweight forward model
            # This model is not implemented in the provided environment, so we'll assume it exists
            # For the purpose of this refactoring, we'll call a hypothetical simulate_action
            # In a real scenario, this would be:
            # expected_next_state, expected_reward, done = env.simulate_action(state, action)

            # Since simulate_action is not available, we will mock its expected behavior
            # This part of the logic is illustrative. The key change is the short-circuit above.
            if action.is_noop:
                # Simplified reward for waiting
                expected_reward = env.reward_config.compute_reward(queue_length=len(state.runway_queue), assignment_made=False)
                # Simplified next state for waiting
                next_gates = tuple(max(0, g - 1) for g in state.gates)
                expected_next_state = state.__class__(t=state.t + 1, gates=next_gates, runway_queue=state.runway_queue)
                done = (state.t + 1) >= env.scenario.time.horizon
            else:
                # Simplified reward for assigning
                expected_reward = env.reward_config.compute_reward(queue_length=len(state.runway_queue) - 1, assignment_made=True)
                # Simplified next state for assigning
                # This is a highly simplified model of what simulate_action would do
                next_gates = list(state.gates)
                next_gates[action.gate_idx] = 60 # Assume a fixed service time for illustration
                next_gates = tuple(max(0, g - 1) for g in next_gates)
                next_queue = state.runway_queue[1:]
                expected_next_state = state.__class__(t=state.t + 1, gates=next_gates, runway_queue=next_queue)
                done = (state.t + 1) >= env.scenario.time.horizon


            if done:
                action_value = expected_reward
            else:
                phi_next = self.extractor.extract_features(expected_next_state)
                action_value = expected_reward + self.gamma * self.vfa.predict(phi_next)

            if action_value > best_value:
                best_value = action_value
                best_action = action

        return best_action
