"""
Decision-making policies for the airport gate assignment MDP.

Defines the abstract interface and concrete implementations for
selecting actions based on the current state.
"""
from __future__ import annotations
import random
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
        valid_actions = ActionSpace.get_valid_actions(
            state,
            env.active_flights,
            env.scenario.compatibility
        )
        return random.choice(valid_actions)


class GreedyPolicy:
    """
    A policy that greedily selects the assignment with the highest preference score.
    
    It filters for valid assignments first. If multiple assignments share the 
    highest preference score, it breaks ties randomly to aid exploration.
    If no assignments are possible, it returns NO_OP.
    """

    def get_action(self, state: AirportState, env: AirportEnvironment) -> Action:
        valid_actions = ActionSpace.get_valid_actions(
            state,
            env.active_flights,
            env.scenario.compatibility
        )

        # Filter out NO_OP to prioritize assignments
        assignment_actions = [a for a in valid_actions if not a.is_noop]

        # If no assignments are possible, we must wait
        if not assignment_actions:
            return NO_OP

        # Calculate scores for all possible assignments
        # We process this as a list of tuples (score, action) to avoid re-computation
        scored_actions = []
        for action in assignment_actions:
            score = self._get_score(action, env)
            scored_actions.append((score, action))

        # Find the maximum score
        max_score = max(scored_actions, key=lambda x: x[0])[0]

        # Identify all actions that share the maximum score (tie-breaking candidates)
        best_actions = [action for score, action in scored_actions if score == max_score]

        # Break ties randomly
        return random.choice(best_actions)

    def _get_score(self, action: Action, env: AirportEnvironment) -> float:
        """
        Helper to calculate preference score for a specific action.
        
        Args:
            action: The assignment action to evaluate.
            env: The environment context.
            
        Returns:
            The preference score from the compatibility config.
        """
        flight = env.active_flights[action.flight_id]
        try:
            # Use fast integer lookup
            ac_idx = env.scenario.compatibility.type_to_idx[flight.aircraft_type]
            return env.scenario.compatibility.get_preference_idx(ac_idx, action.gate_idx)
        except KeyError:
            # Fallback for configuration mismatches (should ideally not happen)
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
        valid_actions = ActionSpace.get_valid_actions(
            state,
            env.active_flights,
            env.scenario.compatibility
        )

        if not valid_actions:
            return NO_OP

        # 1. EXPLORATION: With probability epsilon, pick a random action
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # 2. EXPLOITATION: Pick the action that maximizes the Bellman equation
        # Grab the first action as a zero-cost fallback instead of calling random.choice()
        best_action = valid_actions[0]
        best_value = -float('inf')

        for action in valid_actions:
            # Lookahead using the lightweight forward model
            expected_next_state, expected_reward, done = env.simulate_action(state, action)

            if done:
                action_value = expected_reward
            else:
                # This call is now massively sped up by our caching trick above!
                phi_next = self.extractor.get_features(expected_next_state)
                action_value = expected_reward + self.gamma * self.vfa.predict(phi_next)

            # Update best action
            if action_value > best_value:
                best_value = action_value
                best_action = action

        return best_action
