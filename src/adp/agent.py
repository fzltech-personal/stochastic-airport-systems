from typing import List, Tuple, Any, Optional

import numpy as np

from src.adp.features import PVFFeatureExtractor
from src.adp.value_function import LinearVFA

class TD0Learner:
    """
    TD(0) Learning algorithm orchestrator for an episode trajectory.
    """

    def __init__(self, vfa: LinearVFA, extractor: PVFFeatureExtractor, gamma: float, alpha: float) -> None:
        """
        Initialize the TD(0) learner.

        Args:
            vfa (LinearVFA): The linear value function approximator.
            extractor (PVFFeatureExtractor): The PVF Feature Extractor.
            gamma (float): Discount factor [0, 1].
            alpha (float): Learning rate.
        """
        self.vfa: LinearVFA = vfa
        self.extractor: PVFFeatureExtractor = extractor
        self.gamma: float = gamma
        self.alpha: float = alpha

    def learn_from_trajectory(self, trajectory: List[Tuple[Any, Any, float, Optional[Any]]]) -> None:
        """
        Iterate through an episode's trajectory and perform TD(0) updates.

        Args:
            trajectory (List[Tuple]): A list of transitions, where each transition is a tuple 
                                      of (state, action, reward, next_state). `next_state` 
                                      can be None if the transition leads to a terminal state.
        """
        for state, action, reward, next_state in trajectory:
            # Extract features for the current state
            phi: np.ndarray = self.extractor.get_features(state)

            if next_state is None:
                # Terminal state: TD Target is just the reward
                target: float = reward
            else:
                # Next state features and predicted value
                next_phi: np.ndarray = self.extractor.get_features(next_state)
                # Calculate TD Target
                target: float = reward + self.gamma * self.vfa.predict(next_phi)

            # Perform standard semi-gradient TD update on the VFA
            self.vfa.update(phi, target, self.alpha)
