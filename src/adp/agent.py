from typing import List, Tuple, Any, Optional

import numpy as np

from src.adp.features import PVFFeatureExtractor
from src.adp.value_function import LinearVFA

class TD0Learner:
    """
    Monte Carlo value learner for episodic airport gate assignment.

    Uses actual discounted returns G_t as learning targets rather than
    bootstrapped TD(0) targets (r + gamma * V(s')).

    Why MC over forward TD(0):
      - Episodes are 720 steps long with gamma=0.99.
      - With forward TD(0), each target = r_t + gamma * V(s_{t+1}) where V ≈ 0
        initially. All targets collapse to the 1-step reward (~-4.6), so the VFA
        learns to predict -4.6 everywhere — useless for distinguishing good vs bad
        gate assignments that affect the next 100+ steps.
      - With MC returns, G_t captures the full discounted future from step t,
        so the VFA learns real value differences between states in ONE episode
        instead of needing T episodes of backward credit propagation.
    """

    def __init__(self, vfa: LinearVFA, extractor: PVFFeatureExtractor, gamma: float, alpha: float) -> None:
        self.vfa: LinearVFA = vfa
        self.extractor: PVFFeatureExtractor = extractor
        self.gamma: float = gamma
        self.alpha: float = alpha

    def learn_from_trajectory(self, trajectory: List[Tuple[Any, Any, float, Optional[Any]]]) -> None:
        """
        Update VFA using Monte Carlo returns for each visited state.

        Computes G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ... backward
        in one pass, then applies semi-gradient updates.
        Feature extraction is batched into one KNN call for the entire trajectory.
        """
        if not trajectory:
            return

        T = len(trajectory)

        # Compute actual discounted returns backward in one pass.
        # G[T-1] = r_{T-1}, G[t] = r_t + gamma * G[t+1]
        returns = np.empty(T, dtype=np.float64)
        G = 0.0
        for t in range(T - 1, -1, -1):
            _, _, reward, _ = trajectory[t]
            G = reward + self.gamma * G
            returns[t] = G

        # Single batch KNN call for all current states
        states = [s for s, _, _, _ in trajectory]
        phi_curr = self.extractor.extract_features_batch(states)

        # Semi-gradient MC update: theta += alpha * (G_t - V(s_t)) * phi(s_t)
        for i in range(T):
            self.vfa.update(phi_curr[i], returns[i], self.alpha)
