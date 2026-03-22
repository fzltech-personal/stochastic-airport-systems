from typing import List, Tuple, Any, Optional

import numpy as np

from src.adp.features import PVFFeatureExtractor
from src.adp.value_function import LinearVFA

class TD0Learner:
    """
    TD(0) Learning algorithm orchestrator for an episode trajectory.
    """

    def __init__(self, vfa: LinearVFA, extractor: PVFFeatureExtractor, gamma: float, alpha: float) -> None:
        self.vfa: LinearVFA = vfa
        self.extractor: PVFFeatureExtractor = extractor
        self.gamma: float = gamma
        self.alpha: float = alpha

    def learn_from_trajectory(self, trajectory: List[Tuple[Any, Any, float, Optional[Any]]]) -> None:
        """
        Perform TD(0) updates for an entire episode trajectory.

        Feature extraction is batched into two KNN calls (one for all current
        states, one for all non-terminal next states) instead of 2×T individual
        calls, which is the dominant runtime cost.
        """
        if not trajectory:
            return

        # Collect states for batch extraction
        states = [s for s, _, _, _ in trajectory]
        next_states_raw = [ns for _, _, _, ns in trajectory]

        terminal_mask = [ns is None for ns in next_states_raw]
        non_terminal_next = [ns for ns in next_states_raw if ns is not None]

        # Two batch KNN calls instead of up to 2×T individual calls
        phi_curr = self.extractor.extract_features_batch(states)
        phi_next_nt = (
            self.extractor.extract_features_batch(non_terminal_next)
            if non_terminal_next
            else np.zeros((0, self.extractor.num_features), dtype=np.float64)
        )

        # TD updates using pre-extracted features
        nt_idx = 0
        for i, (_, _, reward, _) in enumerate(trajectory):
            if terminal_mask[i]:
                target = reward
            else:
                target = reward + self.gamma * self.vfa.predict(phi_next_nt[nt_idx])
                nt_idx += 1
            self.vfa.update(phi_curr[i], target, self.alpha)
