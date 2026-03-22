from typing import List, Tuple, Any, Optional

import numpy as np

from src.adp.features import PVFFeatureExtractor
from src.adp.value_function import LinearVFA


class TDLambdaLearner:
    """
    TD(lambda) learner with eligibility traces for episodic gate-assignment tasks.

    Generalises both TD(0) (lambda=0) and Monte Carlo (lambda=1):

      lambda=0   : one-step bootstrapping — low variance, high bias, slow
                   credit propagation (failed for us: ~6000 episodes to halve error)
      lambda=1   : full MC returns — unbiased but noisy over 720-step episodes;
                   SNR of the improvement signal was ~0.06 vs the noise floor
      lambda=0.9 : effective credit window ~30 steps (half-life = 6 steps with
                   gamma*lambda=0.891), much lower variance than MC while still
                   propagating value far enough to learn gate-quality differences

    Algorithm (backward view, accumulating traces):
      e ← 0
      for t = 0..T-1:
          e ← gamma * lambda * e + phi(s_t)        # accumulate trace
          delta ← r_t + gamma * V(s_{t+1}) - V(s_t)  # TD error
          theta ← theta + alpha * delta * e         # update with trace

    Feature extraction is batched into two KNN calls (current + next states)
    for the entire trajectory, then updates are applied sequentially with the
    live theta so each step's V(s') reflects the latest weights.
    """

    def __init__(
        self,
        vfa: LinearVFA,
        extractor: PVFFeatureExtractor,
        gamma: float,
        alpha: float,
        lambda_: float = 0.9,
    ) -> None:
        self.vfa = vfa
        self.extractor = extractor
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_ = lambda_

    def learn_from_trajectory(
        self, trajectory: List[Tuple[Any, Any, float, Optional[Any]]]
    ) -> None:
        """
        Update VFA using TD(lambda) with eligibility traces.

        Feature extraction is batched; updates are sequential so V(s') in the
        TD error always uses the current (updated) theta.
        """
        if not trajectory:
            return

        T = len(trajectory)

        # ── Batch feature extraction (two KNN calls total) ──────────────────
        states = [s for s, _, _, _ in trajectory]
        next_states_raw = [ns for _, _, _, ns in trajectory]

        phi_curr = self.extractor.extract_features_batch(states)

        terminal_mask = [ns is None for ns in next_states_raw]
        non_terminal_next = [ns for ns in next_states_raw if ns is not None]
        phi_next_nt = (
            self.extractor.extract_features_batch(non_terminal_next)
            if non_terminal_next
            else np.zeros((0, self.extractor.num_features), dtype=np.float64)
        )

        # ── Forward pass with eligibility traces ────────────────────────────
        decay = self.gamma * self.lambda_
        e = np.zeros(self.extractor.num_features, dtype=np.float64)
        nt_idx = 0

        for t in range(T):
            _, _, reward, _ = trajectory[t]

            # Accumulate eligibility trace
            e = decay * e + phi_curr[t]

            # TD error with current theta (live V values)
            v_curr = self.vfa.predict(phi_curr[t])
            if terminal_mask[t]:
                v_next = 0.0
            else:
                v_next = self.vfa.predict(phi_next_nt[nt_idx])
                nt_idx += 1

            delta = reward + self.gamma * v_next - v_curr
            self.vfa.theta += self.alpha * delta * e
