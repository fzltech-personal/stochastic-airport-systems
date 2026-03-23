from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional

import numpy as np

from src.adp.features import PVFFeatureExtractor
from src.adp.value_function import LinearVFA


class BaseLearner(ABC):
    """
    Common interface for all weight-update methods.

    Subclasses only need to implement ``learn_from_trajectory``.
    The training loop calls this method once per episode and never
    needs to know which algorithm is running.
    """

    @abstractmethod
    def learn_from_trajectory(
        self, trajectory: List[Tuple[Any, Any, float, Optional[Any]]]
    ) -> None:
        """Update vfa.theta given one episode of experience."""


class TDLambdaLearner(BaseLearner):
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


class LSTDLearner(BaseLearner):
    """
    LSTD(lambda) learner: solves the Bellman fixed-point equation exactly.

    Semi-gradient TD chases a moving target with noisy steps. LSTD instead
    accumulates the sufficient statistics A and b across all episodes and
    solves the linear system theta = A^{-1} b after each episode.  For a
    linear VFA this is the exact fixed-point solution — no learning-rate
    tuning required, and convergence is typically 10-100x faster in episodes.

    Statistics never reset: each new episode refines the existing estimate,
    so the solve after episode N uses all N episodes of data.

    With lambda=0 (default):
        A  +=  phi(s_t) * [phi(s_t) - gamma * phi(s_{t+1})]^T
        b  +=  phi(s_t) * r_t

    With lambda > 0 (LSTD-lambda, eligibility traces):
        e   = gamma * lambda * e + phi(s_t)          (accumulate trace)
        A  +=  e * [phi(s_t) - gamma * phi(s_{t+1})]^T
        b  +=  e * r_t

    A small regularization term (reg * I) is added before solving to keep
    A invertible in the early episodes when it is rank-deficient.

    Args:
        vfa:       LinearVFA whose theta will be updated after each episode.
        extractor: PVFFeatureExtractor used to map states to feature vectors.
        gamma:     Discount factor.
        lambda_:   Eligibility-trace decay (0 = pure LSTD, 1 = MC-LSTD).
        reg:       Ridge regularization added to A before solving (default 1e-4).
    """

    def __init__(
        self,
        vfa: LinearVFA,
        extractor: PVFFeatureExtractor,
        gamma: float,
        lambda_: float = 0.0,
        reg: float = 1e-4,
    ) -> None:
        self.vfa = vfa
        self.extractor = extractor
        self.gamma = gamma
        self.lambda_ = lambda_
        self.reg = reg

        k = extractor.num_features
        self._A = np.zeros((k, k), dtype=np.float64)
        self._b = np.zeros(k, dtype=np.float64)

    def learn_from_trajectory(
        self, trajectory: List[Tuple[Any, Any, float, Optional[Any]]]
    ) -> None:
        """
        Accumulate LSTD statistics from one episode, then solve for theta.

        Feature extraction is batched (two KNN calls); the A/b accumulation
        and the final solve are both O(k^2) per step and O(k^3) per episode.
        For k <= 200 this is negligible compared to simulation time.
        """
        if not trajectory:
            return

        # ── Batch feature extraction ─────────────────────────────────────────
        states = [s for s, _, _, _ in trajectory]
        next_states_raw = [ns for _, _, _, ns in trajectory]

        phi_curr = self.extractor.extract_features_batch(states)

        terminal_mask = [ns is None for ns in next_states_raw]
        non_terminal_next = [ns for ns in next_states_raw if ns is not None]
        k = self.extractor.num_features
        phi_next_nt = (
            self.extractor.extract_features_batch(non_terminal_next)
            if non_terminal_next
            else np.zeros((0, k), dtype=np.float64)
        )

        # ── Accumulate A and b with eligibility traces ───────────────────────
        decay = self.gamma * self.lambda_
        e = np.zeros(k, dtype=np.float64)
        nt_idx = 0

        for t in range(len(trajectory)):
            _, _, reward, _ = trajectory[t]

            e = decay * e + phi_curr[t]

            if terminal_mask[t]:
                phi_diff = phi_curr[t]            # no next state: gamma*phi_next = 0
            else:
                phi_diff = phi_curr[t] - self.gamma * phi_next_nt[nt_idx]
                nt_idx += 1

            # Rank-1 updates: A += e * phi_diff^T,  b += e * r
            self._A += np.outer(e, phi_diff)
            self._b += e * reward

        # ── Solve (A + reg*I) theta = b ──────────────────────────────────────
        A_reg = self._A + self.reg * np.eye(k, dtype=np.float64)
        try:
            self.vfa.theta = np.linalg.solve(A_reg, self._b)
        except np.linalg.LinAlgError:
            # Fallback to least-squares if the system is still ill-conditioned
            self.vfa.theta, _, _, _ = np.linalg.lstsq(A_reg, self._b, rcond=None)
