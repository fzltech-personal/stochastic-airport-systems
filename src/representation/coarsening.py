"""
State coarsening for PVF graph construction.

Maps raw resource_states (exact service times per gate) to compact coarsened
tuples so that the state-transition graph has genuine node revisitation and
meaningful eigenvectors.

The coarsening is scenario-agnostic: gate groups and queue bins are derived
automatically from the compatibility matrix and Q_max — no hardcoded indices.
"""
import bisect
from typing import List, Tuple

import numpy as np


class CoarsenedStateBuilder:
    """
    Derives a compact state representation from scenario config.

    Two raw resource_states that look different due to stochastic service
    times (e.g., gate 3 has 7 remaining steps vs 8 remaining steps) map to
    the same coarsened tuple if they have the same number of *free* gates in
    each functional group and the same binned queue composition.

    Gate groups are derived automatically: two gates belong to the same group
    if and only if their column in the compatibility matrix is identical —
    i.e. they serve exactly the same set of aircraft types.

    Queue counts per aircraft type are mapped to coarse bins derived from Q_max.
    """

    def __init__(self, scenario) -> None:
        """
        Args:
            scenario: ScenarioConfig with .compatibility and .rewards attributes.
        """
        comp_matrix: np.ndarray = scenario.compatibility.compatibility_matrix
        # shape: (num_active_types, num_gates)
        num_gates: int = comp_matrix.shape[1]

        # ── Derive gate groups from compatibility columns ─────────────────────
        # Two gates are in the same group iff their compatibility column vectors
        # are identical (they serve precisely the same aircraft types).
        groups: dict = {}
        for gate_idx in range(num_gates):
            col = tuple(comp_matrix[:, gate_idx].tolist())
            if col not in groups:
                groups[col] = []
            groups[col].append(gate_idx)

        self._gate_groups: List[List[int]] = list(groups.values())
        self._group_sizes: List[int] = [len(g) for g in self._gate_groups]
        n_groups = len(self._gate_groups)

        print(f"  [Coarsener] Derived {n_groups} gate groups (free-count bins at 0%, 25%, 75%, 100%):")
        for g_idx, (group, size) in enumerate(zip(self._gate_groups, self._group_sizes)):
            b1 = max(1, int(0.25 * size))         # bin 1 upper bound (few free)
            b2_lo = b1 + 1                         # bin 2 lower bound
            b2_hi = int(0.75 * size)               # bin 2 upper bound
            b3_lo = b2_hi + 1                      # bin 3 lower bound
            b3_hi = size - 1                       # bin 3 upper bound
            print(
                f"    Group {g_idx} ({size} gates): "
                f"bins at free=[0 | 1-{b1} | {b2_lo}-{b2_hi} | {b3_lo}-{b3_hi} | {size}]"
            )

        # ── Derive queue bin boundaries from Q_max ────────────────────────────
        # Bin boundaries: [0, 1, 2, 3, Q_max//4, Q_max//2, Q_max]
        # Deduplicated and sorted so small Q_max values don't create duplicate bins.
        q_max: int = scenario.rewards.Q_max
        raw_bounds = [0, 1, 2, 3, q_max // 4, q_max // 2, q_max]
        self._boundaries: List[int] = sorted(set(raw_bounds))

    def _bin_free_count(self, free: int, group_size: int) -> int:
        """
        Map an exact free-gate count to one of 5 capacity-pressure bins.

        Bin 0 — none free  (fully occupied)
        Bin 1 — few free   (0% < fraction <= 25%)
        Bin 2 — some free  (25% < fraction <= 75%)
        Bin 3 — mostly free (75% < fraction < 100%)
        Bin 4 — all free   (completely empty)
        """
        if free == 0:
            return 0
        if free == group_size:
            return 4
        frac = free / group_size
        if frac <= 0.25:
            return 1
        if frac <= 0.75:
            return 2
        return 3

    def coarsen(self, resource_state: Tuple) -> Tuple:
        """
        Map a raw resource_state to a compact coarsened tuple.

        Args:
            resource_state: (gates_tuple, queue_composition_tuple) from
                            AirportState.resource_state.

        Returns:
            A flat tuple of ints: binned free-gate counts per gate group
            (values in {0,1,2,3,4}) concatenated with binned queue counts
            per aircraft type.
        """
        gates_tuple, queue_composition = resource_state

        # Bin the free-gate count per group into 5 capacity-pressure levels
        free_counts = tuple(
            self._bin_free_count(
                sum(1 for i in group if gates_tuple[i] == 0),
                self._group_sizes[g_idx]
            )
            for g_idx, group in enumerate(self._gate_groups)
        )

        # Map each aircraft-type count to a coarse bin index
        binned_queue = tuple(
            bisect.bisect_right(self._boundaries, n)
            for n in queue_composition
        )

        return free_counts + binned_queue
