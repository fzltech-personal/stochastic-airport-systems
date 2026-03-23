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
        n_groups = len(self._gate_groups)
        group_sizes = [len(g) for g in self._gate_groups]
        print(f"  [Coarsener] Derived {n_groups} gate groups: {group_sizes}")

        # ── Derive queue bin boundaries from Q_max ────────────────────────────
        # Bin boundaries: [0, 1, 2, 3, Q_max//4, Q_max//2, Q_max]
        # Deduplicated and sorted so small Q_max values don't create duplicate bins.
        q_max: int = scenario.rewards.Q_max
        raw_bounds = [0, 1, 2, 3, q_max // 4, q_max // 2, q_max]
        self._boundaries: List[int] = sorted(set(raw_bounds))

    def coarsen(self, resource_state: Tuple) -> Tuple:
        """
        Map a raw resource_state to a compact coarsened tuple.

        Args:
            resource_state: (gates_tuple, queue_composition_tuple) from
                            AirportState.resource_state.

        Returns:
            A flat tuple of ints: free counts per gate group concatenated with
            binned queue counts per aircraft type.
        """
        gates_tuple, queue_composition = resource_state

        # Count free gates (remaining service time == 0) per group
        free_counts = tuple(
            sum(1 for i in group if gates_tuple[i] == 0)
            for group in self._gate_groups
        )

        # Map each aircraft-type count to a coarse bin index
        binned_queue = tuple(
            bisect.bisect_right(self._boundaries, n)
            for n in queue_composition
        )

        return free_counts + binned_queue
