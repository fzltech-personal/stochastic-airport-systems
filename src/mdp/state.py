"""
MDP State Definition.

Defines the immutable state of the airport system at a specific timestep.
Optimized for hashing and equality checks to support graph construction and caching.
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AirportState:
    """
    Immutable snapshot of the airport system state.

    Attributes:
        t: Current simulation timestep.
        gates: Tuple of integers representing remaining service time for each gate.
               0 = Free, >0 = Occupied.
               Must be a tuple (not list/array) to ensure hashability.
        runway_queue: Tuple of flight IDs representing aircraft waiting for assignment.
                      Ordered by arrival time.
    """
    t: int
    gates: Tuple[int, ...]
    runway_queue: Tuple[str, ...]

    @property
    def resource_state(self) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
        """
        Get the time-independent resource configuration.
        
        Useful for detecting recurrent states in the graph (e.g., cycles)
        independent of absolute clock time.
        
        Returns:
            A tuple of (gates, runway_queue).
        """
        return self.gates, self.runway_queue

    @property
    def num_waiting(self) -> int:
        """Count of aircraft currently in the queue."""
        return len(self.runway_queue)

    @property
    def num_occupied_gates(self) -> int:
        """Count of gates currently servicing aircraft."""
        return sum(1 for g in self.gates if g > 0)

    def __str__(self) -> str:
        # condense gate vector for display
        active_gates = [i for i, time in enumerate(self.gates) if time > 0]
        return (f"State(t={self.t}, "
                f"queue_len={len(self.runway_queue)}, "
                f"occupied={active_gates})")
