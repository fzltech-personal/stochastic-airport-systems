"""
MDP Action Definition.

Defines the discrete action space for the airport gate assignment problem.
Includes helper methods for generating valid actions based on compatibility.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, TYPE_CHECKING
from functools import lru_cache

if TYPE_CHECKING:
    from src.mdp.state import AirportState
    from src.simulation.realization import ActiveFlight
    from src.config.compatibility import CompatibilityConfig


@dataclass(frozen=True)
class Action:
    """
    Represents a decision to assign a flight to a gate or do nothing.
    
    Attributes:
        flight_id: The unique ID of the flight being acted upon. 
                   None represents a NO_OP (do nothing).
        gate_idx: The index of the gate to assign. 
                  -1 represents no assignment (NO_OP).
    """
    flight_id: Optional[str]
    gate_idx: int

    @property
    def is_noop(self) -> bool:
        """Check if this is a 'do nothing' action."""
        return self.flight_id is None and self.gate_idx == -1

    def __str__(self) -> str:
        if self.is_noop:
            return "Action(NO_OP)"
        return f"Action(Assign {self.flight_id} -> Gate {self.gate_idx})"


# Define the canonical NO_OP instance
NO_OP = Action(flight_id=None, gate_idx=-1)


class ActionSpace:
    """
    Generator for valid actions in a given state.
    """

    @staticmethod
    def get_valid_actions(
        state: 'AirportState',
        active_flights: Dict[str, 'ActiveFlight'],
        compatibility: 'CompatibilityConfig'
    ) -> List[Action]:
        """
        Generate all valid actions for the current state.

        Logic:
        1. Always include NO_OP (wait).
        2. If queue is empty, return [NO_OP].
        3. If queue not empty, take the head of the queue (FIFO policy).
        4. Check all empty gates (where state.gates[i] == 0).
        5. Filter by compatibility using fast integer lookups.
        
        Args:
            state: Current immutable AirportState.
            active_flights: Dictionary mapping flight_ids to ActiveFlight objects.
                            Needed to look up aircraft type for compatibility.
            compatibility: The CompatibilityConfig instance.

        Returns:
            List of valid Action objects.
        """
        actions = [NO_OP]

        # If queue is empty, we can only wait
        if not state.runway_queue:
            return actions

        # Strict FIFO: Only consider the first aircraft in the queue
        flight_id = state.runway_queue[0]
        flight = active_flights[flight_id]
        
        # Get aircraft type index for fast lookup
        # This assumes compatibility config has been initialized correctly
        # and type_to_idx is populated or available
        try:
            ac_type_idx = compatibility.type_to_idx[flight.aircraft_type]
        except KeyError:
            # If type not found in config, no valid assignments possible
            # (Configuration error, but handle gracefully in simulation)
            return actions

        # Iterate over all gates
        # In highly optimized code, we might iterate only over free gates directly
        # but here we iterate over indices for clarity
        for gate_idx, remaining_time in enumerate(state.gates):
            # Constraint 1: Gate must be free
            if remaining_time == 0:
                # Constraint 2: Aircraft must be compatible
                if compatibility.is_compatible_idx(ac_type_idx, gate_idx):
                    actions.append(Action(flight_id=flight_id, gate_idx=gate_idx))

        return actions
