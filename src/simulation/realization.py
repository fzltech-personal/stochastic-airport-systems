"""
Simulation state for aircraft movements.

This module differentiates between the static plan (ScheduledFlight)
and the dynamic realization of that plan in the simulation (ActiveFlight).
"""
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional

from src.mdp.components.flight import ScheduledFlight


class FlightStatus(Enum):
    """Lifecycle states of a flight in the simulation."""
    AIRBORNE = auto()   # Not yet arrived (or taken off)
    QUEUED = auto()     # Waiting for a gate
    SERVICING = auto()  # Currently at a gate
    COMPLETED = auto()  # Finished and removed from system


@dataclass
class ActiveFlight:
    """
    Mutable runtime representation of a flight.
    
    Wraps a ScheduledFlight (immutable plan) and tracks its progress
    through the simulation, including stochastic delays and resource assignments.
    """
    
    schedule: ScheduledFlight
    
    # Stochastic Realizations
    actual_arrival_time: int  # The time it actually enters the system (t_sched + noise)
    
    # Runtime State (Mutable)
    gate_assignment: Optional[int] = None  # Gate index, if assigned
    status: FlightStatus = FlightStatus.AIRBORNE
    
    # Service Tracking
    service_start_time: Optional[int] = None
    remaining_service_time: int = 0
    
    @property
    def flight_id(self) -> str:
        """Proxy to scheduled flight ID."""
        return self.schedule.flight_id
        
    @property
    def aircraft_type(self) -> str:
        """Proxy to scheduled aircraft type."""
        return self.schedule.aircraft_type
        
    @property
    def runway(self) -> int:
        """Proxy to scheduled runway."""
        return self.schedule.runway

    @property
    def total_delay(self) -> int:
        """
        Calculate total delay relative to the schedule.
        
        If the flight has started service, delay is (service_start - scheduled_time).
        If not yet serviced, it is undefined (or could be projected), 
        but for MDP state purposes we often care about 'current waiting time'.
        
        Returns:
             The realized delay in minutes if service has started.
             Otherwise, returns the delay relative to the actual arrival time so far.
        """
        # If service has started, the delay is fixed
        if self.service_start_time is not None:
             return self.service_start_time - self.schedule.scheduled_time
             
        # If not yet serviced, the delay is at least (actual_arrival - scheduled)
        # This is a lower bound.
        initial_delay = self.actual_arrival_time - self.schedule.scheduled_time
        return max(0, initial_delay)

    def __str__(self) -> str:
        gate_str = f", gate={self.gate_assignment}" if self.gate_assignment is not None else ""
        return (f"ActiveFlight({self.flight_id}, {self.status.name}, "
                f"t_act={self.actual_arrival_time}{gate_str})")
