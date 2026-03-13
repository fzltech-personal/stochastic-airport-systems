"""
Flight domain objects.

Defines the static (scheduled) flight entity.
"""
from dataclasses import dataclass, fields
from typing import Optional, Dict


@dataclass(frozen=True)
class ScheduledFlight:
    """
    Immutable representation of a flight as it appears in the master schedule.
    
    This object serves as the static definition/contract for a flight operation.
    It does not contain mutable runtime state (like actual arrival times or 
    current delay), which are handled by the ActiveFlight wrapper in the simulation.
    """

    # Core attributes (always present)
    flight_id: str
    scheduled_time: int  # Minutes from start of day
    runway: int  # Runway index (0-indexed)
    aircraft_type: str  # Type name (e.g., "narrow-body")
    direction: str = "arrival"  # "arrival" or "departure"

    # Optional metadata
    airline: Optional[str] = None
    origin: Optional[str] = None
    registration: Optional[str] = None
    terminal: Optional[str] = None
    priority: int = 1
    linked_flight_id: Optional[str] = None

    # Serialization
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            k: v for k, v in self.__dict__.items()
            if v is not None
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ScheduledFlight':
        """Create ScheduledFlight from dictionary."""
        # Use introspection to determine the valid field names
        valid_fields = {f.name for f in fields(cls)}
        
        # Filter the incoming dictionary to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)

    def __str__(self) -> str:
        return (f"ScheduledFlight({self.flight_id}, {self.direction}, "
                f"t={self.scheduled_time}, runway={self.runway}, "
                f"type={self.aircraft_type})")

    def __repr__(self) -> str:
        return self.__str__()
