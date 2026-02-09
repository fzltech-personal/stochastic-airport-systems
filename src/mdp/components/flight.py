"""
Flight domain object.

Represents a single aircraft arrival in the airport system.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Flight:
    """
    Single flight in a schedule.

    Represents an aircraft arrival with scheduled and actual timing,
    physical constraints (runway, aircraft type), and metadata.

    This is a domain object used throughout the MDP simulation,
    not just a configuration data structure.
    """

    # Core attributes (always present)
    flight_id: str
    scheduled_time: int  # Minutes from start of day
    runway: int  # Runway index (0-indexed)
    aircraft_type: str  # Type name (e.g., "narrow-body")

    # Optional attributes (may be present in historical data)
    actual_time: Optional[int] = None
    airline: Optional[str] = None
    origin: Optional[str] = None
    registration: Optional[str] = None
    terminal: Optional[str] = None
    gate_assigned: Optional[str] = None  # Historical assignment (if known)
    priority: int = 1
    delay_reason: Optional[str] = None

    # Computed properties

    @property
    def actual_delay(self) -> Optional[int]:
        """
        Compute actual delay if actual_time is available.

        Returns:
            Delay in minutes (positive = late, negative = early),
            or None if actual_time not available
        """
        if self.actual_time is not None:
            return self.actual_time - self.scheduled_time
        return None

    @property
    def is_delayed(self) -> bool:
        """Check if flight is delayed (actual > scheduled)."""
        delay = self.actual_delay
        return delay is not None and delay > 0

    # Serialization

    def to_dict(self) -> dict:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with only non-None fields
        """
        return {
            k: v for k, v in self.__dict__.items()
            if v is not None
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Flight':
        """
        Create Flight from dictionary.

        Args:
            data: Dictionary with flight attributes

        Returns:
            Flight instance
        """
        return cls(**data)

    # Display

    def __str__(self) -> str:
        delay_str = ""
        if self.actual_delay is not None:
            delay_str = f", delay={self.actual_delay:+d}min"

        return (f"Flight({self.flight_id}, t={self.scheduled_time}, "
                f"runway={self.runway}, type={self.aircraft_type}{delay_str})")

    def __repr__(self) -> str:
        return self.__str__()
