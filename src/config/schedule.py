"""
Flight schedule configuration.

Handles loading and validation of flight schedules from various sources
(JSON, CSV, or generation parameters).
"""
from typing import Dict, List, Optional
from pathlib import Path
import json
import csv
from attrs import frozen, field, validators

from src.mdp.components.flight import ScheduledFlight  # Import from domain objects


def _validate_exclusive_schedule_source(instance, attribute, value):
    """Ensure only one specification method is used."""
    modes = [
        instance.flights is not None,
        instance.generation_params is not None,
        instance.schedule_file is not None
    ]
    if sum(modes) > 1:
        raise ValueError(
            "Cannot specify multiple schedule sources. "
            "Use either 'flights', 'generation_params', or 'schedule_file'"
        )
    if sum(modes) == 0:
        raise ValueError(
            "Must specify one of: 'flights', 'generation_params', or 'schedule_file'"
        )


def _validate_flight_count(instance, attribute, value):
    """Ensure flight list matches num_flights if explicit."""
    if instance.flights is not None and len(instance.flights) != instance.num_flights:
        raise ValueError(
            f"Flight list length {len(instance.flights)} != num_flights {instance.num_flights}"
        )


@frozen
class ScheduleConfig:
    """
    Flight schedule configuration.

    Supports three modes:
    1. Explicit flight list (loaded from JSON/CSV)
    2. Synthetic generation (parameters only)
    3. File reference (loads from JSON/CSV)
    """

    scenario_name: str = field(validator=validators.min_len(1))
    num_flights: int = field(validator=validators.gt(1))

    # Mode 1: Explicit flights (from file or inline)
    flights: Optional[List[ScheduledFlight]] = field(default=None)

    # Mode 2: Generation parameters (synthetic)
    generation_params: Optional[Dict] = field(default=None)

    # Mode 3: File reference (loads from JSON/CSV)
    schedule_file: Optional[Path] = field(default=None)

    # Metadata
    metadata: Optional[Dict] = field(default=None)

    # Validators applied to the instance (not individual fields)
    @scenario_name.validator
    def _validate_sources(self, attribute, value):
        """Validate that exactly one source is specified."""
        _validate_exclusive_schedule_source(self, attribute, value)

    @num_flights.validator
    def _validate_count(self, attribute, value):
        """Validate flight count matches if explicit."""
        _validate_flight_count(self, attribute, value)

    def get_flights(self) -> List[ScheduledFlight]:
        """
        Get the flight list, loading from file if necessary.

        Returns:
            List of ScheduledFlight objects

        Raises:
            ValueError: If using generation_params (needs ScheduleGenerator)
            FileNotFoundError: If schedule file doesn't exist
        """
        if self.flights is not None:
            return self.flights

        if self.schedule_file is not None:
            return self._load_from_file(self.schedule_file)

        # Generation params case - needs to be handled by schedule generator
        raise ValueError(
            "Cannot get flights from generation_params. "
            "Use ScheduleGenerator.generate() first."
        )

    def _load_from_file(self, filepath: Path) -> List[ScheduledFlight]:
        """Load flights from JSON or CSV file."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Schedule file not found: {filepath}")

        if filepath.suffix == '.json':
            return self._load_from_json(filepath)
        elif filepath.suffix == '.csv':
            return self._load_from_csv(filepath)
        else:
            raise ValueError(f"Unsupported schedule file format: {filepath.suffix}")

    def _load_from_json(self, filepath: Path) -> List[ScheduledFlight]:
        """Load flights from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        flights = []
        for flight_data in data['flights']:
            flights.append(ScheduledFlight.from_dict(flight_data))

        return flights

    def _load_from_csv(self, filepath: Path) -> List[ScheduledFlight]:
        """Load flights from CSV file."""
        flights = []

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                flight_data = {
                    'flight_id': row['flight_id'],
                    'scheduled_time': int(row['scheduled_time']),
                    'runway': int(row['runway']),
                    'aircraft_type': row['aircraft_type'],
                    'priority': int(row.get('priority', 1)),
                }

                # Add optional fields if present and not empty
                optional_fields = [
                    'airline', 'origin', 'registration',
                    'terminal', 'linked_flight_id'
                ]
                for field_name in optional_fields:
                    if field_name in row and row[field_name]:
                        flight_data[field_name] = row[field_name]

                flights.append(ScheduledFlight.from_dict(flight_data))

        return flights

    def __str__(self) -> str:
        if self.schedule_file:
            return (f"ScheduleConfig({self.scenario_name}, "
                   f"{self.num_flights} flights from {self.schedule_file.name})")
        elif self.flights:
            return (f"ScheduleConfig({self.scenario_name}, "
                   f"{self.num_flights} explicit flights)")
        else:
            return (f"ScheduleConfig({self.scenario_name}, "
                   f"{self.num_flights} flights, generated)")