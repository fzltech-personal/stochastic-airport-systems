"""
Generate synthetic flight schedules based on templates.
"""
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import json

from .flight import Flight  # Import from same package


class ScheduleGenerator:
    """Generate synthetic flight schedules."""

    @staticmethod
    def generate(
        scenario_name: str,
        num_flights: int,
        generation_params: Dict,
        num_runways: int,
        aircraft_type_probabilities: Dict[str, float],
        rng: np.random.Generator
    ) -> List[Flight]:
        """
        Generate a synthetic schedule.

        Args:
            scenario_name: Name of the scenario
            num_flights: Number of flights to generate
            generation_params: Parameters controlling generation
            num_runways: Number of available runways
            aircraft_type_probabilities: Dict mapping type name to probability
            rng: Random number generator

        Returns:
            List of generated Flight objects
        """
        pattern = generation_params.get('arrival_pattern', 'uniform')
        time_window = generation_params['time_window']

        # Generate arrival times based on pattern
        if pattern == 'uniform':
            times = ScheduleGenerator._generate_uniform(
                num_flights, time_window, rng
            )
        elif pattern == 'normal_peak':
            times = ScheduleGenerator._generate_normal_peak(
                num_flights, generation_params, rng
            )
        elif pattern == 'disrupted_wave':
            times = ScheduleGenerator._generate_disrupted_wave(
                num_flights, generation_params, num_runways, rng
            )
        else:
            raise ValueError(f"Unknown arrival pattern: {pattern}")

        # Sort times
        times = np.sort(times)

        # Generate runway assignments
        runways = ScheduleGenerator._generate_runway_assignments(
            num_flights, num_runways, generation_params, rng
        )

        # Generate aircraft types
        type_names = list(aircraft_type_probabilities.keys())
        type_probs = list(aircraft_type_probabilities.values())
        aircraft_types = rng.choice(type_names, size=num_flights, p=type_probs)

        # Create Flight objects
        flights = []
        for i in range(num_flights):
            flight = Flight(
                flight_id=f"F{i+1:03d}",
                scheduled_time=int(times[i]),
                runway=int(runways[i]),
                aircraft_type=aircraft_types[i],
                priority=1
            )
            flights.append(flight)

        return flights

    @staticmethod
    def _generate_uniform(
        num_flights: int,
        time_window: List[int],
        rng: np.random.Generator
    ) -> np.ndarray:
        """Generate uniformly distributed arrival times."""
        return rng.uniform(time_window[0], time_window[1], size=num_flights)

    @staticmethod
    def _generate_normal_peak(
        num_flights: int,
        params: Dict,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Generate arrivals with normal distribution around peak time."""
        peak_time = params['peak_time']
        peak_std = params['peak_std']
        time_window = params['time_window']

        times = rng.normal(peak_time, peak_std, size=num_flights)

        # Clip to time window
        times = np.clip(times, time_window[0], time_window[1])

        return times

    @staticmethod
    def _generate_disrupted_wave(
        num_flights: int,
        params: Dict,
        num_runways: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Generate arrivals with disruption event causing bunching."""
        time_window = params['time_window']
        disruption = params.get('disruption_event', {})

        # Start with uniform base
        times = rng.uniform(time_window[0], time_window[1], size=num_flights)

        if disruption:
            # Identify flights affected by disruption
            disruption_start = disruption['start_time']
            disruption_end = disruption_start + disruption['duration']

            # Flights during disruption get delayed and bunched
            affected = (times >= disruption_start) & (times < disruption_end)
            delay = rng.exponential(scale=15, size=affected.sum())
            times[affected] += delay

        return times

    @staticmethod
    def _generate_runway_assignments(
        num_flights: int,
        num_runways: int,
        params: Dict,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Generate runway assignments."""
        assignment_mode = params.get('runway_assignment', 'uniform')

        if assignment_mode == 'uniform':
            return rng.integers(0, num_runways, size=num_flights)

        elif assignment_mode == 'alternating':
            return np.arange(num_flights) % num_runways

        elif assignment_mode == 'weighted':
            probs = params.get('runway_probabilities', None)
            if probs is None:
                probs = [1.0 / num_runways] * num_runways
            return rng.choice(num_runways, size=num_flights, p=probs)

        else:
            raise ValueError(f"Unknown runway assignment mode: {assignment_mode}")

    @staticmethod
    def save_schedule(
        flights: List[Flight],
        filepath: Path,
        metadata: Optional[Dict] = None
    ):
        """
        Save generated schedule to JSON file.

        Args:
            flights: List of Flight objects
            filepath: Output file path
            metadata: Optional metadata to include
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'metadata': metadata or {},
            'flights': [f.to_dict() for f in flights]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)