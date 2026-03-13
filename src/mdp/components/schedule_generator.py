"""
Generate synthetic flight schedules based on templates.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json

from .flight import ScheduledFlight


class ScheduleGenerator:
    """Generate synthetic flight schedules."""

    @staticmethod
    def generate(
            scenario_name: str,
            num_flights: int,
            generation_params: Dict[str, Any],
            num_runways: int,
            aircraft_types: List['AircraftTypeConfig'],
            rng: np.random.Generator
    ) -> List[ScheduledFlight]:

        flights = []

        # Unpack parameters
        time_window = generation_params.get('time_window', [360, 600])
        peak_time = generation_params.get('peak_time', 480)
        peak_std = generation_params.get('peak_std', 60)

        # Extract probabilities for the fleet mix
        types = [at.name for at in aircraft_types]
        probs = [at.probability for at in aircraft_types]
        probs = np.array(probs) / np.sum(probs)

        flight_id_counter = 1

        # num_flights is now actual_num_flights (sampled from Poisson in main.py)
        for _ in range(num_flights):
            # 1. Sample Aircraft Type
            chosen_type_name = rng.choice(types, p=probs)
            ac_config = next(at for at in aircraft_types if at.name == chosen_type_name)

            # 2. Sample Arrival Time
            arrival_time_float = rng.normal(loc=peak_time, scale=peak_std)
            arrival_time_float = np.clip(arrival_time_float, time_window[0], time_window[1])
            arrival_time = int(round(arrival_time_float))

            # 3. Sample Turnaround (Service) Time based on aircraft type
            service_time_float = rng.normal(
                loc=ac_config.base_service_mean,
                scale=ac_config.base_service_std
            )
            service_time = int(max(20, round(service_time_float)))

            # 4. Calculate Departure Time
            departure_time = arrival_time + service_time

            # 5. Create Identifiers
            tail_number = f"PH-{flight_id_counter:03d}"  # Dutch tail numbers!
            arr_id = f"A{flight_id_counter:04d}"
            dep_id = f"D{flight_id_counter:04d}"

            # 6. Assign random runways (can be overwritten by MDP later)
            arr_runway = int(rng.integers(0, num_runways))
            dep_runway = int(rng.integers(0, num_runways))

            # 7. Create the paired Flight objects
            arr_flight = ScheduledFlight(
                flight_id=arr_id,
                direction="arrival",
                scheduled_time=arrival_time,
                runway=arr_runway,
                aircraft_type=chosen_type_name,
                registration=tail_number,
                linked_flight_id=dep_id
            )

            dep_flight = ScheduledFlight(
                flight_id=dep_id,
                direction="departure",
                scheduled_time=departure_time,
                runway=dep_runway,
                aircraft_type=chosen_type_name,
                registration=tail_number,
                linked_flight_id=arr_id
            )

            flights.extend([arr_flight, dep_flight])
            flight_id_counter += 1

        # Sort all movements chronologically
        flights.sort(key=lambda x: x.scheduled_time)

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
        flights: List[ScheduledFlight],
        filepath: Path,
        metadata: Optional[Dict] = None
    ):
        """
        Save generated schedule to JSON file.

        Args:
            flights: List of ScheduledFlight objects
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