"""
Generate synthetic flight schedules based on templates.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json

from .flight import Flight


class ScheduleGenerator:
    """Generate synthetic flight schedules."""

    @staticmethod
    def generate(
        scenario_name: str,
        num_flights: int,
        generation_params: Dict,
        num_runways: int,
        aircraft_types: List[Any],  # List[AircraftTypeConfig]
        rng: np.random.Generator
    ) -> List[Flight]:
        """
        Generate a synthetic schedule with arrivals and departures.

        Args:
            scenario_name: Name of the scenario
            num_flights: Number of flights to generate
            generation_params: Parameters controlling generation
            num_runways: Number of available runways
            aircraft_types: List of aircraft type configurations
            rng: Random number generator

        Returns:
            List of generated Flight objects (arrivals and departures)
        """
        pattern = generation_params.get('arrival_pattern', 'uniform')
        time_window = generation_params['time_window']
        occupancy_time = generation_params.get('runway_occupancy_time', 2)

        # 1. Generate arrival times
        if pattern == 'uniform':
            times = ScheduleGenerator._generate_uniform(num_flights, time_window, rng)
        elif pattern == 'normal_peak':
            times = ScheduleGenerator._generate_normal_peak(num_flights, generation_params, rng)
        elif pattern == 'disrupted_wave':
            times = ScheduleGenerator._generate_disrupted_wave(num_flights, generation_params, num_runways, rng)
        else:
            raise ValueError(f"Unknown arrival pattern: {pattern}")

        # 2. Generate runway assignments and types
        runways = ScheduleGenerator._generate_runway_assignments(num_flights, num_runways, generation_params, rng)
        
        # Extract probabilities for type generation
        type_names = [ac.name for ac in aircraft_types]
        type_probs = [ac.probability for ac in aircraft_types]
        ac_type_names = rng.choice(type_names, size=num_flights, p=type_probs)

        # 3. Create initial arrival flights
        arrivals = []
        for i in range(num_flights):
            flight = Flight(
                flight_id=f"F{i+1:03d}",
                scheduled_time=int(times[i]),
                runway=int(runways[i]),
                aircraft_type=ac_type_names[i],
                direction="arrival"
            )
            arrivals.append(flight)

        # 4. Resolve arrival conflicts
        # Sort by time to process sequentially
        arrivals.sort(key=lambda f: f.scheduled_time)
        
        runway_free_time = {r: -1000 for r in range(num_runways)}
        
        for flight in arrivals:
            r = flight.runway
            # Ensure start time is after previous flight finished
            start_time = max(flight.scheduled_time, runway_free_time[r])
            flight.scheduled_time = int(start_time)
            runway_free_time[r] = start_time + occupancy_time

        # 5. Generate departures
        departures = []
        # Map type name to config for service time lookup
        type_map = {ac.name: ac for ac in aircraft_types}
        
        for flight in arrivals:
            ac_config = type_map[flight.aircraft_type]
            # Service time + taxi time (approximate)
            turnaround_time = ac_config.base_service_mean + 15 
            
            dep_time = flight.scheduled_time + turnaround_time
            
            # Assign departure runway (randomly for now)
            dep_runway = rng.integers(0, num_runways)
            
            dep_flight = Flight(
                flight_id=flight.flight_id,
                scheduled_time=int(dep_time),
                runway=dep_runway,
                aircraft_type=flight.aircraft_type,
                direction="departure"
            )
            departures.append(dep_flight)

        # 6. Resolve departure conflicts (respecting arrivals)
        departures.sort(key=lambda f: f.scheduled_time)
        
        # Track all occupied blocks: (start, end, runway)
        occupied_blocks = []
        for f in arrivals:
            occupied_blocks.append((f.scheduled_time, f.scheduled_time + occupancy_time, f.runway))
            
        # Sort blocks by start time
        occupied_blocks.sort()
        
        for flight in departures:
            r = flight.runway
            t = flight.scheduled_time
            duration = occupancy_time
            
            # Find a free slot
            while True:
                conflict = False
                for start, end, runway in occupied_blocks:
                    if runway != r:
                        continue
                    # Check overlap: not (end <= t or start >= t + duration)
                    if not (end <= t or start >= t + duration):
                        # Conflict! Move to end of this block
                        t = end
                        conflict = True
                        # Restart check from new time (optimization: continue from this block?)
                        # Restarting is safer to catch earlier blocks if we moved back? 
                        # No, we only move forward. But we might move into a block we already passed?
                        # Since blocks are sorted by start time, if we move t forward, we might overlap with a later block.
                        break
                
                if not conflict:
                    break
            
            flight.scheduled_time = int(t)
            # Add this departure to occupied blocks
            occupied_blocks.append((t, t + duration, r))
            # Re-sort blocks (could be optimized)
            occupied_blocks.sort()

        # Combine and return
        all_flights = arrivals + departures
        all_flights.sort(key=lambda f: f.scheduled_time)
        
        return all_flights

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