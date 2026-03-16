"""
MDP Environment Implementation.

This is the core simulation logic that advances the system state.
It handles state transitions, reward calculation, and stochastic dynamics.
Optimized for performance with O(1) lookups in the inner loop.
"""
from typing import Dict, List, Tuple, Any, Optional
from collections import deque, defaultdict
import numpy as np

from src.mdp.state import AirportState
from src.mdp.action import Action
from src.simulation.realization import ActiveFlight
from src.config.scenario import ScenarioConfig
from src.config.reward import RewardConfig


class AirportEnvironment:
    """
    Markov Decision Process environment for airport gate assignment.

    Attributes:
        scenario: The complete configuration for the scenario.
        active_flights: Mutable dictionary of flight instances (for delay tracking).
        
        _current_queue: Mutable deque for efficient queue management internally.
        _current_gates: Mutable numpy array for efficient gate timers.
        _base_service_times: Pre-computed map of aircraft type -> mean service time.
        _arrivals_map: Pre-computed map of time -> list of scheduled flights.
    """

    def __init__(self, scenario: ScenarioConfig):
        self.scenario = scenario
        self.active_flights: Dict[str, ActiveFlight] = {}
        
        # Internal mutable state (for efficiency)
        self._current_queue: deque[str] = deque()
        # This now represents the time until each gate is available. 0 means free.
        self._gate_available_time: np.ndarray = np.zeros(scenario.airport.num_gates, dtype=int)
        
        # Reward configuration shortcut
        self.reward_config: RewardConfig = scenario.rewards
        
        # Performance Optimization: Pre-compute base service times for O(1) lookup
        self._base_service_times: Dict[str, float] = {
            ac.name: ac.base_service_mean 
            for ac in scenario.aircraft_types
        }
        
        # Pre-process arrivals for O(1) access by time
        self._arrivals_map: Dict[int, List[ActiveFlight]] = defaultdict(list)
        self._init_arrivals_map()
        
        # Current timestep
        self.t: int = 0
        
        # Done flag
        self.done: bool = False

    def _init_arrivals_map(self):
        """Populate the arrivals map from the scenario schedule, applying stochastic noise."""
        flights = self.scenario.schedule.get_flights()
        
        # We need a random generator for sampling delays.
        # Using a new generator without a fixed seed means it will be stochastic on every run.
        rng = np.random.default_rng()
        
        for sched_flight in flights:
            if sched_flight.direction == "arrival":
                # Base scheduled time
                arrival_time = sched_flight.scheduled_time
                
                # Apply noise from the scenario configuration (arrival noise)
                if getattr(self.scenario, 'noise_models', None) and self.scenario.noise_models.arrival:
                    delay_float = self.scenario.noise_models.arrival.sample(rng=rng, size=1)[0]
                    delay_minutes = int(round(delay_float))
                elif getattr(self.scenario, 'noise_model', None):
                    # Fallback for old configs
                    delay_float = self.scenario.noise_model.sample(rng=rng, size=1)[0]
                    delay_minutes = int(round(delay_float))
                else:
                    delay_minutes = 0
                
                # Calculate actual arrival time and ensure it doesn't go below 0
                actual_arrival_time = max(0, arrival_time + delay_minutes)
                
                active_flight = ActiveFlight(
                    schedule=sched_flight,
                    actual_arrival_time=actual_arrival_time
                )
                
                # Map the flight using the actual (stochastic) arrival time
                self._arrivals_map[actual_arrival_time].append(active_flight)

    def reset(self) -> AirportState:
        """
        Reset the environment to the initial state (t=0).

        Returns:
            The initial AirportState.
        """
        self.t = 0
        self.done = False
        
        # Clear internal state
        self.active_flights.clear()
        self._current_queue.clear()
        self._gate_available_time.fill(0)
        
        # Process initial arrivals at t=0
        self._process_arrivals(0)
        
        # Return immutable snapshot
        return self._get_state_snapshot()

    def get_valid_actions(self, state: AirportState) -> List[Action]:
        """
        Returns a list of valid actions for the given state, enforcing hard constraints.
        An action is valid if the gate is compatible and available.
        """
        # If nothing is in the queue, the only action is to wait.
        if not state.runway_queue:
            return [Action(flight_id=None, gate_idx=-1)]

        flight_id = state.runway_queue[0]
        flight = self.active_flights[flight_id]
        
        valid_actions = []
        num_gates = self.scenario.airport.num_gates
        
        try:
            # We need the aircraft index to query compatibility efficiently
            ac_idx = self.scenario.compatibility.type_to_idx[flight.aircraft_type]
            
            for gate_idx in range(num_gates):
                # 1. HARD CONSTRAINT: Gate must be free at the current time.
                if state.gates[gate_idx] == 0:
                    # 2. COMPATIBILITY: Gate must be compatible with aircraft type
                    if self.scenario.compatibility.is_compatible_idx(ac_idx, gate_idx):
                        valid_actions.append(Action(flight_id=flight_id, gate_idx=gate_idx))
                        
        except KeyError:
            # If the aircraft type somehow isn't in the config, it can't dock anywhere.
            pass

        # If no gates are available for this flight, the only option is to hold.
        # This is represented by a NO_OP action. The simulation will advance,
        # and the flight will remain in the queue, incurring a penalty.
        if not valid_actions:
            return [Action(flight_id=None, gate_idx=-1)]
            
        return valid_actions

    def step(self, action: Action) -> Tuple[AirportState, float, bool, Dict[str, Any]]:
        """
        Execute one timestep in the environment.
        """
        if self.done:
            raise RuntimeError("Cannot step() on a finished environment. Call reset() first.")

        # --- 1. Apply Action ---
        assignment_made = False
        preference_score = 0.0

        if not action.is_noop:
            flight_id = action.flight_id
            gate_idx = action.gate_idx
            
            # VALIDATION: Ensure action is valid
            # 1. Flight is at the head of the queue
            # 2. Gate is actually available
            if self._current_queue and self._current_queue[0] == flight_id and self._gate_available_time[gate_idx] == 0:
                self._current_queue.popleft()
                
                flight = self.active_flights[flight_id]
                flight.gate_assignment = gate_idx
                flight.service_start_time = self.t
                
                taxi_time = self.scenario.airport.get_taxiing_time(flight.runway, gate_idx)
                base_service = self._base_service_times[flight.aircraft_type]
                
                # Sample stochastic service time
                rng = np.random.default_rng()
                if getattr(self.scenario, 'noise_models', None) and self.scenario.noise_models.service:
                    # Create a temporary noise model overriding the mean to be base_service
                    # We assume distribution is normal as per the new structure
                    service_std = self.scenario.noise_models.service.params.get('std', 0)
                    if service_std > 0:
                        sampled_service = rng.normal(loc=base_service, scale=service_std)
                    else:
                        sampled_service = base_service
                else:
                    sampled_service = base_service
                
                # Ensure it doesn't go below a minimum of 15 minutes
                actual_service = max(15, int(round(sampled_service)))
                
                total_service_time = actual_service + taxi_time
                
                # Set the gate's timer to the total service duration
                self._gate_available_time[gate_idx] = total_service_time
                
                assignment_made = True
                
                try:
                    ac_idx = self.scenario.compatibility.type_to_idx[flight.aircraft_type]
                    preference_score = self.scenario.compatibility.get_preference_idx(ac_idx, gate_idx)
                except KeyError:
                    preference_score = 0.0
        
        # If action is NO_OP (hold), no assignment is made, flight stays in queue.

        # --- 2. Reward Calculation ---
        queue_len = len(self._current_queue)
        reward = self.reward_config.compute_reward(
            queue_length=queue_len,
            assignment_made=assignment_made,
            preference_score=preference_score
        )

        # --- 3. Dynamics (Tick) ---
        self.t += 1
        self._gate_available_time = np.maximum(0, self._gate_available_time - 1)

        # --- 4. Stochastic Arrivals ---
        self._process_arrivals(self.t)

        # --- 5. Termination Condition ---
        if self.t >= self.scenario.time.horizon:
            self.done = True

        # --- Return Snapshot ---
        next_state = self._get_state_snapshot()
        info = {
            "assignment_made": assignment_made,
            "queue_length": len(self._current_queue),
            "occupied_gates": np.sum(self._gate_available_time > 0)
        }
        
        return next_state, reward, self.done, info

    def simulate_action(self, state: AirportState, action: Action) -> Tuple[AirportState, float, bool]:
        """
        Simulate the effect of an action on a given state without mutating the actual environment.
        Used strictly by the ADP agent for lookahead (evaluating V(s')).
        """
        # 1. Setup temporary variables for the hypothetical future
        next_t = state.t + 1
        is_done = next_t >= self.scenario.time.horizon

        # Copy the immutable tuples into mutable lists for the calculation
        next_gates = list(state.gates)
        next_queue = list(state.runway_queue)

        assignment_made = False
        preference_score = 0.0

        # 2. Apply action logic
        if not getattr(action, 'is_noop', True) and action.flight_id is not None:
            # If the action is valid, process the assignment
            if next_queue and next_queue[0] == action.flight_id:
                next_queue.pop(0)  # Remove the assigned flight from the queue

                flight = self.active_flights[action.flight_id]
                taxi_time = self.scenario.airport.get_taxiing_time(flight.runway, action.gate_idx)
                base_service = self._base_service_times[flight.aircraft_type]
                
                # Sample stochastic service time
                rng = np.random.default_rng()
                if getattr(self.scenario, 'noise_models', None) and self.scenario.noise_models.service:
                    service_std = self.scenario.noise_models.service.params.get('std', 0)
                    if service_std > 0:
                        sampled_service = rng.normal(loc=base_service, scale=service_std)
                    else:
                        sampled_service = base_service
                else:
                    sampled_service = base_service
                
                actual_service = max(15, int(round(sampled_service)))
                total_service = actual_service + taxi_time

                # Occupy the gate
                next_gates[action.gate_idx] = total_service
                assignment_made = True

                try:
                    ac_idx = self.scenario.compatibility.type_to_idx[flight.aircraft_type]
                    preference_score = self.scenario.compatibility.get_preference_idx(ac_idx, action.gate_idx)
                except KeyError:
                    preference_score = 0.0

        # 3. Calculate Expected Reward
        reward = self.reward_config.compute_reward(
            queue_length=len(next_queue),
            assignment_made=assignment_made,
            preference_score=preference_score
        )

        # 4. Apply Dynamics (Tick time forward)
        # Decrease all occupied gate timers by 1, floor at 0
        next_gates = [max(0, g - 1) for g in next_gates]

        # 5. Inject Expected Future Arrivals
        # The agent expects planes scheduled for the next minute to appear in the queue
        new_flights = self._arrivals_map.get(next_t, [])
        for f in new_flights:
            next_queue.append(f.flight_id)

        # 6. Package back into an immutable state
        next_state = AirportState(
            t=next_t,
            gates=tuple(next_gates),
            runway_queue=tuple(next_queue)
        )

        return next_state, reward, is_done

    def _process_arrivals(self, time: int):
        """Helper to inject new arrivals into the system."""
        new_flights = self._arrivals_map.get(time)
        if new_flights:
            for flight in new_flights:
                self.active_flights[flight.flight_id] = flight
                self._current_queue.append(flight.flight_id)

    def _get_state_snapshot(self) -> AirportState:
        """Create an immutable AirportState from current mutable internals."""
        return AirportState(
            t=self.t,
            gates=tuple(self._gate_available_time.tolist()),
            runway_queue=tuple(self._current_queue)
        )
