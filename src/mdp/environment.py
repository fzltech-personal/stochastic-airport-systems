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
        self._current_gates: np.ndarray = np.zeros(scenario.num_gates, dtype=int)
        
        # Reward configuration shortcut
        self.reward_config: RewardConfig = scenario.rewards
        
        # Performance Optimization: Pre-compute base service times for O(1) lookup
        self._base_service_times: Dict[str, float] = {
            ac.name: ac.base_service_mean 
            for ac in scenario.aircraft_types
        }
        
        # Pre-process arrivals for O(1) access by time
        # In a full simulation, this might be dynamic/stochastic, 
        # but for the core MDP logic, we often work with a realization.
        self._arrivals_map: Dict[int, List[ActiveFlight]] = defaultdict(list)
        self._init_arrivals_map()
        
        # Current timestep
        self.t: int = 0
        
        # Done flag
        self.done: bool = False

    def _init_arrivals_map(self):
        """Populate the arrivals map from the scenario schedule."""
        flights = self.scenario.schedule.get_flights()
        for sched_flight in flights:
            if sched_flight.direction == "arrival":
                # Create realization (ActiveFlight)
                # Apply simple noise model if needed, or stick to schedule for baseline
                # For now, deterministic arrival at scheduled time
                arrival_time = sched_flight.scheduled_time
                
                active_flight = ActiveFlight(
                    schedule=sched_flight,
                    actual_arrival_time=arrival_time
                )
                self._arrivals_map[arrival_time].append(active_flight)

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
        self._current_gates.fill(0)
        
        # Process initial arrivals at t=0
        self._process_arrivals(0)
        
        # Return immutable snapshot
        return self._get_state_snapshot()

    def step(self, action: Action) -> Tuple[AirportState, float, bool, Dict[str, Any]]:
        """
        Execute one timestep in the environment.
        
        Sequence:
        1. Apply Action (Assign gate)
        2. Calculate Reward (Based on queue state *before* new arrivals)
        3. Dynamics (Tick: t+1, decrement gates)
        4. Stochastic Arrivals (Add new flights to queue)

        Args:
            action: The action chosen by the agent.

        Returns:
            next_state: The new immutable state.
            reward: The scalar reward for this transition.
            done: Whether the episode has ended.
            info: Diagnostic information.
        """
        if self.done:
            raise RuntimeError("Cannot step() on a finished environment. Call reset() first.")

        # --- 1. Apply Action ---
        assignment_made = False
        preference_score = 0.0

        if not action.is_noop:
            flight_id = action.flight_id
            gate_idx = action.gate_idx
            
            # Validation: ensure action matches head of queue
            if self._current_queue and self._current_queue[0] == flight_id:
                # Remove from queue (FIFO assumption: head of queue)
                self._current_queue.popleft()
                
                # Update ActiveFlight status
                flight = self.active_flights[flight_id]
                flight.gate_assignment = gate_idx
                flight.service_start_time = self.t
                
                # Determine service time
                # Taxiing depends on runway (from flight) and gate (from action)
                taxi_time = self.scenario.airport.get_taxiing_time(flight.runway, gate_idx)
                
                # Base service time (O(1) lookup)
                base_service = self._base_service_times[flight.aircraft_type]
                
                total_service_time = int(base_service + taxi_time)
                self._current_gates[gate_idx] = total_service_time
                
                assignment_made = True
                
                # Calculate preference score using fast lookup
                try:
                    ac_idx = self.scenario.compatibility.type_to_idx[flight.aircraft_type]
                    preference_score = self.scenario.compatibility.get_preference_idx(ac_idx, gate_idx)
                except KeyError:
                    preference_score = 0.0

        # --- 2. Reward Calculation ---
        # Calculate penalty based on the queue *after* assignment but *before* time step
        queue_len = len(self._current_queue)
        reward = self.reward_config.compute_reward(
            queue_length=queue_len,
            assignment_made=assignment_made,
            preference_score=preference_score
        )

        # --- 3. Dynamics (Tick) ---
        # Advance time
        self.t += 1
        
        # Decrement gate timers
        # Use numpy maximum to clamp at 0
        self._current_gates = np.maximum(0, self._current_gates - 1)

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
            "occupied_gates": np.sum(self._current_gates > 0)
        }
        
        return next_state, reward, self.done, info

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
            gates=tuple(self._current_gates.tolist()),
            runway_queue=tuple(self._current_queue)
        )
