# Roadmap

The end goal is a **practical gate assignment system** that an airport operations team can use in real time: import the schedule for the coming hours, provide the current airport state and any known delays, and receive an optimised gate assignment plan. The system is backed by a trained ADP policy and exposed through a REST API, with an optional frontend for interactive use.

The architecture is built in layers with strict separation of concerns:

1. **MDP Layer** — pure mathematical rules of the airport (State, Transition, Reward)
2. **Simulation Layer** — stochastic episode execution and trajectory generation
3. **Representation Layer** — offline spectral graph construction and PVF computation
4. **ADP Layer** — value function learning and policy inference
5. **Ingestion Layer** — bridging real operational data into the MDP representation
6. **Inference Layer** — serving the trained policy against real input
7. **API Layer** — REST interface for external systems and the frontend
8. **Frontend** — interactive visualisation and manual override

---

## Phase 1–5: Core Research System ✅

The foundational MDP simulation, spectral representation learning, and TD training pipeline are implemented. Key components:

- `AirportState` / `Action` / `AirportEnvironment` with stochastic arrivals and service times
- `StateGraph` + `PVFCreator` for offline basis function computation
- `LinearVFA` + `TD0Learner` + `ADPPolicy` with one-step Bellman lookahead
- MLOps pipeline scripts (`01` through `05`) and `run_pipeline.py` orchestrator
- Scenario configs for Schiphol with realistic taxi matrices, fleet mixes, and compatibility constraints

---

## Phase 6: Real Schedule Ingestion

**Goal:** Accept a real flight schedule and current airport state as input, rather than relying on synthetic generation.

### 6.1 Schedule Parser

Build an ingestion layer that reads external schedule formats and produces a list of `ScheduledFlight` objects.

- Define a canonical internal schedule schema (JSON/CSV)
- Implement parser for at least one real format (IATA SSIM or a simple CSV export from an AODB)
- Map real ICAO aircraft type codes (e.g. `B738`, `A320`, `A388`) to internal categories (`narrow-body`, `wide-body`, `super-heavy`)
- Handle missing or unknown aircraft types with a configurable fallback category
- Validate schedule consistency (no duplicate flight IDs, times within simulation horizon)

### 6.2 Current Airport State Import

Allow the system to start from a non-empty airport state rather than always from `t=0`.

- Define an `AirportStateSnapshot` schema: for each gate, provide current aircraft type and estimated remaining service time (or free)
- Build a loader that hydrates `_gate_available_time` from this snapshot
- Validate that the snapshot is consistent with the airport topology config

### 6.3 Delay Integration

Incorporate known delays at ingestion time so the policy plans against actual expected arrival times.

- Accept a delay manifest: `{flight_id: delay_minutes}` alongside the schedule
- Apply delays on top of scheduled arrival times before populating `_arrivals_map`
- Allow partial delay information (only some flights have known delays; others use noise model defaults)

---

## Phase 7: Online Inference Engine

**Goal:** Run the trained ADP policy against real input and emit a concrete gate assignment plan.

### 7.1 Inference Mode

Separate training behaviour from inference behaviour in the policy.

- Add an `inference_mode` flag to `ADPPolicy` that sets `epsilon=0` and disables any weight updates
- Load pre-trained `theta` weights and PVF basis functions from disk at startup
- Expose a `plan(current_state, schedule_window)` method that returns a list of `(flight_id, gate_idx)` assignments

### 7.2 Rolling Horizon Planning

Real operations use a rolling window, not a fixed horizon.

- Define a `planning_horizon` (e.g. 120 minutes ahead) that limits how far into the schedule is considered
- Re-run planning on a configurable interval (e.g. every 5 minutes, or on each new arrival)
- Support incremental re-planning: flights already assigned are locked; only unassigned flights are re-optimised

### 7.3 Assignment Output Format

Define what the system returns.

- Return assignments as a structured list: `[{flight_id, gate, estimated_start, estimated_end, confidence}]`
- Flag assignments where the policy had to fall back to KNN (unseen state) so operators know which decisions are less certain
- Include a short reason for each assignment (gate preference score, no compatible alternatives, etc.)

---

## Phase 8: Robustness & Production Hardening

**Goal:** Make the system reliable when real data is messy.

### 8.1 Graceful Constraint Handling

- If no compatible gate exists for a flight (all occupied or incompatible), emit a `HOLD` assignment with estimated wait time rather than crashing
- If an aircraft type is unknown, log a warning and assign using the nearest compatible category
- Cap queue overflow rather than letting the reward diverge silently

### 8.2 Re-Planning on New Information

- Trigger re-planning automatically when a new delay update arrives that affects an unassigned flight
- If a previously assigned gate becomes unavailable (e.g. maintenance), re-plan affected flights only

### 8.3 Logging & Audit Trail

- Log every assignment decision with: timestamp, flight, chosen gate, policy value estimate, alternatives considered
- Persist logs to a file or database for post-hoc analysis and model improvement

### 8.4 Model Versioning

- Tag each saved `theta` with the scenario and training run that produced it
- Support loading a specific model version at inference time
- Provide a simple benchmark script to compare two model versions on a held-out schedule

---

## Phase 9: REST API Layer

**Goal:** Expose the inference engine as a service that external systems (AODB, OPS displays, frontend) can call.

### 9.1 Core Endpoints

Build a REST API using FastAPI.

- `POST /assignments` — accepts `{schedule, airport_state, delays}`, returns assignment plan
- `GET /assignments/{job_id}` — retrieve result of an async planning job
- `POST /replan` — trigger re-planning with updated delays, returns diff of changed assignments
- `GET /status` — health check, loaded model version, last planning run timestamp

### 9.2 Input/Output Schemas

- Define Pydantic models for all request and response bodies
- Validate aircraft types, gate indices, and time values on ingestion
- Return structured errors with clear messages (unknown aircraft type, no feasible assignment, etc.)

### 9.3 Async Planning

Planning an episode takes non-trivial time at high traffic loads.

- Offload planning to a background task (FastAPI `BackgroundTasks` or a task queue)
- Return a `job_id` immediately; client polls `GET /assignments/{job_id}` for the result
- Set a configurable timeout; return best partial plan if full planning exceeds it

### 9.4 Authentication & Rate Limiting

- API key authentication for all endpoints
- Rate limiting per client to prevent runaway re-plan requests

---

## Phase 10: Frontend

**Goal:** Provide an interactive interface for operations staff to view, override, and monitor gate assignments.

### 10.1 Assignment Dashboard

- Gantt chart view of the planning window: gates on the Y-axis, time on the X-axis, flights as coloured blocks
- Colour-code by aircraft type and assignment confidence
- Live-updating as new assignments are produced by the API

### 10.2 Manual Override

- Allow an operator to drag a flight to a different gate
- On override, the system re-plans remaining unassigned flights respecting the manually set assignments
- Show estimated downstream impact (queue length, blocked gates) before confirming an override

### 10.3 Delay & Disruption View

- Highlight flights with known delays
- Show which assignments are affected by a new delay and what the system proposes to change

### 10.4 Technical Notes

- Thin client calling the Phase 9 API
- Real-time updates via WebSocket or polling the `/assignments` endpoint
- Framework: React or a lightweight alternative; keep the frontend stateless (all state lives in the API)
