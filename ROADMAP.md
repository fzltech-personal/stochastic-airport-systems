### Architectural Summary

This roadmap transforms the current "Static Configuration System" into a "Dynamic Stochastic Optimization Engine."

The architecture follows a strict Separation of Concerns:
1.  **MDP Layer:** Pure mathematical rules of the airport (State, Transition, Reward).
2.  **Simulation Layer:** The engine that drives time forward and manages stochastic realizations (Random variables).
3.  **Representation Layer:** The offline learner that compresses the state space into basis functions (Spectral/Graph theory).
4.  **ADP Layer:** The online learner that estimates value functions using those basis functions (Bellman equations).

---

### Phase 1: Foundation & Refactoring

**Goal:** Prepare the data structures for dynamic simulation by separating "Configuration" from "Runtime State" and fixing technical debt in resource loading.

#### Step 1.1: Split Flight Entity
**Target Files:**
* `src/mdp/components/flight.py` (Modify)
* `src/simulation/realization.py` (Create)

**Architectural Actions:**
* Rename `Flight` to `ScheduledFlight`. Make it fully immutable (`frozen=True`). This represents the Master Schedule.
* Create `ActiveFlight` (mutable). This represents a flight instance in the simulation. It wraps a `ScheduledFlight` but adds runtime attributes: `current_delay`, `actual_arrival_time`, `gate_assignment`, `status` (AIRBORNE, QUEUED, SERVICING, COMPLETED).

**Mathematical/OR Implementation:**
* Differentiation between parameter $t_{sched}$ (deterministic input) and random variable $T_{actual} = t_{sched} + \xi$ (stochastic realization).

#### Step 1.2: Pathing & Resource Injection
**Target Files:**
* `src/config/loader.py` (Modify)
* `src/utils/paths.py` (Create)

**Architectural Actions:**
* Remove hardcoded relative paths (`../../`).
* Implement a `ProjectPaths` singleton or dependency injection mechanism to locate `configs/` and `data/` reliably from any execution context (IDE, CLI, Docker).

---

### Phase 2: The Core MDP Engine

**Goal:** Implement the mathematical definition of the system ($S, A, T, R$).

#### Step 2.1: Define the State Space
**Target Files:**
* `src/mdp/state.py` (Implement)

**Architectural Actions:**
* Create class `AirportState`.
* **Attributes:**
    * `t` (int): Current timestep.
    * `gates` (np.array): Vector of size $N_{gates}$. Value = remaining service time (0 = free).
    * `runway_queue` (List[ActiveFlight]): Ordered list of aircraft waiting for gates.
    * `arrivals` (List[ActiveFlight]): Incoming flights not yet in the queue (horizon view).
* Implement `__hash__` and `__eq__` to allow states to be used as graph nodes later.

**Mathematical/OR Implementation:**
* $S_t = (t, \mathbf{g}_t, \mathbf{q}_t)$
* $\mathbf{g}_t \in \mathbb{N}^{N_{gates}}$ where $g_i$ is residual time.

#### Step 2.2: Define the Action Space
**Target Files:**
* `src/mdp/action.py` (Implement)

**Architectural Actions:**
* Create class `Action`.
* **Structure:** A mapping or tuple `(flight_id, gate_idx)`.
* Implement `ActionSpace` generator that filters invalid actions based on `CompatibilityConfig` (Hard Constraints).

**Mathematical/OR Implementation:**
* $A_t \subseteq \mathcal{A}$
* Constraint checks: $C_{flight, gate} = 1$ (Compatible).

#### Step 2.3: The Transition Logic (Step Function)
**Target Files:**
* `src/mdp/environment.py` (Implement)

**Architectural Actions:**
* Implement `AirportEnvironment`.
* **Method `step(action)`:**
    1.  **Apply Action:** Move flight from queue to gate. Set gate timer.
    2.  **Dynamics:** Decrement all gate timers. Remove finished flights.
    3.  **Stochasticity:** Ingest new arrivals from the schedule based on $t$.
    4.  **Reward:** Calculate $R_t$.
    5.  **Return:** `(next_state, reward, done, info)`.

**Mathematical/OR Implementation:**
* $S_{t+1} = f(S_t, A_t, W_{t+1})$
* $W_{t+1}$ represents the stochastic arrival process.

---

### Phase 3: The Simulation Loop

**Goal:** Generate data. We need "Experiences" (Trajectories) to build the graph for the spectral methods.

#### Step 3.1: The Simulator Driver
**Target Files:**
* `src/simulation/simulator.py` (Create)

**Architectural Actions:**
* Create `Simulator` class.
* Accepts: `ScenarioConfig`, `Agent` (Policy).
* **Loop:** Initialize `AirportEnvironment`. While $t < T_{max}$: get action from Agent, call `env.step()`, record tuple.

**Mathematical/OR Implementation:**
* Monte Carlo simulation of sample path $\omega$.
* Trajectory $\tau = \{(s_0, a_0, r_0), (s_1, a_1, r_1), \dots\}$.

#### Step 3.2: Baseline Policies
**Target Files:**
* `src/adp/policies.py` (Create)

**Architectural Actions:**
* Implement `RandomPolicy` (valid random moves).
* Implement `GreedyPolicy` (assign to first compatible gate, minimize immediate $c_{wait}$).
* *Note: These are needed to explore the state space and generate the "Graph" for Phase 4.*

---

### Phase 4: Automatic Feature Creation (Spectral Methods)

**Goal:** Convert the high-dimensional state space into low-dimensional basis functions $\phi(s)$ using Spectral Graph Theory (Proto-Value Functions).

#### Step 4.1: State-Transition Graph Builder
**Target Files:**
* `src/representation/graph_builder.py` (Create)
* `experiments/02_graph_construction.py` (Create)

**Architectural Actions:**
* Run `Simulator` $N$ times with `RandomPolicy`.
* Collect all unique visited "Resource States" $(\mathbf{g}, \mathbf{q})$ (ignoring time $t$ to allow cycles/recurrence).
* Build Adjacency Matrix $W$ where $W_{ij} = 1$ if transition $i \to j$ was observed.

**Mathematical/OR Implementation:**
* Graph $G = (V, E)$.
* Nodes $V$ are unique configurations of resources.
* Time-collapse: $S_t \to \tilde{S}$ to ensure the graph isn't a DAG (Directed Acyclic Graph).

#### Step 4.2: Eigen-Decomposition (PVFs)
**Target Files:**
* `src/representation/spectral.py` (Create)
* `experiments/03_pvf_computation.py` (Create)

**Architectural Actions:**
* Compute Graph Laplacian: $L = D - W$ (unnormalized) or $\mathcal{L} = I - D^{-1/2}WD^{-1/2}$ (normalized).
* Perform SVD/Eigen-decomposition to find eigenvectors.
* Store the top $k$ eigenvectors. These are your Basis Functions $\phi(s)$.

**Mathematical/OR Implementation:**
* Solving $L\mathbf{v} = \lambda\mathbf{v}$.
* The eigenvectors capture the "geometry" of the state space (e.g., bottlenecks, clusters of similar states).

---

### Phase 5: The ADP Solver

**Goal:** Learn the Value Function $V(s)$ using the basis functions created in Phase 4.

#### Step 5.1: Linear Value Function Approximation
**Target Files:**
* `src/adp/value_function.py` (Create)

**Architectural Actions:**
* Class `LinearValueFunction`.
* Stores weights vector $\theta$.
* **Method `estimate(state)`:** Returns $\theta^T \cdot \phi(state)$.
* **Crucial:** Need a mapping from a runtime state to its corresponding row index in the eigenvector matrix from Phase 4. (Nearest Neighbor or Exact Match).

**Mathematical/OR Implementation:**
* $V(s) \approx \sum_{i=1}^k \theta_i \phi_i(s)$

#### Step 5.2: TD Learning Agent
**Target Files:**
* `src/adp/agent.py` (Create)

**Architectural Actions:**
* Implement `TDLearningAgent`.
* **Update Rule:** Standard TD(0) or Least Squares TD (LSTD).
* Update $\theta$ based on the Bellman Error.

**Mathematical/OR Implementation:**
* $\delta_t = R_t + \gamma V(S_{t+1}) - V(S_t)$
* $\theta \leftarrow \theta + \alpha \delta_t \phi(S_t)$

#### Step 5.3: Evaluation
**Target Files:**
* `experiments/04_adp_training.py` (Create)

**Architectural Actions:**
* Train the agent over $M$ episodes.
* Compare cumulative reward of `TDLearningAgent` vs. `GreedyPolicy` on held-out test schedules.