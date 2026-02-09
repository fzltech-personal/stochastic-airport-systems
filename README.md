# Approximate Dynamic Programming for Stochastic Airport Operations

A research project investigating state representation methods for Approximate Dynamic Programming (ADP) in the context of airport gate assignment under uncertainty.

## Project Overview

This project studies stochastic airport scheduling problems—primarily gate assignment—using Markov Decision Processes (MDPs) and Approximate Dynamic Programming. Unlike generic queueing models, we solve a realistic variant of the problem that includes:

- **Explicit MDP formulation** incorporating **schedule-based arrivals with stochastic delays**.
- **Spatially heterogeneous service times**, where gate occupation depends on taxiing distances from specific runways.
- **Automatic state representation learning** via spectral methods (Proto-Value Functions) and Successor Features.
- **Linear value function approximation** for scalable ADP.
- **Comparison of handcrafted vs. learned basis functions** in a safety-critical domain.
- **Aircraft-gate compatibility constraints**, distinguishing physical limitations (hard constraints) from operational preferences (soft rewards).

**This is NOT a deep reinforcement learning project.** We use simulation and structured mathematical methods (Spectral Graph Theory) to maintain interpretability, safety awareness, and theoretical rigor.

---

## Repository Structure


```

airport-mdp-adp/
├── README.md
├── .gitignore
│
├── docs/                       # Thesis and documentation
│   ├── thesis.tex
│   ├── references.bib
│   ├── figures/
│   └── sections/
│
├── src/                        # Core implementation
│   ├── mdp/                    # MDP formulation (states, transitions, rewards)
│   ├── simulation/             # Trajectory generation under representative schedules
│   ├── representation/         # Basis function construction (PVFs, successor features)
│   ├── adp/                    # Value function approximation and TD learning
│   ├── utils/                  # Configuration and visualization
│   └── config/                 # Aircraft types, compatibility matrices, reward weights
│
├── experiments/                # Runnable scripts for each project stage
│   ├── 01_trajectory_generation.py
│   ├── 02_graph_construction.py
│   ├── 03_pvf_computation.py
│   └── results/                # Saved trajectories, graphs, basis functions
│
├── tests/                      # Mathematical sanity checks
│
└── notebooks/                  # Exploratory analysis (optional)

```

---

## Project Stages

### Stage 1: MDP Formulation ✓ (Recently Updated)
- **State space:** Time, gate occupancy vectors, queue of waiting flights (including runway origin **and aircraft type**).
- **Action space:** Gate assignment decisions subject to **aircraft-gate compatibility constraints**.
- **Dynamics:** 
    - **Arrivals:** Master schedule + stochastic noise (perturbations).
    - **Service:** Base service time (**type-dependent**) + **Runway-to-Gate taxiing penalty**.
- **Reward structure:** Penalties for waiting time and queue overflow; rewards for throughput **and aircraft-gate matching**.

### Stage 2: Trajectory Generation (in progress)
- Simulate episodes using **representative daily schedules** (e.g., Morning Rush, Delayed Evening).
- **Sample aircraft types** from realistic distributions for each flight.
- Collect state-action-reward-next_state tuples under baseline policies (Random, Greedy, **Compatibility-Aware Greedy**).
- Store trajectories to capture the "interaction dynamics" of the airport.

### Stage 3: State-Transition Graph Construction (In Progress)
- **Time-Collapsed Graph:** Construct a graph where nodes represent unique resource configurations $(\mathbf{g}, \mathbf{q})$ independent of absolute time.
- **Goal:** Identify recurrent operational states and bottlenecks, avoiding the "Directed Acyclic Graph" trap of pure time-series data.
- **Weighting:** Edges weighted by empirical transition probabilities or state-kernel similarity.

### Stage 4: Automatic Basis Construction (Planned)
- **Proto-Value Functions (PVFs):** Laplacian eigenfunctions of the time-collapsed graph to capture global geometry.
- **Successor Features:** Learned representations predicting future occupancy of specific **gate/runway/type** features.
- **State Aggregation:** Clustering-based dimensionality reduction.

### Stage 5: ADP with Linear VFA (Planned)
- Temporal Difference (TD) learning using the learned basis functions.
- Compare performance: **Handcrafted Heuristics** vs. **Spectral Basis** vs. **Successor Features**.

### Stage 6: Evaluation (Planned)
- Test under **Disruption Scenarios** (e.g., Runway closure, severe weather delays).
- Measure approximation quality (Bellman error) and operational metrics (Throughput, Overflow count).

### Stage 7: Analysis (Planned)
- **Interpretability:** Visualize the "eigen-states" to see if the AI detects operational bottlenecks.
- **Robustness:** Analyze policy performance on unseen schedules.

---

## Current Focus

**Active Development:** Integrating aircraft-gate compatibility into the MDP formulation and trajectory generation pipeline.

**Next Immediate Steps:**
1. Define realistic compatibility matrix $\mathbf{C}$ and preference matrix $\mathbf{P}$ for experimental scenarios
2. Implement type-aware trajectory generation with compatibility checking
3. Verify that baseline policies respect hard constraints
---

## Technical Foundation

**Mathematical Framework:**
- Finite-horizon, discrete-time MDPs.
- **Bellman equations** and dynamic programming.
- **Linear value function approximation:** $\hat{V}(s) = \phi(s)^\top \theta$, where $s = (t, \mathbf{g}, \mathbf{q})$ with $\mathbf{q}$ encoding runway origins **and aircraft types**.
- **Temporal difference learning** for parameter estimation.

**Key Methods:**
- **Proto-Value Functions:** Eigenvectors of graph Laplacian capture "geometry" of the state space.
  - *Rationale:* Smooth over frequently-visited state transitions to generalize across different schedules.
  - *Connection:* Diffusion processes, spectral graph theory.
- **Successor Features:** Decoupling dynamics (future occupancy) from reward (current goals) for transfer learning.
- **No neural network control:** Autoencoders (if used) serve ONLY for state embedding, not decision-making.

---

## Installation

```bash
# Clone repository
git clone [https://github.com/fzltech-personal/stochastic-airport-systems.git](https://github.com/fzltech-personal/stochastic-airport-systems.git)
cd airport-mdp-adp

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

**Core dependencies:**

* `numpy`, `scipy` (numerical computation, linear algebra)
* `networkx` (graph construction and spectral analysis)
* `matplotlib` (visualization)
* `pytest` (testing)

---

## Usage

### Run Experiments Sequentially

```bash
# Stage 2: Generate trajectories (using defined schedules)
python experiments/01_trajectory_generation.py

# Stage 3: Construct state-transition graph (Time-Collapsed)
python experiments/02_graph_construction.py

# Stage 4: Compute Proto-Value Functions (Eigen-decomposition)
python experiments/03_pvf_computation.py

```

Results are saved to `experiments/results/`.

### Run Tests

```bash
pytest tests/

```

---

## Data Management

* **Trajectories:** Stored as `.npy` files in `experiments/results/trajectories/`.
* Format: Array of `(state, action, reward, next_state)` tuples.


* **Graphs:** Stored as `.npy` adjacency matrices or NetworkX pickles in `experiments/results/graphs/`.
* **Basis Functions:** Stored as `.npy` arrays (each column = one basis function).

* **Configuration Files:** Aircraft type definitions, compatibility matrix $\mathbf{C}$, preference matrix $\mathbf{P}$, and reward parameters stored as JSON or YAML in `src/config/`.

**Not tracked in git:** large trajectory files, and LaTeX build artifacts (see `.gitignore`).

---

## Design Philosophy

1. **MDP first:** Explicit state/action/transition definitions, not black-box methods.
2. **Math first:** Derive, then implement; verify mathematical properties (e.g., spectral gap).
3. **Structure first:** Modular code separating concerns (MDP, simulation, representation, ADP).
4. **Minimal AI assistance:** Focus on reasoning, not auto-completion.

---

## References

Key literature foundations (to be expanded in `references.bib`):

* **Proto-Value Functions:** Mahadevan & Maggioni (2007) - "Proto-value functions: A Laplacian framework for learning representation and control in Markov decision processes"
* **Approximate Dynamic Programming:** Powell (2011) - "Approximate Dynamic Programming"
* **Successor Representations:** Dayan (1993) - "Improving generalization for temporal difference learning: The successor representation"
* **Gate Assignment Problems:** [Add relevant OR/scheduling references as you encounter them]
* **Aircraft-Gate Compatibility:** [Industry standards or academic papers on terminal design]


---

## Contact / Contribution

This is a research project. External contributions are not expected at this stage.

For questions or collaboration inquiries: [Jeroen Fränzel](mailto:j.franzel@hotmail.com)

---

## License

[Specify license, e.g., MIT, or indicate this is research code not yet licensed]
