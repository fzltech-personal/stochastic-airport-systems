# Approximate Dynamic Programming for Stochastic Airport Operations

A research project investigating state representation methods for Approximate Dynamic Programming (ADP) in the context of airport gate assignment under uncertainty.

## Project Overview

This project studies stochastic airport scheduling problems—primarily gate assignment—using Markov Decision Processes (MDPs) and Approximate Dynamic Programming. The focus is on:

- **Explicit MDP formulation** of airport operations with stochastic arrivals and service times
- **Simulation-based trajectory generation** under baseline policies
- **Automatic state representation learning** via spectral methods and successor features
- **Linear value function approximation** for scalable ADP
- **Comparison of handcrafted vs. learned basis functions**

**This is NOT a deep reinforcement learning project.** We use simulation and structured mathematical methods to maintain interpretability, safety awareness, and theoretical rigor.

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
│   ├── simulation/             # Trajectory generation under baseline policies
│   ├── representation/         # Basis function construction (PVFs, successor features)
│   ├── adp/                    # Value function approximation and TD learning
│   └── utils/                  # Configuration and visualization
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

### Stage 1: MDP Formulation ✓
- State space: time, gate occupancy, waiting flights
- Action space: gate assignment decisions
- Stochastic dynamics: arrivals, service completions
- Reward structure: waiting time penalties, throughput

### Stage 2: Trajectory Generation ✓
- Simulate episodes under simple policies (random, greedy)
- Collect state-action-reward-next_state tuples
- Store trajectories for downstream representation learning

### Stage 3: State-Transition Graph Construction (In Progress)
- Build graph from observed state transitions
- Weight edges by empirical transition probabilities
- Prepare for spectral analysis

### Stage 4: Automatic Basis Construction (Planned)
- **Proto-Value Functions (PVFs)**: Laplacian eigenfunctions of state-transition graph
- **Successor Features**: Expected discounted state occupancy
- **State Aggregation**: Clustering-based dimensionality reduction

### Stage 5: ADP with Linear VFA (Planned)
- Temporal difference learning with learned basis functions
- Compare performance: handcrafted vs. automatic features

### Stage 6: Evaluation (Planned)
- Test under congestion and disruption scenarios
- Measure approximation quality (Bellman error, policy performance)

### Stage 7: Analysis (Planned)
- Interpretability of learned representations
- Robustness and scalability analysis

---

## Technical Foundation

**Mathematical Framework:**
- Finite-horizon, discrete-time MDPs
- Bellman equations and dynamic programming
- Linear value function approximation: $\hat{V}(s) = \phi(s)^\top \theta$
- Temporal difference learning for parameter estimation

**Key Methods:**
- **Proto-Value Functions**: Eigenvectors of graph Laplacian capture "geometry" of state space
  - Rationale: Smooth over frequently-visited state transitions
  - Connection: Diffusion processes, spectral graph theory
- **Successor Features**: State occupancy representations for transfer learning
- **No neural network control**: If autoencoders are used, they serve ONLY for state embedding, not decision-making

---

## Installation

```bash
# Clone repository
git clone https://github.com/fzltech-personal/stochastic-airport-systems.git
cd airport-mdp-adp

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies:**
- `numpy`, `scipy` (numerical computation, linear algebra)
- `networkx` (graph construction and analysis)
- `matplotlib` (visualization)
- `pytest` (testing)

---

## Usage

### Run Experiments Sequentially

```bash
# Stage 2: Generate trajectories
python experiments/01_trajectory_generation.py

# Stage 3: Construct state-transition graph
python experiments/02_graph_construction.py

# Stage 4: Compute Proto-Value Functions
python experiments/03_pvf_computation.py
```

Results are saved to `experiments/results/`.

### Run Tests

```bash
pytest tests/
```

---

## Data Management

- **Trajectories**: Stored as `.npy` files in `experiments/results/trajectories/`
  - Format: Array of `(state, action, reward, next_state)` tuples
- **Graphs**: Stored as `.npy` adjacency matrices or NetworkX pickles in `experiments/results/graphs/`
- **Basis Functions**: Stored as `.npy` arrays (each column = one basis function)

**Not tracked in git**: `thesis.pdf`, large trajectory files (see `.gitignore`)

---

## Design Philosophy

1. **MDP first**: Explicit state/action/transition definitions, not black-box methods
2. **Math first**: Derive, then implement; verify mathematical properties
3. **Structure first**: Modular code separating concerns (MDP, simulation, representation, ADP)
4. **Minimal AI assistance**: Focus on reasoning, not auto-completion

---

## References

Key literature foundations (to be expanded in `references.bib`):

- **Proto-Value Functions**: Mahadevan & Maggioni (2007) - "Proto-value functions: A Laplacian framework for learning representation and control in Markov decision processes"
- **Approximate Dynamic Programming**: Powell (2011) - "Approximate Dynamic Programming"
- **Successor Representations**: Dayan (1993) - "Improving generalization for temporal difference learning: The successor representation"

---

## Contact / Contribution

This is a research project. External contributions are not expected at this stage.

For questions or collaboration inquiries: [Add contact information if desired]

---

## License

[Specify license, e.g., MIT, or indicate this is research code not yet licensed]
