# Approximate Dynamic Programming for Stochastic Airport Operations

A research project investigating state representation methods for Approximate Dynamic Programming (ADP) in the context of airport gate assignment under uncertainty.

## Project Overview

This project studies stochastic airport scheduling problems—primarily gate assignment—using Markov Decision Processes (MDPs) and Approximate Dynamic Programming. Unlike generic queueing models, we solve a realistic variant of the problem that includes:

- **Explicit MDP formulation** incorporating **stochastic schedule generation** (Poisson arrival rates, deterministic capacity limits, normal peaks).
- **Spatially heterogeneous service times**, where gate occupation depends on taxiing distances from specific runways.
- **Automatic state representation learning** via spectral methods (Proto-Value Functions) to map the topological geometry of the airport without hardcoded rules.
- **Linear value function approximation** for scalable ADP using Temporal Difference (TD) learning.
- **Aircraft-gate compatibility constraints**, distinguishing physical limitations (e.g., A380 Super-Heavies) from operational preferences (e.g., Pier proximity).
- **Automated MLOps Pipeline**, decoupling environment simulation from AI training to allow for zero-shot evaluation on unseen traffic schedules.

**This is NOT a deep reinforcement learning project.** We use stochastic simulation and structured mathematical methods (Spectral Graph Theory) to maintain interpretability, safety awareness, and theoretical rigor.

---

## Repository Structure

```text
airport-mdp-adp/
├── README.md
├── main.py                     # Unified CLI for Data Engineering and MLOps
│
├── configs/                    # YAML/JSON configurations
│   ├── airports/               # Airport layouts, delta matrices (e.g., schiphol.yaml)
│   ├── components/             # Aircraft types, compatibility matrices, rewards
│   └── scenarios/              # Traffic scenarios (stress_test, morning_rush, greedy_trap)
│
├── src/                        # Core implementation
│   ├── mdp/                    # MDP formulation, states, and stochastic schedule generators
│   ├── simulation/             # Trajectory generation and environment stepping
│   ├── representation/         # Spectral graph builder and PVF computation
│   └── adp/                    # Value function approximation, TD learning, Baselines
│
├── experiments/                # MLOps Pipeline Scripts
│   ├── run_pipeline.py         # Master Orchestrator
│   ├── 01_generate_features.py # Random walks to build state-transition graphs
│   ├── 02_visualize_features.py# Spectral embedding visualization
│   ├── 03_train_agent.py       # TD(0) learning loop
│   ├── 04_evaluate_policy.py   # Baseline vs. ADP metrics evaluation
│   └── 05_visualize_schedule.py# Output Gantt chart generation
│
└── data/processed/             # Saved weights (.npy), graphs (.pkl), and basis functions

```

---

## Technical Foundation

**Mathematical Framework:**

* Finite-horizon, discrete-time MDPs.
* **Bellman equations** and dynamic programming.
* **Linear value function approximation:** $\hat{V}(s) = \phi(s)^\top \theta$, where $s = (t, \mathbf{g}, \mathbf{q})$ with $\mathbf{q}$ encoding runway origins and aircraft types.
* **Temporal difference learning** for parameter estimation.

**Key Methods:**

* **Proto-Value Functions (PVFs):** Eigenvectors of the graph Laplacian capture the "geometry" of the state space. This smooths over frequently-visited state transitions to generalize across different schedules.
* **Resource Contention:** Training forces the agent to learn the difference between immediate rewards (Greedy) and future bottlenecks (e.g., the "Super-Heavy Trap").

---

## Installation

Ensure you have Python 3.9+ installed.

```bash
# Clone repository
git clone [https://github.com/fzltech-personal/stochastic-airport-systems.git](https://github.com/fzltech-personal/stochastic-airport-systems.git)
cd stochastic-airport-systems

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

---

## 🛠️ CLI Tutorial & Usage

The entire framework is driven by the root `main.py` command-line interface. It acts as a Swiss Army Knife for both Data Engineering (inspecting environments) and MLOps (training and evaluating AI).

### Part 1: Data Engineering & Inspection

Before running complex AI models, you can safely inspect and visualize the stochastic environments to ensure your scenario math is correct.

**1. Inspect a scenario configuration:**
Validates your YAML file and prints the loaded configuration.

```bash
python main.py configs/scenarios/stress_test.yaml --inspect

```

**2. Generate and preview a synthetic schedule:**
Runs the stochastic generator (evaluating Poisson rates or capacity limits) and prints the chronological flight list.

```bash
python main.py configs/scenarios/stress_test.yaml --generate --seed 99

```

**3. Visualize the raw schedule constraints:**

```bash
python main.py configs/scenarios/stress_test.yaml --visualize-schedule

```

### Part 2: MLOps & AI Training

Once the environment is verified, use the `--run-pipeline` flag to hand control over to the MLOps orchestrator. This executes the 5-step pipeline: Feature Generation $\to$ Visualization $\to$ TD Training $\to$ Baseline Evaluation $\to$ Gantt Chart Plotting.

**Train a new AI agent from scratch:**
Pass the scenario filename and the `--train` flag. The AI will explore the airport, learn the geometry, train its weights ($\theta$), and output the final performance metrics.

```bash
python main.py stress_test.yaml --run-pipeline --train

```

### Part 3: Zero-Shot Evaluation

Because the AI learns the geometry of the airport (not just the schedule), you can evaluate a trained brain on completely unseen traffic scenarios without retraining it.

**Evaluate a new scenario using a pre-trained brain:**
Use the `--model` flag to point the evaluation scripts to the weights you trained in Part 2.

```bash
python main.py morning_rush.yaml --run-pipeline --model stress_test

```

---

## 📊 Outputs & Artifacts

Running the pipeline automatically pits the ADP Agent against a Random baseline and a Myopic Greedy heuristic. Results are saved to `experiments/results/`, including:

1. **Evaluation CSVs:** A breakdown of the mean reward and standard deviation (proving the ADP agent outsmarts greedy heuristics during resource contention).
2. **Spectral Graph Plots:** 2D visualizations of the Fiedler vector and proto-value functions.
3. **Training Curves:** Visualizing the agent's total reward over episodes.
4. **Gantt Charts:** A color-coded timeline showing exact gate assignments, aircraft types, and stochastic service turnaround times.

---

## References

* **Proto-Value Functions:** Mahadevan & Maggioni (2007) - "Proto-value functions: A Laplacian framework for learning representation and control in Markov decision processes"
* **Approximate Dynamic Programming:** Powell (2011) - "Approximate Dynamic Programming"

---

## Contact

This is a research project. External contributions are not expected at this stage.
For questions or collaboration inquiries: [Jeroen Fränzel](mailto:j.franzel@hotmail.com)
