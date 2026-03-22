# Approximate Dynamic Programming for Stochastic Airport Gate Assignment

A research and engineering project building a practical gate assignment system for airport operations under uncertainty. The system learns optimal gate assignment policies using Approximate Dynamic Programming (ADP) and can be queried at runtime with a real schedule, the current airport state, and known flight delays to produce optimal gate assignments for incoming flights.

---

## What It Does

Given:
- A **flight schedule** for the coming hours (importable from operational systems)
- The **current state of the airport** (which gates are occupied and for how long)
- Any **known delays** for inbound flights

The system outputs an optimal gate assignment plan by running a trained ADP policy over the incoming traffic window. Unlike greedy heuristics that assign the best available gate right now, the policy reasons about future resource contention — holding back a gate for an incoming wide-body rather than filling it with a regional flight that will block the pier.

---

## Technical Approach

**This is not a deep reinforcement learning project.** The system uses structured mathematical methods to maintain interpretability and theoretical rigor:

- **MDP formulation** — State $S_t = (t, \mathbf{g}_t, \mathbf{q}_t)$ where $\mathbf{g}$ is the gate occupancy vector and $\mathbf{q}$ is the runway queue composition.
- **Proto-Value Functions (PVFs)** — Eigenvectors of the graph Laplacian, computed once from random exploration, capture the geometric structure of the state space without handcrafted features. This enables zero-shot transfer to unseen traffic patterns.
- **Linear value function approximation** — $\hat{V}(s) = \phi(s)^\top \theta$, trained via TD(0) semi-gradient updates.
- **Spatially heterogeneous service times** — Gate occupation depends on taxiing distance from the arrival runway, modelled per runway-gate pair.
- **Aircraft-gate compatibility** — Hard constraints (e.g. A380 cannot dock at narrow-body piers) and soft preferences (e.g. Pier B preferred for regionals) are encoded in the scenario config.

---

## Repository Structure

```text
stochastic-airport-systems/
├── configs/
│   ├── airports/           # Airport topology: gate count, taxi matrix, compatibility (schiphol.yaml)
│   ├── components/         # Reusable definitions: aircraft types, fleet mixes, reward structures
│   └── scenarios/          # Training and evaluation scenarios (master_training, morning_rush, ...)
├── src/
│   ├── config/             # YAML loaders and frozen dataclasses for all config types
│   ├── mdp/                # MDP core: AirportState, Action, AirportEnvironment, ScheduleGenerator
│   ├── simulation/         # Simulator driver, ActiveFlight runtime state
│   ├── representation/     # StateGraph construction, spectral PVF computation
│   └── adp/                # LinearVFA, PVFFeatureExtractor, TD0Learner, policies
├── experiments/            # Numbered MLOps pipeline scripts + run_pipeline.py orchestrator
├── data/
│   ├── processed/          # Saved PVF basis functions, state mappings, learned weights
│   └── schedules/          # Generated synthetic schedules
└── main.py                 # Unified CLI entry point
```

---

## Installation

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Training a model

Run the full pipeline — graph construction, PVF computation, TD training, evaluation:

```bash
python main.py master_training.yaml --run-pipeline --train
```

Or step by step:
```bash
python experiments/01_generate_features.py master_training.yaml
python experiments/03_train_agent.py master_training.yaml
python experiments/04_evaluate_policy.py master_training.yaml
```

### Zero-shot evaluation on a different scenario

Because the policy learns airport geometry (not a specific schedule), it can be evaluated on unseen traffic without retraining:

```bash
python main.py morning_rush.yaml --run-pipeline --model master_training
```

### Inspecting a scenario

```bash
python main.py configs/scenarios/stress_test.yaml --inspect
python main.py configs/scenarios/stress_test.yaml --generate --seed 42
```

---

## Outputs

Results are saved to `experiments/results/`:
- **Evaluation CSVs** — Mean reward and std dev per policy (Random, Greedy, ADP)
- **Training curves** — Total reward per episode with moving average
- **Gantt charts** — Gate assignment timeline with aircraft types and service durations
- **Spectral graph plots** — PVF embedding visualizations

---

## References

- Mahadevan & Maggioni (2007) — *Proto-value functions: A Laplacian framework for learning representation and control in Markov decision processes*
- Powell (2011) — *Approximate Dynamic Programming*

---

## Contact

For questions or collaboration: [Jeroen Fränzel](mailto:j.franzel@hotmail.com)
