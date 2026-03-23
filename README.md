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
- **Linear value function approximation** — $\hat{V}(s) = \phi(s)^\top \theta$, trained via LSTD or TD(λ) updates.
- **Spatially heterogeneous service times** — Gate occupation depends on taxiing distance from the arrival runway, modelled per runway-gate pair.
- **Aircraft-gate compatibility** — Hard constraints (e.g. A380 cannot dock at narrow-body piers) and soft preferences (e.g. Pier B preferred for regionals) are encoded in the scenario config.

---

## Repository Structure

```text
stochastic-airport-systems/
├── configs/
│   ├── airports/           # Airport topology: gate count, taxi matrix, compatibility (schiphol.yaml)
│   ├── components/         # Reusable definitions: aircraft types, fleet mixes, reward structures
│   └── scenarios/          # Training and evaluation scenarios (see list below)
├── src/
│   ├── config/             # YAML loaders and frozen dataclasses for all config types
│   ├── mdp/                # MDP core: AirportState, Action, AirportEnvironment, ScheduleGenerator
│   ├── simulation/         # Simulator driver, ActiveFlight runtime state
│   ├── representation/     # StateGraph construction, spectral PVF computation
│   └── adp/                # LinearVFA, PVFFeatureExtractor, learners (LSTD, TD-λ), policies
├── experiments/            # Numbered MLOps pipeline scripts + run_pipeline.py orchestrator
├── data/
│   ├── processed/          # Saved PVF basis functions, state mappings, learned weights
│   └── schedules/          # Generated synthetic schedules
└── main.py                 # Unified CLI entry point
```

**Available scenarios** (`configs/scenarios/`):

| Scenario | Description |
|---|---|
| `master_training.yaml` | General-purpose training scenario, recommended starting point |
| `morning_rush.yaml` | High-density morning push, tests throughput under load |
| `stress_test.yaml` | Near-capacity operations, heavy contention |
| `fully_booked.yaml` | All gates in use, no slack — maximum pressure |
| `greedy_trap.yaml` | Designed to expose greedy-policy failures |
| `disrupted_evening.yaml` | Late-running flights, cascading delays |
| `toy_problems.yaml` | Small-scale sanity checks for development |

---

## Installation

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
# or: .venv\Scripts\activate    # Windows CMD / PowerShell
pip install -r requirements.txt
```

---

## CLI Tutorial

All functionality is accessed through `main.py`. The first argument is always a scenario filename (looked up in `configs/scenarios/`) or a path to a raw `.json` schedule.

```
python main.py <scenario> [action flag] [options]
```

---

### 1. Inspect a scenario

Prints the full parsed configuration — airport topology, schedule parameters, noise model, reward structure, and time horizon. Useful for verifying a config before running anything expensive.

```bash
python main.py master_training.yaml --inspect
```

---

### 2. Generate a synthetic schedule

Generates a flight schedule according to the scenario's generation parameters and saves it to `data/schedules/synthetic/`.

```bash
python main.py master_training.yaml --generate
python main.py master_training.yaml --generate --seed 123   # reproducible
```

---

### 3. View or visualize a schedule

Print the first N flights from the schedule to the terminal:

```bash
python main.py master_training.yaml --view-schedule
python main.py master_training.yaml --view-schedule --num-flights 20
```

Render a Gantt-style runway schedule plot:

```bash
python main.py master_training.yaml --visualize-schedule
python main.py master_training.yaml --visualize-schedule --seed 42
```

---

### 4. Run the full training pipeline

When no action flag is given, `main.py` routes to the pipeline orchestrator (`experiments/run_pipeline.py`). Without `--train` it runs evaluation only (assuming weights already exist).

**Train from scratch** — builds the state graph, computes PVF basis functions, trains the agent, and evaluates:

```bash
python main.py master_training.yaml --train
```

**Continue training** from a saved checkpoint (resumes epsilon decay from where it left off):

```bash
python main.py master_training.yaml --train --continue-training
# or short form:
python main.py master_training.yaml --train -c
```

**Evaluate only** (no training, loads existing weights):

```bash
python main.py master_training.yaml
```

---

### 5. Zero-shot evaluation on a different scenario

Because the policy learns airport geometry (not a specific schedule), trained weights can be applied to an unseen scenario without retraining. Use `--model` to specify which scenario's weights to load:

```bash
# Train on master_training, then test against morning_rush
python main.py master_training.yaml --train
python main.py morning_rush.yaml --model master_training
```

```bash
# Test stress_test weights against the greedy_trap scenario
python main.py greedy_trap.yaml --model stress_test
```

---

### 6. Real-time injection from a JSON schedule

If the input is a `.json` file (e.g. from an operational system), `main.py` wraps it with physics from a base config and runs the pipeline:

```bash
python main.py path/to/live_schedule.json
```

Use `--base-config` to choose which scenario's physics to apply (default: `master_training.yaml`):

```bash
python main.py path/to/live_schedule.json --base-config stress_test.yaml
```

---

### Full flag reference

| Flag | Description |
|---|---|
| `--inspect` | Print parsed scenario config and exit |
| `--generate` | Generate and save a synthetic flight schedule |
| `--view-schedule` | Print flights from the schedule to the terminal |
| `--visualize-schedule` | Render a runway schedule Gantt plot |
| `--train` | Run the training pipeline (graph → PVFs → train → evaluate) |
| `-c` / `--continue-training` | Resume training from saved checkpoint (use with `--train`) |
| `--extra-epochs <int>` | Train for this many additional episodes beyond the current checkpoint (implies `-c`) |
| `--model <prefix>` | Load weights from a different scenario (default: `master_training`) |
| `--seed <int>` | Random seed for generation tasks (default: 42) |
| `--num-flights <int>` | Number of flights to show with `--view-schedule` (default: 10) |
| `--base-config <yaml>` | Base physics config for JSON real-time injection (default: `master_training.yaml`) |

---

## Typical Workflow

```bash
# 1. Inspect your scenario
python main.py master_training.yaml --inspect

# 2. Generate a schedule and verify it
python main.py master_training.yaml --generate --seed 42
python main.py master_training.yaml --view-schedule --num-flights 15

# 3. Train the agent
python main.py master_training.yaml --train

# 4. If you want more training, continue from the checkpoint
python main.py master_training.yaml --train -c

# 5. Evaluate against a harder scenario
python main.py stress_test.yaml --model master_training
```

---

## Outputs

Results are saved to `experiments/results/`:
- **Evaluation CSVs** — Mean reward and std dev per policy (Random, Greedy, ADP)
- **Training curves** — Total reward per episode with moving average and eval checkpoints
- **Training logs** — Per-episode CSV with reward, epsilon, and wall-clock time
- **Gantt charts** — Gate assignment timeline with aircraft types and service durations
- **Spectral graph plots** — PVF embedding visualizations

---

## References

- Mahadevan & Maggioni (2007) — *Proto-value functions: A Laplacian framework for learning representation and control in Markov decision processes*
- Powell (2011) — *Approximate Dynamic Programming*

---

## Contact

For questions or collaboration: [Jeroen Fränzel](mailto:j.franzel@hotmail.com)
