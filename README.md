# UAV-Assisted Edge Computing: RL-Based Task Splitting and Multi-Level Offloading

Implementation of our mid-semester report: a centralized DDQN-based system for
deciding where subtasks of a UAV-generated task should execute (local User
UAV, helper User UAV, Edge UAV, or Cloud), using EdgeSimPy as the simulation
framework.

## Repository structure

- `src/topology/` — network topology, UAV mobility, and link setup
  (EdgeSimPy `User`, `EdgeServer`, `NetworkLink` configuration).
- `src/wireless/` — wireless rate/channel model (Eq. 11 in the report).
- `src/tasks/` — task generation, splitting into subtasks, and aggregation.
- `src/metrics/` — latency, energy, and load-imbalance calculations.
- `src/rl/` — state/action representation and the DDQN agent.
- `src/baselines/` — local-only, nearest-helper, greedy-cost, and
  literature-inspired baselines.
- `data/` — EdgeSimPy JSON datasets.
- `notebooks/` — exploratory/Colab notebooks.
- `tests/` — unit tests for each module.
- `docs/` — report, problem statement, and design notes.

## Setup

```
python -m pip install --requirement requirements.txt
```

## Status

- [x] Topology, UAV mobility, dynamic link bandwidth/delay (`src/topology/`)
- [ ] Task generation and splitting (`src/tasks/`)
- [ ] Latency/energy/load metrics (`src/metrics/`)
- [ ] State/action representation and DDQN (`src/rl/`)
- [ ] Baselines (`src/baselines/`)
- [ ] Scalability experiments

## Implementation Progress

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for the current
implementation status, completed components, testing, and pending work.