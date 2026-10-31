# First task-offloading milestone

This is a separate task-level simulator layer. It does not delete or replace the
existing EdgeSimPy topology/mobility demo.

## Included
- 10 User UAVs, 3 Edge UAVs, 1 Cloud.
- K=2 nearest helper candidates and P=2 equal subtasks.
- Seven candidate destinations and 28 unordered destination-pair actions.
- Configurable Shannon-style radio rate assumptions.
- User-to-user, user-to-edge, and edge-to-cloud communication paths.
- CPU service queues, split/aggregation overhead, energy and load metrics.
- Local-only, nearest-remote-node, and greedy immediate-objective policies.
- Basic unit tests.

## Important modelling scope
This is a lightweight, discrete task-level prototype, not a complete EdgeSimPy
NetworkFlow implementation. Cloud access is routed through an Edge UAV. The
wireless parameters are explicit assumptions because the report does not fully
specify numerical radio constants. The current greedy policy uses a one-task
immediate objective; it is not a learned policy. DDQN is not included in this
first code increment.

## Run
From the repository root:
```bash
python run_demo.py
python -m unittest discover -s tests -v
```
