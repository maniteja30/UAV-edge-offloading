# Milestone 2A — report-aligned workload and base configuration

This patch updates the first milestone's task-level simulator without changing
or replacing the original EdgeSimPy topology demonstration.

## Changes

- Uses the report's base values: 10 User UAVs, 3 Edge UAVs, K=2 helpers,
  P=2 subtasks, 2 MB input, 0.1 MB output, 5e8 CPU cycles, 256 MB memory,
  2 s deadline, 60 s episode, and 0.5 tasks/s/User.
- Aligns CPU capacities to User=4 GHz, Edge=10 GHz, Cloud=100 GHz.
- Aligns per-active-link bandwidth to 20 MHz.
- Default task generation is an independent Poisson process for each User UAV.
  The expected number of arrivals in the base episode is 300, not a fixed count.
- Keeps a deterministic 40-task trace for quick smoke tests via `--smoke`.
- Places UAV coordinates within the report's 500 m x 500 m area and uses the
  configured 100 m altitude in distance calculations.
- Corrects P95 to use the nearest-rank percentile definition.

## Explicit assumptions still present

The report does not supply all numerical radio constants, battery capacities,
or memory capacities. Those remain configurable assumptions in `Config`. The
CPU energy coefficient and the current energy/load normalization references
are also provisional assumptions; the report requires fixed references to be
calibrated from a common baseline set before DDQN training.

The current simulator still does not implement link-bandwidth sharing, link
queues, dynamic battery accounting/action masking, or complete shortest-feasible
U-U/U-E/E-E/E-C routing. Its load imbalance remains an episode-average CPU
utilization approximation, not event-time sampling. These are the next
Milestone 2B items; do not describe them as completed by this patch.

## Run from the repository root

```bash
python3 run_demo.py --smoke
python3 run_demo.py
python3 -m unittest discover -s tests -v
```

The normal demo uses a seeded Poisson trace. For 10 users, lambda=0.5 tasks/s
per user, and 60 seconds, the expected task count is 300; the realized count
varies with the seed.
