from src.tasksim.simulator import Config, OffloadingSimulator


def main():
    cfg = Config(task_count=40, seed=7)
    sim = OffloadingSimulator(cfg)

    print("UAV EDGE-OFFLOADING FIRST MILESTONE")
    print(f"Users={cfg.n_users}, Edge UAVs={cfg.n_edges}, Cloud=1")
    print(f"Helpers/source={cfg.helpers_per_source}, subtasks={cfg.subtasks}")
    print(f"Candidate destinations={len(sim.candidates('U0'))}")
    print(f"Unordered destination-pair actions={len(sim.action_pairs('U0'))}")
    print("\nPolicy comparison (same task sequence and simulator model)")
    for policy in ("local", "nearest", "greedy"):
        results = sim.run_policy(policy)
        summary = sim.summarize(results)
        print(f"\n{policy.upper()}")
        for key, value in summary.items():
            print(f"  {key}: {value:.6f}" if isinstance(value, float)
                  else f"  {key}: {value}")
    print("\nExample task from greedy policy:")
    sample = sim.run_policy("greedy")[0]
    print(f"  Task {sample.task_id}, source={sample.source_id}")
    print(f"  destinations={sample.destinations}")
    print(f"  latency={sample.latency_s:.6f}s, energy={sample.energy_j:.6f}J")
    print(f"  deadline_met={sample.deadline_met}, load_imbalance={sample.load_imbalance:.6f}")


if __name__ == "__main__":
    main()
