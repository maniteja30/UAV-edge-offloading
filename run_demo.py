import argparse

from src.tasksim.simulator import Config, OffloadingSimulator


def main():
    parser = argparse.ArgumentParser(description="UAV edge-offloading simulation demo")
    parser.add_argument("--smoke", action="store_true",
                        help="use a small deterministic 40-task trace instead of Poisson arrivals")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = Config(seed=args.seed, task_count=40 if args.smoke else None)
    sim = OffloadingSimulator(cfg)

    print("UAV EDGE-OFFLOADING MILESTONE 2A")
    print(f"Users={cfg.n_users}, Edge UAVs={cfg.n_edges}, Cloud=1")
    print(f"Episode={cfg.episode_duration_s:.1f}s, arrival rate={cfg.arrival_rate_per_user:.2f} tasks/s/User")
    print(f"Generated tasks={len(sim.tasks)} (Poisson arrivals unless --smoke is used)")
    print(f"Helpers/source={cfg.helpers_per_source}, subtasks={cfg.subtasks}")
    print(f"Candidate destinations={len(sim.candidates('U0'))}")
    print(f"Unordered destination-pair actions={len(sim.action_pairs('U0'))}")
    print("Base CPU: User=4 GHz, Edge=10 GHz, Cloud=100 GHz; bandwidth=20 MHz")
    print("\nPolicy comparison (same generated task trace)")
    for policy in ("local", "nearest", "greedy"):
        summary = sim.summarize(sim.run_policy(policy))
        print(f"\n{policy.upper()}")
        for key, value in summary.items():
            print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
