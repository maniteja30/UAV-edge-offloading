import argparse
import csv
import random
import statistics
from collections import Counter
from math import ceil, isfinite
from pathlib import Path

import numpy as np
import torch

from tasksim.simulator import Config, OffloadingSimulator
from tasksim.rl_env import OffloadingRLEnv
from rl.dqn_agent import DQNAgent


POLICIES = ["local", "nearest", "greedy", "dqn"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DQN against UAV offloading baselines"
    )

    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--task-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=100)

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained DQN checkpoint",
    )

    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=0.5,
        help="Task arrival rate per user (tasks/second)",
    )

    parser.add_argument(
        "--cpu-cycles",
        type=float,
        default=5e8,
        help="CPU cycles required per task",
    )

    parser.add_argument(
        "--deadline",
        type=float,
        default=2.0,
        help="Task deadline in seconds",
    )

    parser.add_argument(
        "--energy-coefficient",
        type=float,
        default=1e-28,
        help="CPU energy coefficient used by the simulator",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional name for this evaluation's output files",
    )

    return parser.parse_args()


def finite_mean(values):
    valid = [
        float(x) for x in values
        if isfinite(x)
    ]

    return (
        statistics.mean(valid)
        if valid else float("nan")
    )


def calculate_metrics(results):
    """Calculate metrics without allowing failed tasks to hide."""

    total = len(results)

    if total == 0:
        return {
            "tasks": 0,
            "mean_latency_s": float("nan"),
            "p95_latency_s": float("nan"),
            "mean_energy_j": float("nan"),
            "deadline_violation_rate": 0.0,
            "success_rate": 0.0,
            "mean_load_imbalance": float("nan"),
            "mean_objective": float("nan"),
            "infeasible_rate": 0.0,
        }

    feasible = [
        r for r in results
        if isfinite(r.latency_s)
        and isfinite(r.energy_j)
        and isfinite(r.objective)
    ]

    latencies = sorted(
        r.latency_s for r in feasible
    )

    energies = [
        r.energy_j for r in feasible
    ]

    loads = [
        r.load_imbalance for r in feasible
    ]

    objectives = [
        r.objective for r in feasible
    ]

    p95 = (
        latencies[
            max(0, ceil(0.95 * len(latencies)) - 1)
        ]
        if latencies else float("nan")
    )

    successes = sum(
        r.deadline_met for r in results
    )

    return {
        "tasks": total,
        "mean_latency_s": finite_mean(latencies),
        "p95_latency_s": p95,
        "mean_energy_j": finite_mean(energies),
        "deadline_violation_rate": (
            1.0 - successes / total
        ),
        "success_rate": successes / total,
        "mean_load_imbalance": finite_mean(loads),
        "mean_objective": finite_mean(objectives),
        "infeasible_rate": (
            1.0 - len(feasible) / total
        ),
    }


def evaluate_dqn(config, agent, seed):
    """
    Evaluate the trained DQN using greedy action selection.

    Also record how frequently each destination pair is selected.
    """

    env = OffloadingRLEnv(config)
    state, info = env.reset(seed=seed)

    results = []
    action_counts = Counter()

    while not env.done:
        action_mask = np.asarray(
            env.get_action_mask(),
            dtype=bool,
        )

        action = agent.select_action(
            state,
            action_mask,
            epsilon=0.0,
        )

        if action is None:
            action_counts["NO_FEASIBLE_ACTION"] += 1
        else:
            selected_pair = env.get_action_pairs()[action]

            pair_label = (
                f"{selected_pair[0]} + {selected_pair[1]}"
            )

            action_counts[pair_label] += 1

        next_state, reward, terminated, truncated, step_info = (
            env.step(action)
        )

        results.append(env.results[-1])
        state = next_state

        if terminated or truncated:
            break

    return results, action_counts


def save_csv(rows, path, fields):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def save_comparison_plot(summary_rows, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Matplotlib unavailable; skipping comparison graph."
        )
        return

    metrics = [
        ("mean_latency_s", "Mean Latency (s)"),
        ("mean_energy_j", "Mean Energy (J)"),
        ("success_rate", "Success Rate"),
        ("mean_objective", "Mean Objective"),
    ]

    labels = [
        row["policy"].upper()
        for row in summary_rows
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8),
    )

    for ax, (key, title) in zip(axes.flat, metrics):
        values = [
            row[key] if isfinite(row[key]) else np.nan
            for row in summary_rows
        ]

        ax.bar(labels, values)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    if args.task_count is not None and args.task_count <= 0:
        raise ValueError("task-count must be positive")

    if args.arrival_rate <= 0:
        raise ValueError("arrival-rate must be positive")

    if args.cpu_cycles <= 0:
        raise ValueError("cpu-cycles must be positive")

    if args.deadline <= 0:
        raise ValueError("deadline must be positive")

    if args.energy_coefficient <= 0:
        raise ValueError("energy-coefficient must be positive")

    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("UAV TASK-OFFLOADING POLICY EVALUATION")
    print("=" * 70)

    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Episodes: {args.episodes}")
    print(
        f"Arrival rate per user: {args.arrival_rate} tasks/s"
    )
    print(
        f"CPU cycles per task: {args.cpu_cycles:.2e}"
    )
    print(f"Task deadline: {args.deadline} s")
    print(
        f"CPU energy coefficient: "
        f"{args.energy_coefficient:.2e}"
    )

    print(
        "Task generation:",
        "Poisson arrivals"
        if args.task_count is None
        else f"{args.task_count} fixed tasks",
    )

    print("=" * 70)

    if args.task_count is not None:
        print(
            "Note: fixed task_count produces the same task trace "
            "across episodes in the current simulator."
        )

    # Initialize DQN using the environment's dimensions.
    initial_config = Config(
        task_count=args.task_count,
        seed=args.seed,
        arrival_rate_per_user=args.arrival_rate,
        cpu_cycles=args.cpu_cycles,
        deadline_s=args.deadline,
        cpu_energy_coefficient=args.energy_coefficient,
    )

    initial_env = OffloadingRLEnv(initial_config)

    _, info = initial_env.reset(seed=args.seed)

    agent = DQNAgent(
        state_size=info["state_size"],
        action_size=info["action_size"],
        device=device,
        seed=args.seed,
    )

    agent.load(str(model_path))

    print(f"Loaded DQN model: {model_path}")
    print(
        f"State size: {info['state_size']} | "
        f"Action size: {info['action_size']}"
    )

    # Store task results for pooled evaluation.
    all_results = {
        policy: [] for policy in POLICIES
    }

    total_action_counts = Counter()
    episode_rows = []

    for episode in range(args.episodes):
        episode_seed = args.seed + episode

        config = Config(
            task_count=args.task_count,
            seed=episode_seed,
            arrival_rate_per_user=args.arrival_rate,
            cpu_cycles=args.cpu_cycles,
            deadline_s=args.deadline,
            cpu_energy_coefficient=args.energy_coefficient,
        )

        # Every policy receives the same task trace
        # and initial resources.
        simulator = OffloadingSimulator(config)

        for policy in ("local", "nearest", "greedy"):
            results = simulator.run_policy(policy)

            all_results[policy].extend(results)

            metrics = calculate_metrics(results)

            episode_rows.append({
                "episode": episode + 1,
                "seed": episode_seed,
                "policy": policy.upper(),
                **metrics,
            })

        dqn_results, action_counts = evaluate_dqn(
            config,
            agent,
            episode_seed,
        )

        all_results["dqn"].extend(dqn_results)
        total_action_counts.update(action_counts)

        metrics = calculate_metrics(dqn_results)

        episode_rows.append({
            "episode": episode + 1,
            "seed": episode_seed,
            "policy": "DQN",
            **metrics,
        })

        local_metrics = calculate_metrics(
            all_results["local"][-len(simulator.tasks):]
        )

        greedy_metrics = calculate_metrics(
            all_results["greedy"][-len(simulator.tasks):]
        )

        print(
            f"Episode {episode + 1:3d}/{args.episodes} "
            f"| Tasks: {len(simulator.tasks):4d} "
            f"| LOCAL: {local_metrics['success_rate']:.3f} "
            f"| GREEDY: {greedy_metrics['success_rate']:.3f} "
            f"| DQN: {metrics['success_rate']:.3f}"
        )

    # Aggregate results over all evaluation episodes.
    summary_rows = []

    for policy in POLICIES:
        metrics = calculate_metrics(
            all_results[policy]
        )

        summary_rows.append({
            "policy": policy.upper(),
            **metrics,
        })

    # Output paths.
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    tag = args.tag or (
        f"energy_{args.energy_coefficient:g}"
        f"_rate{args.arrival_rate:g}"
        f"_cpu{args.cpu_cycles:g}"
        f"_deadline{args.deadline:g}"
    )

    episode_path = (
        output_dir /
        f"dqn_evaluation_{tag}_episodes.csv"
    )

    summary_path = (
        output_dir /
        f"dqn_evaluation_{tag}_summary.csv"
    )

    plot_path = (
        output_dir /
        f"dqn_policy_comparison_{tag}.png"
    )

    fields = [
        "episode",
        "seed",
        "policy",
        "tasks",
        "mean_latency_s",
        "p95_latency_s",
        "mean_energy_j",
        "deadline_violation_rate",
        "success_rate",
        "mean_load_imbalance",
        "mean_objective",
        "infeasible_rate",
    ]

    save_csv(
        episode_rows,
        episode_path,
        fields,
    )

    summary_fields = [
        "policy",
        "tasks",
        "mean_latency_s",
        "p95_latency_s",
        "mean_energy_j",
        "deadline_violation_rate",
        "success_rate",
        "mean_load_imbalance",
        "mean_objective",
        "infeasible_rate",
    ]

    save_csv(
        summary_rows,
        summary_path,
        summary_fields,
    )

    save_comparison_plot(
        summary_rows,
        plot_path,
    )

    # Final policy comparison.
    print("\n" + "=" * 95)
    print("FINAL POLICY COMPARISON")
    print("=" * 95)

    print(
        f"{'Policy':<10}"
        f"{'Latency(s)':>13}"
        f"{'P95(s)':>12}"
        f"{'Energy(J)':>13}"
        f"{'Success':>11}"
        f"{'Objective':>13}"
        f"{'Infeasible':>13}"
    )

    for row in summary_rows:
        print(
            f"{row['policy']:<10}"
            f"{row['mean_latency_s']:>13.4f}"
            f"{row['p95_latency_s']:>12.4f}"
            f"{row['mean_energy_j']:>13.4f}"
            f"{row['success_rate']:>11.4f}"
            f"{row['mean_objective']:>13.4f}"
            f"{row['infeasible_rate']:>13.4f}"
        )

    # DQN action distribution.
    total_dqn_actions = sum(
        total_action_counts.values()
    )

    print("\n" + "=" * 70)
    print("DQN ACTION DISTRIBUTION")
    print("=" * 70)

    print(f"Total selected actions: {total_dqn_actions}")

    for pair, count in total_action_counts.most_common():
        percentage = (
            100.0 * count / total_dqn_actions
            if total_dqn_actions else 0.0
        )

        print(
            f"{pair:<25} "
            f"{count:>8} "
            f"({percentage:6.2f}%)"
        )

    print("\nEvaluation completed.")
    print(f"Episode results: {episode_path}")
    print(f"Summary results: {summary_path}")
    print(f"Comparison graph: {plot_path}")


if __name__ == "__main__":
    main()