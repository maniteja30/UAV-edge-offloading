"""Evaluate PPO against LOCAL, NEAREST and GREEDY on common test seeds."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter
from math import ceil, isfinite
from pathlib import Path

import numpy as np
import torch

from tasksim.simulator import Config, OffloadingSimulator
from tasksim.rl_env import OffloadingRLEnv
from rl.ppo_agent import PPOAgent

POLICIES = ("local", "nearest", "greedy", "ppo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PPO against common UAV task-offloading baselines"
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--task-count", type=int, default=None)
    parser.add_argument("--model", type=str, required=True)

    # Keep identical to the common experimental configuration.
    parser.add_argument("--n-users", type=int, default=10)
    parser.add_argument("--n-edges", type=int, default=3)
    parser.add_argument("--helpers-per-source", type=int, default=2)
    parser.add_argument("--episode-duration", type=float, default=60.0)
    parser.add_argument("--arrival-rate", type=float, default=0.8)
    parser.add_argument("--cpu-cycles", type=float, default=1e9)
    parser.add_argument("--deadline", type=float, default=2.0)
    parser.add_argument("--energy-coefficient", type=float, default=1e-28)
    parser.add_argument("--tag", type=str, default="ppo_experiment2_test")
    return parser.parse_args()


def build_config(args: argparse.Namespace, seed: int) -> Config:
    return Config(
        n_users=args.n_users,
        n_edges=args.n_edges,
        helpers_per_source=args.helpers_per_source,
        episode_duration_s=args.episode_duration,
        arrival_rate_per_user=args.arrival_rate,
        cpu_cycles=args.cpu_cycles,
        deadline_s=args.deadline,
        cpu_energy_coefficient=args.energy_coefficient,
        task_count=args.task_count,
        seed=seed,
    )


def finite_mean(values: list[float]) -> float:
    valid = [float(value) for value in values if isfinite(float(value))]
    return statistics.mean(valid) if valid else float("nan")


def calculate_metrics(results, nodes, config: Config) -> dict[str, float | int]:
    """Use the same task-level metric definitions as the DQN evaluator.

    Network lifetime is sourced from the simulator's existing lifetime
    calculation. If no UAV crosses its depletion threshold during an episode,
    lifetime is recorded as NaN rather than inventing a value beyond the run.
    """
    total = len(results)
    feasible = [
        result for result in results
        if isfinite(result.latency_s)
        and isfinite(result.energy_j)
        and isfinite(result.objective)
    ]
    latencies = sorted(result.latency_s for result in feasible)
    energies = [result.energy_j for result in feasible]
    imbalances = [result.load_imbalance for result in feasible]
    objectives = [result.objective for result in feasible]
    successes = sum(bool(result.deadline_met) for result in results)
    p95 = (
        latencies[max(0, ceil(0.95 * len(latencies)) - 1)]
        if latencies else float("nan")
    )
    lifetime = OffloadingSimulator.lifetime_summary(
        nodes,
        config.lifetime_death_threshold_fraction,
    )
    lifetime_s = lifetime.get("network_lifetime_s")

    return {
        "tasks": total,
        "mean_latency_s": finite_mean(latencies),
        "p95_latency_s": p95,
        "mean_energy_j": finite_mean(energies),
        "deadline_violation_rate": (
            1.0 - successes / total if total else 0.0
        ),
        "success_rate": successes / total if total else 0.0,
        "mean_load_imbalance": finite_mean(imbalances),
        "mean_objective": finite_mean(objectives),
        "infeasible_rate": (
            1.0 - len(feasible) / total if total else 0.0
        ),
        "network_lifetime_s": (
            float(lifetime_s) if lifetime_s is not None else float("nan")
        ),
        "episodes_with_node_death": int(lifetime_s is not None),
    }


def evaluate_ppo(
    config: Config,
    agent: PPOAgent,
    seed: int,
) -> tuple[list, Counter, dict, dict]:
    env = OffloadingRLEnv(config)
    state, _info = env.reset(seed=seed)
    action_counts: Counter = Counter()

    while not env.done:
        action_mask = np.asarray(env.get_action_mask(), dtype=np.bool_)
        action, _log_prob, _value = agent.select_action(
            state,
            action_mask,
            deterministic=True,
        )
        if action is None:
            action_counts["NO_FEASIBLE_ACTION"] += 1
        else:
            pair = env.get_action_pairs()[action]
            action_counts[f"{pair[0]} + {pair[1]}"] += 1

        next_state, _reward, terminated, truncated, _step_info = env.step(action)
        state = next_state
        if terminated or truncated:
            break

    summary = env.get_episode_summary()
    return env.results, action_counts, summary, env.nodes


def save_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_plot(rows: list[dict], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib unavailable; comparison graph was not generated.")
        return

    metrics = [
        ("mean_latency_s", "Mean latency (s)"),
        ("mean_energy_j", "Mean energy (J)"),
        ("success_rate", "Success rate"),
        ("mean_objective", "Mean objective"),
        ("mean_load_imbalance", "Mean load imbalance"),
        ("network_lifetime_s", "Network lifetime (s)"),
    ]
    labels = [row["policy"].upper() for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, (key, title) in zip(axes.flat, metrics):
        values = [
            float(row[key]) if isfinite(float(row[key])) else np.nan
            for row in rows
        ]
        axis.bar(labels, values)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.task_count is not None and args.task_count <= 0:
        raise ValueError("task-count must be positive when supplied")
    if min(args.n_users, args.n_edges) <= 0:
        raise ValueError("n-users and n-edges must be positive")
    if args.helpers_per_source < 0:
        raise ValueError("helpers-per-source cannot be negative")
    if args.episode_duration <= 0 or args.arrival_rate <= 0:
        raise ValueError("episode-duration and arrival-rate must be positive")
    if args.cpu_cycles <= 0 or args.deadline <= 0:
        raise ValueError("cpu-cycles and deadline must be positive")
    if args.energy_coefficient <= 0:
        raise ValueError("energy-coefficient must be positive")

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {model_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    initial_config = build_config(args, args.seed)
    initial_env = OffloadingRLEnv(initial_config)
    _state, initial_info = initial_env.reset(seed=args.seed)
    agent = PPOAgent(
        state_size=initial_info["state_size"],
        action_size=initial_info["action_size"],
        device=device,
        seed=args.seed,
    )
    checkpoint_metadata = agent.load(str(model_path))

    print("=" * 82)
    print("PPO UAV TASK-OFFLOADING EVALUATION")
    print("=" * 82)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {model_path}")
    print(f"Evaluation episodes: {args.episodes}")
    print(f"Evaluation seeds: {args.seed} through {args.seed + args.episodes - 1}")
    print(
        f"Common config: users={args.n_users}, edges={args.n_edges}, "
        f"helpers={args.helpers_per_source}, duration={args.episode_duration}s, "
        f"arrival={args.arrival_rate}, CPU={args.cpu_cycles:.2e}, "
        f"deadline={args.deadline}s, energy coefficient={args.energy_coefficient:.2e}"
    )
    if checkpoint_metadata:
        train_config = checkpoint_metadata.get("common_simulator_config", {})
        if train_config and train_config != {
            "n_users": args.n_users,
            "n_edges": args.n_edges,
            "helpers_per_source": args.helpers_per_source,
            "episode_duration_s": args.episode_duration,
            "arrival_rate_per_user": args.arrival_rate,
            "expected_tasks_per_episode": (
                args.n_users * args.arrival_rate * args.episode_duration
            ),
            "cpu_cycles": args.cpu_cycles,
            "deadline_s": args.deadline,
            "cpu_energy_coefficient": args.energy_coefficient,
            "task_count": args.task_count,
        }:
            print("WARNING: evaluation config differs from training checkpoint metadata.")
    print("=" * 82)

    all_results: dict[str, list] = {policy: [] for policy in POLICIES}
    lifetime_values: dict[str, list[float]] = {policy: [] for policy in POLICIES}
    episode_rows: list[dict] = []
    total_action_counts: Counter = Counter()

    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        config = build_config(args, episode_seed)
        simulator = OffloadingSimulator(config)

        # The baseline runner returns its final node resources so lifetime is
        # calculated from exactly the same committed energy events.
        for policy in ("local", "nearest", "greedy"):
            results, nodes = simulator.run_policy(policy, return_resources=True)
            metrics = calculate_metrics(results, nodes, config)
            all_results[policy].extend(results)
            if isfinite(float(metrics["network_lifetime_s"])):
                lifetime_values[policy].append(float(metrics["network_lifetime_s"]))
            episode_rows.append({
                "episode": episode + 1,
                "seed": episode_seed,
                "policy": policy.upper(),
                **metrics,
            })

        ppo_results, action_counts, _ppo_summary, ppo_nodes = evaluate_ppo(
            config,
            agent,
            episode_seed,
        )
        ppo_metrics = calculate_metrics(
            ppo_results,
            ppo_nodes,
            config,
        )
        all_results["ppo"].extend(ppo_results)
        if isfinite(float(ppo_metrics["network_lifetime_s"])):
            lifetime_values["ppo"].append(float(ppo_metrics["network_lifetime_s"]))
        total_action_counts.update(action_counts)
        episode_rows.append({
            "episode": episode + 1,
            "seed": episode_seed,
            "policy": "PPO",
            **ppo_metrics,
        })

        print(
            f"Episode {episode + 1:2d}/{args.episodes} | "
            f"Tasks: {len(ppo_results):4d} | "
            f"LOCAL: {episode_rows[-4]['success_rate']:.3f} | "
            f"GREEDY: {episode_rows[-2]['success_rate']:.3f} | "
            f"PPO: {ppo_metrics['success_rate']:.3f}"
        )

    summary_rows: list[dict] = []
    for policy in POLICIES:
        # Recreate a representative episode's final node state is not needed
        # for task metrics; lifetime is averaged across episode-level values.
        pooled_results = all_results[policy]
        finite_results = [
            result for result in pooled_results
            if isfinite(result.latency_s)
            and isfinite(result.energy_j)
            and isfinite(result.objective)
        ]
        total_tasks = len(pooled_results)
        successes = sum(bool(result.deadline_met) for result in pooled_results)
        sorted_latencies = sorted(result.latency_s for result in finite_results)
        p95 = (
            sorted_latencies[max(0, ceil(0.95 * len(sorted_latencies)) - 1)]
            if sorted_latencies else float("nan")
        )
        summary_rows.append({
            "policy": policy.upper(),
            "tasks": total_tasks,
            "mean_latency_s": finite_mean([r.latency_s for r in finite_results]),
            "p95_latency_s": p95,
            "mean_energy_j": finite_mean([r.energy_j for r in finite_results]),
            "deadline_violation_rate": (
                1.0 - successes / total_tasks if total_tasks else 0.0
            ),
            "success_rate": successes / total_tasks if total_tasks else 0.0,
            "mean_load_imbalance": finite_mean([r.load_imbalance for r in finite_results]),
            "mean_objective": finite_mean([r.objective for r in finite_results]),
            "infeasible_rate": (
                1.0 - len(finite_results) / total_tasks if total_tasks else 0.0
            ),
            "network_lifetime_s": finite_mean(lifetime_values[policy]),
            "episodes_with_node_death": len(lifetime_values[policy]),
        })

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / f"ppo_evaluation_{args.tag}_episodes.csv"
    summary_path = output_dir / f"ppo_evaluation_{args.tag}_summary.csv"
    plot_path = output_dir / f"ppo_policy_comparison_{args.tag}.png"
    action_path = output_dir / f"ppo_evaluation_{args.tag}_action_counts.csv"
    metadata_path = output_dir / f"ppo_evaluation_{args.tag}_metadata.json"

    metric_fields = [
        "tasks", "mean_latency_s", "p95_latency_s", "mean_energy_j",
        "deadline_violation_rate", "success_rate", "mean_load_imbalance",
        "mean_objective", "infeasible_rate", "network_lifetime_s",
        "episodes_with_node_death",
    ]
    save_csv(
        episode_rows,
        episode_path,
        ["episode", "seed", "policy"] + metric_fields,
    )
    save_csv(summary_rows, summary_path, ["policy"] + metric_fields)
    save_comparison_plot(summary_rows, plot_path)
    save_csv(
        [
            {"destination_pair": pair, "selection_count": count}
            for pair, count in total_action_counts.most_common()
        ],
        action_path,
        ["destination_pair", "selection_count"],
    )
    evaluation_metadata = {
        "algorithm": "PPO",
        "checkpoint": str(model_path),
        "evaluation_episodes": args.episodes,
        "test_seed_start": args.seed,
        "test_seeds": [args.seed + i for i in range(args.episodes)],
        "deterministic_policy": True,
        "common_simulator_config": {
            "n_users": args.n_users,
            "n_edges": args.n_edges,
            "helpers_per_source": args.helpers_per_source,
            "episode_duration_s": args.episode_duration,
            "arrival_rate_per_user": args.arrival_rate,
            "expected_tasks_per_episode": (
                args.n_users * args.arrival_rate * args.episode_duration
            ),
            "cpu_cycles": args.cpu_cycles,
            "deadline_s": args.deadline,
            "cpu_energy_coefficient": args.energy_coefficient,
            "task_count": args.task_count,
        },
        "baseline_policies": ["LOCAL", "NEAREST", "GREEDY"],
        "objective": (
            "0.5*(L/D) + 0.3*(E/0.1) + 0.2*(I/0.25) "
            "+ max(0, (L-D)/D)"
        ),
        "network_lifetime_note": (
            "Mean is over test episodes where a UAV crossed the simulator's "
            "depletion threshold; episodes without a node-death event are NaN. "
            "episodes_with_node_death records the number of finite episodes."
        ),
    }
    metadata_path.write_text(
        json.dumps(evaluation_metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 118)
    print("FINAL POLICY COMPARISON")
    print("=" * 118)
    print(
        f"{'Policy':<10}{'Latency(s)':>13}{'Energy(J)':>14}"
        f"{'Success':>11}{'Objective':>13}{'Lifetime(s)':>14}{'Infeasible':>13}"
    )
    for row in summary_rows:
        print(
            f"{row['policy']:<10}"
            f"{row['mean_latency_s']:>13.4f}"
            f"{row['mean_energy_j']:>14.4f}"
            f"{row['success_rate']:>11.4f}"
            f"{row['mean_objective']:>13.4f}"
            f"{row['network_lifetime_s']:>14.4f}"
            f"{row['infeasible_rate']:>13.4f}"
        )

    print("\nOutput files:")
    print(f"  Episode-level metrics: {episode_path}")
    print(f"  Summary metrics:       {summary_path}")
    print(f"  Comparison graph:      {plot_path}")
    print(f"  PPO action counts:     {action_path}")
    print(f"  Evaluation metadata:   {metadata_path}")


if __name__ == "__main__":
    main()
