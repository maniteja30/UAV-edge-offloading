"""Train masked PPO using the shared ``src/tasksim`` environment.

Run from the repository root:
    PYTHONPATH=src python src/train_ppo.py --episodes 100 ...

The simulator, observation/action spaces, action masks and environment reward
are used unchanged. PPO scales rewards internally for critic stability only.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from tasksim.simulator import Config
from tasksim.rl_env import OffloadingRLEnv
from rl.ppo_agent import PPOAgent, PPOHyperparameters, PPORolloutBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO for UAV task offloading using src/tasksim/"
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--task-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)

    # Shared DQN/DDQN/PPO experiment settings.
    parser.add_argument("--n-users", type=int, default=10)
    parser.add_argument("--n-edges", type=int, default=3)
    parser.add_argument("--helpers-per-source", type=int, default=2)
    parser.add_argument("--episode-duration", type=float, default=60.0)
    parser.add_argument("--arrival-rate", type=float, default=0.8)
    parser.add_argument("--cpu-cycles", type=float, default=1e9)
    parser.add_argument("--deadline", type=float, default=2.0)
    parser.add_argument("--energy-coefficient", type=float, default=1e-28)

    # PPO-specific settings. reward-scale affects PPO internals only.
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.20)
    parser.add_argument("--value-clip-coef", type=float, default=10.0)
    parser.add_argument("--value-coef", type=float, default=0.50)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.50)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--reward-scale", type=float, default=1000.0)
    parser.add_argument("--target-kl", type=float, default=0.03)

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save the latest checkpoint every N completed episodes (default: 1)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/ppo_model.pt",
    )
    parser.add_argument("--tag", type=str, default="ppo_experiment2")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
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
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")


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


def save_training_plot(rows: list[dict], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib unavailable; training plot was not generated.")
        return

    episodes = [row["episode"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    rewards = [row["mean_reward_per_task"] for row in rows]
    axes[0, 0].plot(episodes, rewards, alpha=0.35, label="Per-task reward")
    if len(rewards) >= 10:
        moving = np.convolve(rewards, np.ones(10) / 10.0, mode="valid")
        axes[0, 0].plot(episodes[9:], moving, label="10-episode moving average")
    axes[0, 0].set_title("PPO Mean Reward per Task (raw environment reward)")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Mean reward/task")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(episodes, [row["success_rate"] for row in rows])
    axes[0, 1].set_title("PPO Deadline Success Rate")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Success rate")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(episodes, [row["value_loss"] for row in rows], label="Scaled critic loss")
    axes[1, 0].plot(episodes, [row["explained_variance"] for row in rows], label="Explained variance")
    axes[1, 0].set_title("Critic Diagnostics")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(episodes, [row["entropy"] for row in rows], label="Entropy")
    axes[1, 1].plot(episodes, [row["approx_kl"] for row in rows], label="Approx. KL")
    axes[1, 1].plot(episodes, [row["clip_fraction"] for row in rows], label="Clip fraction")
    axes[1, 1].set_title("Policy Update Diagnostics")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_checkpoint_atomically(
    agent: PPOAgent,
    checkpoint_path: Path,
    metadata: dict,
) -> None:
    """Write to a temporary file before replacing the latest checkpoint."""
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    agent.save(str(temporary_path), metadata=metadata)
    temporary_path.replace(checkpoint_path)


def main() -> None:
    args = parse_args()
    validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.model_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    hp = PPOHyperparameters(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        value_clip_coef=args.value_clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
        reward_scale=args.reward_scale,
        target_kl=args.target_kl,
    )

    initial_env = OffloadingRLEnv(build_config(args, args.seed))
    _initial_state, initial_info = initial_env.reset(seed=args.seed)
    agent = PPOAgent(
        state_size=initial_info["state_size"],
        action_size=initial_info["action_size"],
        device=device,
        hyperparameters=hp,
        seed=args.seed,
    )

    csv_path = output_dir / f"ppo_training_{args.tag}.csv"
    metadata_path = output_dir / f"ppo_training_{args.tag}_metadata.json"
    plot_path = output_dir / f"ppo_training_{args.tag}.png"

    print("=" * 88)
    print("PPO UAV TASK-OFFLOADING TRAINING (IMPROVED)")
    print("=" * 88)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Episodes: {args.episodes}")
    print(
        "Common config: "
        f"users={args.n_users}, edges={args.n_edges}, "
        f"helpers={args.helpers_per_source}, duration={args.episode_duration}s, "
        f"arrival={args.arrival_rate} tasks/s/user, CPU={args.cpu_cycles:.2e}, "
        f"deadline={args.deadline}s, energy coefficient={args.energy_coefficient:.2e}"
    )
    print(
        f"Expected tasks per episode (Poisson mean): "
        f"{args.n_users * args.arrival_rate * args.episode_duration:.0f}"
    )
    print(f"Observation size: {agent.state_size} | Action size: {agent.action_size}")
    print(f"PPO hyperparameters: {asdict(hp)}")
    print(f"Checkpoint: {checkpoint_path} (saved every {args.checkpoint_every} episode(s))")
    print("Raw simulator rewards are unchanged; reward_scale is applied inside PPO only.")
    print("=" * 88)

    training_start = time.perf_counter()
    episode_rows: list[dict] = []
    fields: list[str] | None = None

    base_metadata = {
        "algorithm": "PPO",
        "requested_episodes": args.episodes,
        "training_seed": args.seed,
        "training_seed_sequence": [args.seed + i for i in range(args.episodes)],
        "device": str(device),
        "observation_size": agent.state_size,
        "action_size": agent.action_size,
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
        "ppo_hyperparameters": asdict(hp),
        "objective": (
            "0.5*(L/D) + 0.3*(E/0.1) + 0.2*(I/0.25) "
            "+ max(0, (L-D)/D)"
        ),
        "rollout_strategy": "one complete simulator episode per PPO update",
        "mask_handling": (
            "invalid actions removed from categorical policy; all-false masks "
            "use env.step(None) and critic-only training"
        ),
        "reward_scaling_note": (
            "raw environment reward is preserved; PPO divides rollout rewards "
            "by reward_scale only for GAE and critic regression"
        ),
        "checkpoint_interval_episodes": args.checkpoint_every,
    }

    # Stream the CSV as training progresses so completed episodes are not lost
    # if a long experiment is interrupted.
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = None

        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            env = OffloadingRLEnv(build_config(args, episode_seed))
            state, _info = env.reset(seed=episode_seed)
            rollout = PPORolloutBuffer()

            total_reward = 0.0
            no_feasible_action_count = 0
            task_count = 0

            while not env.done:
                action_mask = np.asarray(env.get_action_mask(), dtype=np.bool_)
                action, log_prob, value = agent.select_action(
                    state,
                    action_mask,
                    deterministic=False,
                )
                policy_valid = action is not None
                if action is None:
                    no_feasible_action_count += 1
                    stored_action = 0  # critic-only placeholder; env receives None
                else:
                    stored_action = action

                next_state, reward, terminated, truncated, _step_info = env.step(action)
                done = bool(terminated or truncated)
                rollout.add(
                    state=state,
                    action=stored_action,
                    reward=reward,
                    done=done,
                    log_prob=log_prob,
                    value=value,
                    action_mask=action_mask,
                    policy_valid=policy_valid,
                )
                total_reward += float(reward)
                task_count += 1
                state = next_state

            update_metrics = agent.update(rollout, next_value=0.0)
            episode_summary = env.get_episode_summary()
            elapsed = time.perf_counter() - training_start

            row = {
                "episode": episode + 1,
                "seed": episode_seed,
                "tasks": task_count,
                "total_reward": total_reward,
                "mean_reward_per_task": total_reward / max(task_count, 1),
                "success_rate": float(episode_summary.get("success_rate", 0.0)),
                "network_lifetime_s": (
                    episode_summary.get("network_lifetime_s")
                    if episode_summary.get("network_lifetime_s") is not None
                    else float("nan")
                ),
                "no_feasible_action_count": no_feasible_action_count,
                "policy_loss": update_metrics["policy_loss"],
                "value_loss": update_metrics["value_loss"],
                "loss": update_metrics["loss"],
                "entropy": update_metrics["entropy"],
                "approx_kl": update_metrics["approx_kl"],
                "clip_fraction": update_metrics["clip_fraction"],
                "explained_variance": update_metrics["explained_variance"],
                "return_target_mean_scaled": update_metrics["return_target_mean"],
                "return_target_std_scaled": update_metrics["return_target_std"],
                "value_prediction_mean_scaled": update_metrics["value_prediction_mean"],
                "value_prediction_std_scaled": update_metrics["value_prediction_std"],
                "optimizer_minibatch_updates": int(update_metrics["updates"]),
                "target_kl_early_stop": bool(update_metrics["early_stopped"]),
                "elapsed_seconds": elapsed,
            }
            episode_rows.append(row)

            if writer is None:
                fields = list(row.keys())
                writer = csv.DictWriter(csv_file, fieldnames=fields)
                writer.writeheader()
            writer.writerow(row)
            csv_file.flush()

            completed = episode + 1
            if completed % args.checkpoint_every == 0 or completed == args.episodes:
                current_metadata = {
                    **base_metadata,
                    "episodes_completed": completed,
                    "training_complete": completed == args.episodes,
                    "training_time_seconds_at_checkpoint": time.perf_counter() - training_start,
                }
                save_checkpoint_atomically(agent, checkpoint_path, current_metadata)
                write_metadata(metadata_path, current_metadata)

            print(
                f"Episode {completed:3d}/{args.episodes} "
                f"| Tasks: {task_count:4d} "
                f"| Reward/task: {row['mean_reward_per_task']:9.3f} "
                f"| Success: {row['success_rate']:.3f} "
                f"| Policy: {row['policy_loss']:+.4f} "
                f"| Value: {row['value_loss']:.4f} "
                f"| EV: {row['explained_variance']:+.3f} "
                f"| Entropy: {row['entropy']:.3f} "
                f"| KL: {row['approx_kl']:.4f} "
                f"| Clip: {row['clip_fraction']:.3f} "
                f"| No-feasible: {no_feasible_action_count}"
            )

    total_training_seconds = time.perf_counter() - training_start
    final_metadata = {
        **base_metadata,
        "episodes_completed": len(episode_rows),
        "training_complete": len(episode_rows) == args.episodes,
        "training_time_seconds": total_training_seconds,
    }
    write_metadata(metadata_path, final_metadata)
    save_training_plot(episode_rows, plot_path)

    print("\nTraining finished.")
    print(f"Training time: {total_training_seconds:.2f} seconds")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training curve CSV: {csv_path}")
    print(f"Training metadata: {metadata_path}")
    print(f"Training graph: {plot_path}")


if __name__ == "__main__":
    main()
