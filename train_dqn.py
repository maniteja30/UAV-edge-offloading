import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch

from tasksim.simulator import Config
from tasksim.rl_env import OffloadingRLEnv
from rl.dqn_agent import DQNAgent
from rl.replay_buffer import ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a DQN agent for UAV task offloading"
    )

    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--task-count", type=int, default=None)

    parser.add_argument("--arrival-rate", type=float, default=1.0)
    parser.add_argument("--cpu-cycles", type=float, default=1e9)
    parser.add_argument("--deadline", type=float, default=2.0)

    parser.add_argument(
        "--energy-coefficient",
        type=float,
        default=1e-28,
        help="CPU energy coefficient used by the simulator",
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-capacity", type=int, default=50000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.98)

    parser.add_argument("--learning-starts", type=int, default=500)
    parser.add_argument("--target-update", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/dqn_model_v2.pt",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag used to distinguish output files",
    )

    return parser.parse_args()


def validate_args(args):
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    if args.task_count is not None and args.task_count <= 0:
        raise ValueError("task-count must be positive")

    if args.arrival_rate <= 0:
        raise ValueError("arrival-rate must be positive")

    if args.cpu_cycles <= 0 or args.deadline <= 0:
        raise ValueError("cpu-cycles and deadline must be positive")

    if args.energy_coefficient <= 0:
        raise ValueError("energy-coefficient must be positive")

    if args.batch_size <= 0 or args.buffer_capacity <= 0:
        raise ValueError("batch-size and buffer-capacity must be positive")

    if args.learning_starts < 0 or args.target_update <= 0:
        raise ValueError("Invalid learning-starts or target-update")

    if args.batch_size > args.buffer_capacity:
        raise ValueError("batch-size cannot exceed buffer-capacity")

    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")

    if not 0 <= args.gamma <= 1:
        raise ValueError("gamma must be between 0 and 1")

    if not 0 <= args.epsilon_min <= args.epsilon_start <= 1:
        raise ValueError("Invalid epsilon values")

    if not 0 < args.epsilon_decay <= 1:
        raise ValueError("epsilon-decay must be in (0, 1]")


def save_training_plot(rewards, losses, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is unavailable; skipping training plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(rewards, alpha=0.4, label="Episode reward")

    if len(rewards) >= 10:
        moving_average = np.convolve(
            rewards,
            np.ones(10) / 10,
            mode="valid",
        )

        axes[0].plot(
            range(9, len(rewards)),
            moving_average,
            label="10-episode average",
        )

    axes[0].set_title("DQN Training Rewards")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total reward")
    axes[0].legend()
    axes[0].grid(True)

    if losses:
        axes[1].plot(losses)

    axes[1].set_title("DQN Training Loss")
    axes[1].set_xlabel("Training update")
    axes[1].set_ylabel("Smooth L1 loss")
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path("results")
    checkpoint_path = Path(args.model_path)

    output_dir.mkdir(exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    tag = args.tag or f"energy_{args.energy_coefficient:g}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("DQN UAV TASK-OFFLOADING TRAINING")
    print("=" * 65)
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Episodes: {args.episodes}")
    print(f"Tasks per episode: {args.task_count or 'Poisson arrivals'}")
    print(f"Arrival rate per user: {args.arrival_rate}")
    print(f"CPU cycles per task: {args.cpu_cycles}")
    print(f"Deadline: {args.deadline} seconds")
    print(f"CPU energy coefficient: {args.energy_coefficient:.2e}")
    print(f"Epsilon decay: {args.epsilon_decay}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output tag: {tag}")
    print("=" * 65)

    replay_buffer = ReplayBuffer(
        capacity=args.buffer_capacity,
        seed=args.seed,
    )

    agent = None
    epsilon = args.epsilon_start
    gradient_steps = 0

    episode_rewards = []
    episode_losses = []
    all_losses = []
    episode_success_rates = []
    episode_epsilons = []

    action_counts = {}
    no_feasible_action_count = 0

    for episode in range(args.episodes):
        episode_seed = args.seed + episode

        config = Config(
            task_count=args.task_count,
            arrival_rate_per_user=args.arrival_rate,
            cpu_cycles=args.cpu_cycles,
            deadline_s=args.deadline,
            cpu_energy_coefficient=args.energy_coefficient,
            seed=episode_seed,
        )

        env = OffloadingRLEnv(config)
        state, info = env.reset(seed=episode_seed)

        if agent is None:
            agent = DQNAgent(
                state_size=info["state_size"],
                action_size=info["action_size"],
                learning_rate=args.learning_rate,
                gamma=args.gamma,
                device=device,
                seed=args.seed,
            )

        current_epsilon = epsilon
        total_reward = 0.0
        losses = []
        episode_done = False

        while not episode_done:
            action_mask = np.asarray(
                env.get_action_mask(),
                dtype=bool,
            )

            action = agent.select_action(
                state,
                action_mask,
                epsilon=current_epsilon,
            )

            next_state, reward, terminated, truncated, step_info = (
                env.step(action)
            )

            episode_done = terminated or truncated

            if action is not None:
                action_counts[action] = action_counts.get(action, 0) + 1
            else:
                no_feasible_action_count += 1

            if episode_done:
                next_action_mask = np.zeros(
                    info["action_size"],
                    dtype=bool,
                )
            else:
                next_action_mask = np.asarray(
                    env.get_action_mask(),
                    dtype=bool,
                )

            # Store transitions only when a valid action was selected.
            if action is not None:
                replay_buffer.push(
                    state,
                    action,
                    reward,
                    next_state,
                    episode_done,
                    next_action_mask,
                )

                if len(replay_buffer) >= max(
                    args.batch_size,
                    args.learning_starts,
                ):
                    loss = agent.update(
                        replay_buffer,
                        batch_size=args.batch_size,
                    )

                    if loss is not None:
                        losses.append(loss)
                        all_losses.append(loss)
                        gradient_steps += 1

                        if gradient_steps % args.target_update == 0:
                            agent.update_target_network()

            total_reward += reward
            state = next_state

        summary = env.get_episode_summary()
        success_rate = summary.get("success_rate", 0.0)

        episode_rewards.append(total_reward)
        episode_losses.append(
            float(np.mean(losses)) if losses else 0.0
        )
        episode_success_rates.append(success_rate)
        episode_epsilons.append(current_epsilon)

        print(
            f"Episode {episode + 1:4d}/{args.episodes} | "
            f"Reward: {total_reward:10.3f} | "
            f"Loss: {episode_losses[-1]:.5f} | "
            f"Epsilon: {current_epsilon:.3f} | "
            f"Success: {success_rate:.3f} | "
            f"Tasks: {summary.get('tasks', 0)}"
        )

        epsilon = max(
            args.epsilon_min,
            epsilon * args.epsilon_decay,
        )

    # Save the trained model.
    agent.save(str(checkpoint_path))

    # Save episode-level training results.
    csv_path = output_dir / f"dqn_training_{tag}.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "episode",
            "total_reward",
            "average_loss",
            "epsilon",
            "success_rate",
        ])

        for i in range(args.episodes):
            writer.writerow([
                i + 1,
                episode_rewards[i],
                episode_losses[i],
                episode_epsilons[i],
                episode_success_rates[i],
            ])

    # Save action-selection counts.
    action_csv_path = (
        output_dir / f"dqn_training_action_counts_{tag}.csv"
    )

    with action_csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["action_index", "selection_count"])

        for action in sorted(action_counts):
            writer.writerow([
                action,
                action_counts[action],
            ])

        writer.writerow([
            "NO_FEASIBLE_ACTION",
            no_feasible_action_count,
        ])

    # Save training graph.
    plot_path = output_dir / f"dqn_training_{tag}.png"

    save_training_plot(
        episode_rewards,
        all_losses,
        plot_path,
    )

    print("\nTraining completed.")
    print(f"Model saved: {checkpoint_path}")
    print(f"Results saved: {csv_path}")
    print(f"Action counts saved: {action_csv_path}")
    print(f"Training graph: {plot_path}")
    print(f"Total gradient updates: {gradient_steps}")
    print(
        "Transitions skipped (no feasible action): "
        f"{no_feasible_action_count}"
    )

    print("\nAction selection counts:")
    for action in sorted(action_counts):
        print(f"Action {action:2d}: {action_counts[action]}")

    print(f"Total selected actions: {sum(action_counts.values())}")


if __name__ == "__main__":
    main()