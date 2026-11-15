"""Masked Proximal Policy Optimization for the UAV task-offloading environment.

This module is intentionally independent of ``src/tasksim``. It consumes the
existing observation vectors, discrete actions, action masks and rewards
without changing the shared simulator or RL environment.

Stability improvements over the initial implementation:
- scale rewards internally for critic/GAE calculations (raw env rewards remain
  unchanged in the simulator and in reported episode metrics);
- clipped value-function loss;
- optional target-KL early stopping;
- orthogonal network initialization with a small actor-head gain;
- explained-variance and value-target diagnostics;
- all-false masks remain critic-only transitions and are never treated as
  policy decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class PPOHyperparameters:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.20
    value_clip_coef: float = 10.0
    value_coef: float = 0.50
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.50
    update_epochs: int = 4
    batch_size: int = 256
    # This is an algorithm-internal scale only. The environment's reward
    # function and the raw rewards recorded in training logs are unchanged.
    reward_scale: float = 1000.0
    # Stop the current PPO update early if the policy changes too far from the
    # rollout policy. Set to 0 to disable this guard.
    target_kl: float = 0.03

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if not 0.0 < self.clip_coef < 1.0:
            raise ValueError("clip_coef must be in (0, 1)")
        if self.value_clip_coef < 0:
            raise ValueError("value_clip_coef cannot be negative")
        if self.value_coef < 0 or self.entropy_coef < 0:
            raise ValueError("value_coef and entropy_coef cannot be negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.update_epochs <= 0 or self.batch_size <= 0:
            raise ValueError("update_epochs and batch_size must be positive")
        if self.reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        if self.target_kl < 0:
            raise ValueError("target_kl cannot be negative")


class ActorCritic(nn.Module):
    """Shared MLP encoder with categorical actor and scalar critic heads."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: tuple[int, ...] = (256, 128),
    ):
        super().__init__()
        if state_size <= 0 or action_size <= 0:
            raise ValueError("state_size and action_size must be positive")

        layers: list[nn.Module] = []
        input_size = state_size
        for width in hidden_sizes:
            layers.extend([nn.Linear(input_size, width), nn.Tanh()])
            input_size = width

        self.shared = nn.Sequential(*layers)
        self.actor = nn.Linear(input_size, action_size)
        self.critic = nn.Linear(input_size, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.constant_(module.bias, 0.0)
        # Start with a near-uniform policy over the currently valid actions.
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(states)
        return self.actor(features), self.critic(features).squeeze(-1)


class PPORolloutBuffer:
    """Stores one on-policy rollout, including its action masks.

    ``policy_valid`` is false for forced no-feasible-action transitions. Those
    transitions train the critic but are omitted from the actor and entropy
    losses because the policy had no legal decision to make.
    """

    def __init__(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.action_masks: list[np.ndarray] = []
        self.policy_valid: list[bool] = []

    def add(
        self,
        state: np.ndarray | list[float],
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        action_mask: np.ndarray | list[bool],
        policy_valid: bool = True,
    ) -> None:
        self.states.append(np.asarray(state, dtype=np.float32).copy())
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.action_masks.append(np.asarray(action_mask, dtype=np.bool_).copy())
        self.policy_valid.append(bool(policy_valid))

    def __len__(self) -> int:
        return len(self.states)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.action_masks.clear()
        self.policy_valid.clear()


class PPOAgent:
    """PPO agent with categorical actions and strict action-mask handling."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        device: str | torch.device = "cpu",
        hyperparameters: PPOHyperparameters | None = None,
        seed: int | None = None,
    ):
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        if self.state_size <= 0 or self.action_size <= 0:
            raise ValueError("state_size and action_size must be positive")

        self.device = torch.device(device)
        self.hyperparameters = hyperparameters or PPOHyperparameters()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.network = ActorCritic(self.state_size, self.action_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hyperparameters.learning_rate,
            eps=1e-5,
        )

    def _state_tensor(self, state: np.ndarray | list[float]) -> torch.Tensor:
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (self.state_size,):
            raise ValueError(
                f"Expected state shape {(self.state_size,)}, received {state_array.shape}"
            )
        if not np.isfinite(state_array).all():
            raise ValueError("Observation contains NaN or infinite values")
        return torch.as_tensor(
            state_array, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

    def _masked_distribution(
        self,
        logits: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor]:
        """Build a categorical distribution after excluding invalid actions.

        An all-false row is critic-only. One dummy action is enabled only to
        keep the distribution finite; that row is excluded from actor losses.
        """
        masks = action_masks.to(dtype=torch.bool).clone()
        no_valid_action = ~masks.any(dim=-1)
        if no_valid_action.any():
            masks[no_valid_action, 0] = True
        masked_logits = logits.masked_fill(~masks, -1e9)
        return Categorical(logits=masked_logits), masks

    @torch.no_grad()
    def value(self, state: np.ndarray | list[float]) -> float:
        _logits, value = self.network(self._state_tensor(state))
        return float(value.item())

    @torch.no_grad()
    def select_action(
        self,
        state: np.ndarray | list[float],
        action_mask: np.ndarray | list[bool],
        deterministic: bool = False,
    ) -> tuple[int | None, float, float]:
        """Return ``(action, log_probability, value)``.

        Returns ``(None, 0.0, value)`` when no action is feasible, matching the
        existing ``OffloadingRLEnv.step(None)`` contract.
        """
        mask = np.asarray(action_mask, dtype=np.bool_)
        if mask.shape != (self.action_size,):
            raise ValueError(
                f"Expected action mask shape {(self.action_size,)}, received {mask.shape}"
            )

        logits, value = self.network(self._state_tensor(state))
        value_float = float(value.item())
        if not mask.any():
            return None, 0.0, value_float

        mask_tensor = torch.as_tensor(
            mask, dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        distribution, _ = self._masked_distribution(logits, mask_tensor)
        action_tensor = (
            torch.argmax(distribution.logits, dim=-1)
            if deterministic else distribution.sample()
        )
        return (
            int(action_tensor.item()),
            float(distribution.log_prob(action_tensor).item()),
            value_float,
        )

    def update(
        self,
        rollout: PPORolloutBuffer,
        next_value: float = 0.0,
    ) -> dict[str, float]:
        """Update actor and critic using GAE and clipped PPO objectives.

        Rewards are divided by ``reward_scale`` only inside PPO. This reduces
        critic target magnitude for the simulator's long, high-load episodes.
        The environment reward, evaluation objective and reported raw episode
        reward are not modified.
        """
        count = len(rollout)
        empty_metrics = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "explained_variance": 0.0,
            "return_target_mean": 0.0,
            "return_target_std": 0.0,
            "value_prediction_mean": 0.0,
            "value_prediction_std": 0.0,
            "updates": 0.0,
            "early_stopped": 0.0,
        }
        if count == 0:
            return empty_metrics

        hp = self.hyperparameters
        states_np = np.stack(rollout.states).astype(np.float32, copy=False)
        masks_np = np.stack(rollout.action_masks).astype(np.bool_, copy=False)
        if states_np.shape != (count, self.state_size):
            raise ValueError("Rollout contains observations with inconsistent dimensions")
        if masks_np.shape != (count, self.action_size):
            raise ValueError("Rollout contains action masks with inconsistent dimensions")

        states = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(rollout.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(
            rollout.log_probs, dtype=torch.float32, device=self.device
        )
        old_values_np = np.asarray(rollout.values, dtype=np.float32)
        old_values = torch.as_tensor(
            old_values_np, dtype=torch.float32, device=self.device
        )
        action_masks = torch.as_tensor(masks_np, dtype=torch.bool, device=self.device)
        policy_valid = torch.as_tensor(
            rollout.policy_valid, dtype=torch.bool, device=self.device
        )
        rewards = np.asarray(rollout.rewards, dtype=np.float32) / hp.reward_scale
        dones = np.asarray(rollout.dones, dtype=np.float32)

        # Generalized Advantage Estimation. A done transition resets the
        # bootstrap, so returns never leak from one episode into another.
        advantages_np = np.zeros(count, dtype=np.float32)
        gae = 0.0
        for index in reversed(range(count)):
            following_value = (
                float(next_value) if index == count - 1
                else float(old_values_np[index + 1])
            )
            non_terminal = 1.0 - dones[index]
            delta = (
                rewards[index]
                + hp.gamma * following_value * non_terminal
                - old_values_np[index]
            )
            gae = (
                delta
                + hp.gamma * hp.gae_lambda * non_terminal * gae
            )
            advantages_np[index] = gae

        # Critic targets remain in scaled-return units. Actor advantages are
        # normalized separately, a common PPO variance-reduction step.
        returns = torch.as_tensor(
            advantages_np + old_values_np,
            dtype=torch.float32,
            device=self.device,
        )
        advantages = torch.as_tensor(
            advantages_np, dtype=torch.float32, device=self.device
        )
        if policy_valid.any():
            valid_advantages = advantages[policy_valid]
            normalized = (
                valid_advantages - valid_advantages.mean()
            ) / (valid_advantages.std(unbiased=False) + 1e-8)
            advantages = advantages.clone()
            advantages[policy_valid] = normalized

        metric_totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        update_count = 0
        early_stopped = False

        for _epoch in range(hp.update_epochs):
            permutation = torch.randperm(count, device=self.device)
            for start in range(0, count, hp.batch_size):
                indices = permutation[start : start + hp.batch_size]
                logits, predicted_values = self.network(states[indices])
                distribution, _ = self._masked_distribution(
                    logits, action_masks[indices]
                )
                new_log_probs = distribution.log_prob(actions[indices])
                entropy_values = distribution.entropy()

                valid_rows = policy_valid[indices]
                if valid_rows.any():
                    log_ratio = (
                        new_log_probs[valid_rows]
                        - old_log_probs[indices][valid_rows]
                    )
                    ratio = log_ratio.exp()
                    batch_advantages = advantages[indices][valid_rows]
                    unclipped = ratio * batch_advantages
                    clipped = torch.clamp(
                        ratio,
                        1.0 - hp.clip_coef,
                        1.0 + hp.clip_coef,
                    ) * batch_advantages
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    entropy = entropy_values[valid_rows].mean()
                    approx_kl = (ratio - 1.0 - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > hp.clip_coef)
                        .float()
                        .mean()
                    )
                else:
                    policy_loss = torch.zeros((), device=self.device)
                    entropy = torch.zeros((), device=self.device)
                    approx_kl = torch.zeros((), device=self.device)
                    clip_fraction = torch.zeros((), device=self.device)

                if hp.target_kl > 0 and float(approx_kl.detach().item()) > hp.target_kl:
                    early_stopped = True
                    break

                if hp.value_clip_coef > 0:
                    clipped_values = old_values[indices] + torch.clamp(
                        predicted_values - old_values[indices],
                        -hp.value_clip_coef,
                        hp.value_clip_coef,
                    )
                    value_loss_unclipped = (
                        predicted_values - returns[indices]
                    ).pow(2)
                    value_loss_clipped = (
                        clipped_values - returns[indices]
                    ).pow(2)
                    value_loss = 0.5 * torch.maximum(
                        value_loss_unclipped,
                        value_loss_clipped,
                    ).mean()
                else:
                    value_loss = 0.5 * (
                        predicted_values - returns[indices]
                    ).pow(2).mean()

                loss = (
                    policy_loss
                    + hp.value_coef * value_loss
                    - hp.entropy_coef * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), hp.max_grad_norm
                )
                self.optimizer.step()

                metric_totals["loss"] += float(loss.detach().item())
                metric_totals["policy_loss"] += float(policy_loss.detach().item())
                metric_totals["value_loss"] += float(value_loss.detach().item())
                metric_totals["entropy"] += float(entropy.detach().item())
                metric_totals["approx_kl"] += float(approx_kl.detach().item())
                metric_totals["clip_fraction"] += float(clip_fraction.detach().item())
                update_count += 1

            if early_stopped:
                break

        for key in metric_totals:
            metric_totals[key] /= max(update_count, 1)

        # Critic diagnostics are evaluated after the final optimizer update.
        with torch.no_grad():
            _final_logits, final_values = self.network(states)
            residual = returns - final_values
            target_variance = torch.var(returns, unbiased=False)
            if float(target_variance.item()) > 1e-12:
                explained_variance = 1.0 - (
                    torch.var(residual, unbiased=False) / target_variance
                )
                explained_variance_value = float(explained_variance.item())
            else:
                explained_variance_value = 0.0

        metric_totals.update({
            "explained_variance": explained_variance_value,
            "return_target_mean": float(returns.mean().item()),
            "return_target_std": float(returns.std(unbiased=False).item()),
            "value_prediction_mean": float(final_values.mean().item()),
            "value_prediction_std": float(final_values.std(unbiased=False).item()),
            "updates": float(update_count),
            "early_stopped": float(early_stopped),
        })
        return metric_totals

    def save(
        self,
        path: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        checkpoint = {
            "algorithm": "PPO",
            "state_size": self.state_size,
            "action_size": self.action_size,
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "hyperparameters": asdict(self.hyperparameters),
            "metadata": metadata or {},
        }
        torch.save(checkpoint, path)

    def load(self, path: str, load_optimizer: bool = False) -> dict[str, Any]:
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )
        if checkpoint.get("algorithm") != "PPO":
            raise ValueError("Checkpoint is not a PPO checkpoint")
        if int(checkpoint["state_size"]) != self.state_size:
            raise ValueError(
                f"Checkpoint state size {checkpoint['state_size']} does not match "
                f"environment state size {self.state_size}"
            )
        if int(checkpoint["action_size"]) != self.action_size:
            raise ValueError(
                f"Checkpoint action size {checkpoint['action_size']} does not match "
                f"environment action size {self.action_size}"
            )

        self.network.load_state_dict(checkpoint["network_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("metadata", {})
