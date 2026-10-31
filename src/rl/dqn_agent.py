
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Maps an environment state to Q-values for all actions."""

    def __init__(self, state_size=70, action_size=28):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, state):
        return self.network(state)


class DQNAgent:
    """Double DQN agent with experience replay and a target network."""

    def __init__(
        self,
        state_size=70,
        action_size=28,
        learning_rate=1e-3,
        gamma=0.99,
        device=None,
        seed=None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.online_network = QNetwork(
            state_size,
            action_size,
        ).to(self.device)

        self.target_network = QNetwork(
            state_size,
            action_size,
        ).to(self.device)

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate,
        )

        self.loss_function = nn.SmoothL1Loss()
        self.rng = random.Random(seed)

    def select_action(
        self,
        state,
        action_mask,
        epsilon=0.0,
    ):
        """Select an action using epsilon-greedy selection."""

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "epsilon must be between 0 and 1"
            )

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        action_mask = np.asarray(
            action_mask,
            dtype=bool,
        )

        if state.shape != (self.state_size,):
            raise ValueError(
                f"Expected state shape ({self.state_size},), "
                f"got {state.shape}"
            )

        if action_mask.shape != (self.action_size,):
            raise ValueError(
                f"Expected action mask shape ({self.action_size},), "
                f"got {action_mask.shape}"
            )

        valid_actions = np.flatnonzero(action_mask)

        if len(valid_actions) == 0:
            return None

        # Exploration.
        if self.rng.random() < epsilon:
            return int(
                self.rng.choice(valid_actions.tolist())
            )

        # Exploitation.
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.online_network(
                state_tensor
            ).squeeze(0)

            mask_tensor = torch.as_tensor(
                action_mask,
                dtype=torch.bool,
                device=self.device,
            )

            q_values = q_values.masked_fill(
                ~mask_tensor,
                float("-inf"),
            )

            return int(
                torch.argmax(q_values).item()
            )

    def update(
        self,
        replay_buffer,
        batch_size=64,
    ):
        """
        Perform one Double DQN optimization step.

        The online network selects the next action.
        The target network evaluates that action.
        """

        if len(replay_buffer) < batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            next_action_masks,
        ) = replay_buffer.sample(batch_size)

        states = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        )

        next_action_masks = torch.as_tensor(
            next_action_masks,
            dtype=torch.bool,
            device=self.device,
        )

        # Current Q-values for selected actions.
        current_q_values = self.online_network(
            states
        ).gather(
            1,
            actions,
        ).squeeze(1)

        with torch.no_grad():

            # Online network selects the next action.
            online_next_q_values = self.online_network(
                next_states
            )

            online_next_q_values = online_next_q_values.masked_fill(
                ~next_action_masks,
                float("-inf"),
            )

            has_valid_next_action = next_action_masks.any(
                dim=1
            )

            next_actions = online_next_q_values.argmax(
                dim=1,
                keepdim=True,
            )

            # Target network evaluates the selected action.
            target_next_q_values = self.target_network(
                next_states
            )

            next_q_values = target_next_q_values.gather(
                1,
                next_actions,
            ).squeeze(1)

            # No bootstrap when the next state has no valid action.
            next_q_values = torch.where(
                has_valid_next_action,
                next_q_values,
                torch.zeros_like(next_q_values),
            )

            target_q_values = (
                rewards
                + self.gamma
                * (1.0 - dones)
                * next_q_values
            )

        loss = self.loss_function(
            current_q_values,
            target_q_values,
        )

        self.optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self):
        """Copy online network weights to the target network."""

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

    def save(self, path):
        """Save the online and target network weights."""

        torch.save(
            {
                "online_network": self.online_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "state_size": self.state_size,
                "action_size": self.action_size,
                "gamma": self.gamma,
            },
            path,
        )

    def load(self, path):
        """Load saved network weights."""

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )

        if checkpoint["state_size"] != self.state_size:
            raise ValueError(
                "Checkpoint state size does not match"
            )

        if checkpoint["action_size"] != self.action_size:
            raise ValueError(
                "Checkpoint action size does not match"
            )

        self.online_network.load_state_dict(
            checkpoint["online_network"]
        )

        self.target_network.load_state_dict(
            checkpoint["target_network"]
        )

        self.target_network.eval()
