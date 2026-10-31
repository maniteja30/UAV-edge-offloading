
from collections import deque
import random

import numpy as np


class ReplayBuffer:
    """Stores transitions for experience replay."""

    def __init__(self, capacity=50000, seed=None):
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        next_action_mask,
    ):
        """Store one transition."""
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32).copy(),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32).copy(),
                bool(done),
                np.asarray(next_action_mask, dtype=np.bool_).copy(),
            )
        )

    def sample(self, batch_size):
        """Return a random batch of transitions."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if batch_size > len(self.buffer):
            raise ValueError("Not enough transitions in replay buffer")

        batch = self.rng.sample(list(self.buffer), batch_size)

        states, actions, rewards, next_states, dones, masks = zip(*batch)

        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
            np.stack(masks),
        )

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
