
import numpy as np
import pytest

from rl.replay_buffer import ReplayBuffer


def test_push_and_length():
    buffer = ReplayBuffer(capacity=10, seed=1)

    buffer.push(
        np.zeros(70), 2, 1.0,
        np.ones(70), False, np.ones(28, dtype=bool)
    )

    assert len(buffer) == 1


def test_capacity_limit():
    buffer = ReplayBuffer(capacity=2, seed=1)

    for i in range(3):
        buffer.push(
            np.full(70, i), i, float(i),
            np.full(70, i + 1), False,
            np.ones(28, dtype=bool)
        )

    assert len(buffer) == 2


def test_sample_shapes():
    buffer = ReplayBuffer(capacity=10, seed=1)

    for i in range(5):
        buffer.push(
            np.zeros(70), i, 1.0,
            np.ones(70), False,
            np.ones(28, dtype=bool)
        )

    states, actions, rewards, next_states, dones, masks = buffer.sample(3)

    assert states.shape == (3, 70)
    assert actions.shape == (3,)
    assert rewards.shape == (3,)
    assert next_states.shape == (3, 70)
    assert dones.shape == (3,)
    assert masks.shape == (3, 28)


def test_sample_too_large():
    buffer = ReplayBuffer(capacity=10)

    with pytest.raises(ValueError):
        buffer.sample(1)


def test_clear():
    buffer = ReplayBuffer()
    buffer.push(
        np.zeros(70), 0, 0.0,
        np.zeros(70), True,
        np.ones(28, dtype=bool)
    )

    buffer.clear()
    assert len(buffer) == 0