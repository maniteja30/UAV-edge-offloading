
import numpy as np
import torch

from rl.dqn_agent import DQNAgent
from rl.replay_buffer import ReplayBuffer


def test_network_output_dimensions():
    agent = DQNAgent(seed=1)

    state = torch.zeros((1, 70), device=agent.device)
    q_values = agent.online_network(state)

    assert q_values.shape == (1, 28)


def test_action_selection_returns_valid_action():
    agent = DQNAgent(seed=1)

    state = np.zeros(70, dtype=np.float32)
    mask = np.zeros(28, dtype=bool)
    mask[[2, 5, 10]] = True

    action = agent.select_action(state, mask, epsilon=0.0)

    assert action in [2, 5, 10]


def test_exploration_returns_valid_action():
    agent = DQNAgent(seed=1)

    state = np.zeros(70, dtype=np.float32)
    mask = np.zeros(28, dtype=bool)
    mask[[3, 7, 12]] = True

    for _ in range(20):
        action = agent.select_action(state, mask, epsilon=1.0)
        assert action in [3, 7, 12]


def test_all_invalid_actions_return_none():
    agent = DQNAgent(seed=1)

    state = np.zeros(70, dtype=np.float32)
    mask = np.zeros(28, dtype=bool)

    assert agent.select_action(state, mask) is None


def test_update_with_replay_buffer():
    agent = DQNAgent(seed=1)
    buffer = ReplayBuffer(capacity=100, seed=1)

    for _ in range(10):
        mask = np.ones(28, dtype=bool)

        buffer.push(
            np.zeros(70),
            2,
            1.0,
            np.ones(70),
            False,
            mask,
        )

    loss = agent.update(buffer, batch_size=4)

    assert loss is not None
    assert np.isfinite(loss)


def test_update_handles_all_invalid_next_actions():
    agent = DQNAgent(seed=1)
    buffer = ReplayBuffer(capacity=10, seed=1)

    buffer.push(
        np.zeros(70),
        2,
        1.0,
        np.ones(70),
        False,
        np.zeros(28, dtype=bool),
    )

    loss = agent.update(buffer, batch_size=1)

    assert loss is not None
    assert np.isfinite(loss)


def test_target_network_update():
    agent = DQNAgent(seed=1)

    with torch.no_grad():
        for parameter in agent.online_network.parameters():
            parameter.add_(1.0)

    agent.update_target_network()

    for online, target in zip(
        agent.online_network.parameters(),
        agent.target_network.parameters(),
    ):
        assert torch.equal(online, target)


def test_insufficient_replay_buffer():
    agent = DQNAgent(seed=1)
    buffer = ReplayBuffer(capacity=10)

    assert agent.update(buffer, batch_size=4) is None
