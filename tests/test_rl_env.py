
from types import SimpleNamespace

import pytest

from tasksim.simulator import Config
from tasksim.rl_env import OffloadingRLEnv


def test_environment_dimensions():
    env = OffloadingRLEnv(
        Config(task_count=10)
    )

    state, info = env.reset()

    assert len(state) == 70
    assert info["state_size"] == 70
    assert info["action_size"] == 28


def test_action_mask():
    env = OffloadingRLEnv(
        Config(task_count=10)
    )

    mask = env.get_action_mask()

    assert len(mask) == 28
    assert all(isinstance(x, bool) for x in mask)


def test_action_mask_uses_soft_deadline_handling(monkeypatch):
    env = OffloadingRLEnv(
        Config(task_count=1)
    )

    evaluations = [
        SimpleNamespace(
            latency_s=3.0,
            objective=2.0,
            deadline_met=False,
        ),
        SimpleNamespace(
            latency_s=1.0,
            objective=1.0,
            deadline_met=True,
        ),
        SimpleNamespace(
            latency_s=float("inf"),
            objective=float("inf"),
            deadline_met=False,
        ),
    ]

    evaluations.extend([
        SimpleNamespace(
            latency_s=1.0,
            objective=1.0,
            deadline_met=True,
        )
        for _ in range(25)
    ])

    evaluation_iter = iter(evaluations)

    monkeypatch.setattr(
        env.sim,
        "evaluate_pair",
        lambda *args, **kwargs: next(evaluation_iter),
    )

    env._action_mask_cache = None

    mask = env.get_action_mask()

    assert len(mask) == 28

    assert mask[0] is True
    assert mask[1] is True
    assert mask[2] is False


def test_environment_step():
    env = OffloadingRLEnv(
        Config(task_count=10)
    )

    state, info = env.reset()

    mask = env.get_action_mask()

    if any(mask):
        action = mask.index(True)
    else:
        action = None

    next_state, reward, terminated, truncated, step_info = (
        env.step(action)
    )

    assert len(next_state) == 70
    assert isinstance(reward, float)
    assert truncated is False
    assert "objective" in step_info


def test_deadline_miss_penalty():
    env = OffloadingRLEnv(
        Config(task_count=1),
        deadline_miss_penalty=5.0,
    )

    env.get_action_mask = lambda: [True] + [False] * 27

    result = SimpleNamespace(
        destinations=("U0", "U1"),
        latency_s=3.0,
        energy_j=1.0,
        load_imbalance=0.1,
        deadline_met=False,
        objective=2.0,
    )

    env.sim._simulate_pair = lambda *args, **kwargs: (
        result,
        env.nodes,
        env.network,
    )

    _, reward, _, _, info = env.step(0)

    assert reward == -7.0
    assert info["objective"] == 2.0
    assert info["deadline_met"] is False


def test_no_deadline_penalty_when_deadline_is_met():
    env = OffloadingRLEnv(
        Config(task_count=1),
        deadline_miss_penalty=5.0,
    )

    env.get_action_mask = lambda: [True] + [False] * 27

    result = SimpleNamespace(
        destinations=("U0", "U1"),
        latency_s=1.0,
        energy_j=1.0,
        load_imbalance=0.1,
        deadline_met=True,
        objective=2.0,
    )

    env.sim._simulate_pair = lambda *args, **kwargs: (
        result,
        env.nodes,
        env.network,
    )

    _, reward, _, _, info = env.step(0)

    assert reward == -2.0
    assert info["objective"] == 2.0
    assert info["deadline_met"] is True


def test_zero_deadline_penalty_preserves_original_reward():
    env = OffloadingRLEnv(
        Config(task_count=1),
        deadline_miss_penalty=0.0,
    )

    env.get_action_mask = lambda: [True] + [False] * 27

    result = SimpleNamespace(
        destinations=("U0", "U1"),
        latency_s=3.0,
        energy_j=1.0,
        load_imbalance=0.1,
        deadline_met=False,
        objective=2.0,
    )

    env.sim._simulate_pair = lambda *args, **kwargs: (
        result,
        env.nodes,
        env.network,
    )

    _, reward, _, _, info = env.step(0)

    assert reward == -2.0
    assert info["objective"] == 2.0


def test_negative_deadline_miss_penalty_rejected():
    with pytest.raises(ValueError):
        OffloadingRLEnv(
            Config(task_count=1),
            deadline_miss_penalty=-1.0,
        )


def test_episode_completion():
    env = OffloadingRLEnv(
        Config(task_count=5)
    )

    state, info = env.reset()

    for _ in range(5):
        mask = env.get_action_mask()

        action = (
            mask.index(True) if any(mask) else None
        )

        state, reward, terminated, truncated, info = (
            env.step(action)
        )

    assert terminated is True
    assert len(env.results) == 5


def test_reset_restores_episode():
    env = OffloadingRLEnv(
        Config(task_count=5)
    )

    env.reset()

    mask = env.get_action_mask()
    action = mask.index(True) if any(mask) else None

    env.step(action)

    state, info = env.reset()

    assert len(state) == 70
    assert env.current_index == 0
    assert len(env.results) == 0


def test_episode_summary_includes_network_lifetime_metrics():
    env = OffloadingRLEnv(Config(task_count=3))

    for _ in range(3):
        mask = env.get_action_mask()
        action = mask.index(True) if any(mask) else None
        env.step(action)

    summary = env.get_episode_summary()

    assert summary["uav_count"] == 13
    assert summary["active_uav_count"] <= 13
    assert summary["depleted_uav_count"] >= 0
    assert summary["total_initial_energy_j"] > 0.0
    assert summary["total_residual_energy_j"] <= summary["total_initial_energy_j"]
    assert len(summary["residual_energy_by_uav_j"]) == 13
    assert isinstance(summary["energy_timeline"], list)
    assert summary["lifetime_threshold_fraction"] == 0.01
