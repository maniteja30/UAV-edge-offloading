
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
