import numpy as np

from rl.ppo_agent import PPOAgent, PPOHyperparameters, PPORolloutBuffer


def make_agent(state_size=70, action_size=28, reward_scale=1000.0):
    hp = PPOHyperparameters(
        learning_rate=3e-4,
        update_epochs=1,
        batch_size=4,
        reward_scale=reward_scale,
    )
    return PPOAgent(
        state_size=state_size,
        action_size=action_size,
        hyperparameters=hp,
        seed=11,
    )


def test_actor_output_dimensions_and_value():
    agent = make_agent()
    state = np.zeros(70, dtype=np.float32)
    mask = np.ones(28, dtype=np.bool_)

    action, log_prob, value = agent.select_action(state, mask)

    assert 0 <= action < 28
    assert np.isfinite(log_prob)
    assert np.isfinite(value)


def test_action_mask_never_selects_infeasible_action():
    agent = make_agent()
    state = np.zeros(70, dtype=np.float32)
    mask = np.zeros(28, dtype=np.bool_)
    mask[[3, 9, 17]] = True

    for _ in range(30):
        action, _log_prob, _value = agent.select_action(state, mask)
        assert action in {3, 9, 17}


def test_deterministic_action_respects_mask():
    agent = make_agent()
    state = np.ones(70, dtype=np.float32)
    mask = np.zeros(28, dtype=np.bool_)
    mask[12] = True

    action, _log_prob, _value = agent.select_action(
        state, mask, deterministic=True
    )
    assert action == 12


def test_all_false_mask_returns_none_without_selecting_invalid_action():
    agent = make_agent()
    state = np.zeros(70, dtype=np.float32)
    mask = np.zeros(28, dtype=np.bool_)

    action, log_prob, value = agent.select_action(state, mask)

    assert action is None
    assert log_prob == 0.0
    assert np.isfinite(value)


def test_update_accepts_forced_no_action_and_returns_finite_metrics():
    agent = make_agent()
    rollout = PPORolloutBuffer()
    state = np.zeros(70, dtype=np.float32)
    all_valid = np.ones(28, dtype=np.bool_)
    no_valid = np.zeros(28, dtype=np.bool_)

    action, log_prob, value = agent.select_action(state, all_valid)
    rollout.add(state, action, -1.0, False, log_prob, value, all_valid, True)

    _action, _log_prob, value = agent.select_action(state, no_valid)
    rollout.add(state, 0, -10.0, True, 0.0, value, no_valid, False)

    metrics = agent.update(rollout, next_value=0.0)
    for key in (
        "loss", "policy_loss", "value_loss", "entropy", "approx_kl",
        "clip_fraction", "explained_variance", "return_target_mean",
    ):
        assert np.isfinite(metrics[key])
    assert metrics["updates"] > 0


def test_internal_reward_scaling_keeps_critic_targets_small():
    agent = make_agent(reward_scale=1000.0)
    rollout = PPORolloutBuffer()
    state = np.zeros(70, dtype=np.float32)
    mask = np.ones(28, dtype=np.bool_)

    for index, reward in enumerate((-1000.0, -2000.0)):
        action, log_prob, value = agent.select_action(state, mask)
        rollout.add(
            state, action, reward, index == 1, log_prob, value, mask, True
        )

    metrics = agent.update(rollout, next_value=0.0)
    assert metrics["return_target_mean"] < 0.0
    assert abs(metrics["return_target_mean"]) < 10.0
    assert np.isfinite(metrics["value_loss"])


def test_invalid_observation_or_action_mask_shape_raises():
    agent = make_agent()
    with np.testing.assert_raises(ValueError):
        agent.select_action(np.zeros(69), np.ones(28, dtype=bool))
    with np.testing.assert_raises(ValueError):
        agent.select_action(np.zeros(70), np.ones(27, dtype=bool))


def test_invalid_reward_scale_is_rejected():
    with np.testing.assert_raises(ValueError):
        PPOHyperparameters(reward_scale=0.0)
