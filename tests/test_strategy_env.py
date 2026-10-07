"""Tests for src.models.advanced.strategy_env and rl_agent."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.models.advanced.rl_agent import agent_policy, pit_decisions, train_ppo
from src.models.advanced.strategy_env import (
    DSQ_PENALTY_S,
    PitStopEnv,
    RaceState,
    fixed_strategy_policy,
    reactive_policy,
    rollout,
)
from src.models.baselines.monte_carlo import Strategy, draw_lap_noise, draw_safety_car, simulate_race_times
from src.models.baselines.race_model import reference_race
from src.models.baselines.reactive import ReactivePolicy, simulate_reactive_py

PARAMS = reference_race()
L = PARAMS.total_laps


@pytest.fixture(scope="module")
def draws():
    rng = np.random.default_rng(11)
    return draw_safety_car(rng, 400, L, 0.04, PARAMS.sc_duration_laps), draw_lap_noise(rng, 400, PARAMS)


# --- RaceState / rollout: same physics as the Monte Carlo simulators ------------------

@pytest.mark.parametrize("stints", [(("MEDIUM", 24), ("HARD", 33)), (("SOFT", 15), ("HARD", 22), ("MEDIUM", 20))])
def test_rollout_of_fixed_strategy_matches_monte_carlo(draws, stints) -> None:
    sc, noise = draws
    strategy = Strategy(stints)
    np.testing.assert_allclose(
        rollout(fixed_strategy_policy(strategy, PARAMS), PARAMS, sc, noise, start_compound=stints[0][0]),
        simulate_race_times(strategy, PARAMS, sc, noise),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "policy",
    [ReactivePolicy("MEDIUM", "HARD", 26, 12), ReactivePolicy("MEDIUM", "HARD", 24, 8, "SOFT", 10, 8)],
    ids=lambda p: p.name,
)
def test_rollout_of_reactive_policy_matches_reference(draws, policy) -> None:
    sc, noise = draws
    np.testing.assert_allclose(
        rollout(reactive_policy(policy, PARAMS), PARAMS, sc, noise),
        simulate_reactive_py(policy, PARAMS, sc, noise),
        rtol=1e-12,
    )


def test_one_compound_race_is_penalised(draws) -> None:
    sc, noise = draws[0][:5], draws[1][:5]
    never_pit = rollout(lambda s: 0, PARAMS, sc, noise)
    # Same race computed by hand: MEDIUM all race, tyre age = lap number, no stop.
    lap = np.arange(1, L + 1)
    medium = PARAMS.tyres["MEDIUM"]
    green = PARAMS.base_lap_s + medium.offset_s + medium.deg_s_per_lap * lap + PARAMS.fuel_effect_s_per_lap * (L - lap)
    by_hand = np.where(sc, PARAMS.base_lap_s * PARAMS.sc_lap_factor, green + noise).sum(axis=1)
    np.testing.assert_allclose(never_pit, by_hand + DSQ_PENALTY_S, rtol=1e-12)
    # A fresh set of the SAME compound still counts as one compound.
    fresh_mediums = rollout(lambda s: 2 if s.lap == 20 else 0, PARAMS, sc, noise)
    one_stop = rollout(lambda s: 3 if s.lap == 20 else 0, PARAMS, sc, noise)  # MEDIUM -> HARD
    assert (fresh_mediums > one_stop + 50).all()


def test_observation_is_bounded_and_informative(draws) -> None:
    sc, noise = draws
    state = RaceState(PARAMS, sc[:3], noise[:3])
    for _ in range(30):
        obs = state.observe()
        assert obs.shape == (3, len(PARAMS.compounds) + 6) and obs.dtype == np.float32
        assert (obs >= 0).all() and (obs <= 1).all()
        np.testing.assert_array_equal(obs[:, 3 + len(PARAMS.compounds)], sc[:3, state.lap - 1])
        state.step(3 if state.lap == 10 else 0)
    assert (state.observe()[:, -1] == 1).all()  # two compounds used after the stop


def test_race_state_rejects_bad_shapes_and_extra_steps(draws) -> None:
    sc, noise = draws
    with pytest.raises(ValueError):
        RaceState(PARAMS, sc[:, :-1], noise[:, :-1])
    state = RaceState(PARAMS, sc[:1], noise[:1])
    for _ in range(L):
        state.step(np.zeros(1, dtype=np.int64))
    with pytest.raises(RuntimeError):
        state.step(np.zeros(1, dtype=np.int64))


# --- PitStopEnv ---------------------------------------------------------------------------

def test_env_passes_gymnasium_checks() -> None:
    check_env(PitStopEnv(), skip_render_check=True)


def test_env_episode_replays_a_given_race(draws) -> None:
    sc, noise = draws
    env = PitStopEnv(PARAMS)
    obs, _ = env.reset(seed=0, options={"sc_mask": sc[0], "noise": noise[0]})
    rewards, steps, info = [], 0, {}
    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(3 if steps == 23 else 0)  # pit on lap 24
        rewards.append(reward)
        steps += 1
        assert not truncated
    assert steps == L
    expected = simulate_race_times(Strategy((("MEDIUM", 24), ("HARD", 33))), PARAMS, sc[:1], noise[:1])[0]
    assert info["race_time_s"] == pytest.approx(expected)
    # Rewards are the race time, shifted by base_lap_s per lap and scaled.
    assert -np.sum(rewards) * env.reward_scale + L * PARAMS.base_lap_s == pytest.approx(expected)


def test_env_random_races_depend_on_seed() -> None:
    env = PitStopEnv(PARAMS)
    env.reset(seed=1)
    a = env.state.noise.copy()
    env.reset(seed=1)
    np.testing.assert_array_equal(env.state.noise, a)
    env.reset(seed=2)
    assert not np.array_equal(env.state.noise, a)


# --- PPO plumbing (tiny run: checks the pieces fit, not that it learns) ------------------

def test_train_and_evaluate_smoke(draws) -> None:
    sc, noise = draws
    model = train_ppo(PARAMS, total_timesteps=2048, n_envs=2, n_steps=512, batch_size=256, n_epochs=1)
    times = rollout(agent_policy(model), PARAMS, sc[:20], noise[:20])
    assert times.shape == (20,) and np.isfinite(times).all()
    decisions = pit_decisions(agent_policy(model), PARAMS, sc[:20], noise[:20])
    assert decisions.shape[1] == 4
