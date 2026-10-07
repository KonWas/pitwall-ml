"""Tests for src.models.baselines.reactive (the Python reference the C++ port must match)."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.baselines.monte_carlo import Strategy, evaluate_strategies, simulate_race_times
from src.models.baselines.race_model import RaceParams, TyreModel, reference_race
from src.models.baselines.reactive import (
    ReactivePolicy,
    enumerate_reactive_policies,
    evaluate_policies,
    simulate_reactive_py,
)


def _params(total_laps: int = 20) -> RaceParams:
    return RaceParams(
        total_laps=total_laps,
        base_lap_s=90.0,
        tyres={"SOFT": TyreModel(-0.8, 0.12), "MEDIUM": TyreModel(-0.4, 0.07), "HARD": TyreModel(0.0, 0.04)},
        pit_loss_s=20.0,
        lap_noise_s=0.3,
        fuel_effect_s_per_lap=0.06,
    )


def _draws(n_sims: int, total_laps: int, sc_laps: tuple[int, ...] = (), seed: int = 0):
    """Noise for every sim; SC on the given 1-based laps in EVERY sim."""
    noise = np.random.default_rng(seed).normal(0, 0.3, size=(n_sims, total_laps))
    sc = np.zeros((n_sims, total_laps), dtype=bool)
    for lap in sc_laps:
        sc[:, lap - 1] = True
    return sc, noise


def _static(params: RaceParams, sc, noise, *stints: tuple[str, int]):
    return simulate_race_times(Strategy(tuple(stints)), params, sc, noise)


def test_reference_race_prefers_a_medium_hard_one_stop() -> None:
    params = reference_race()
    assert params.compounds == ("SOFT", "MEDIUM", "HARD")
    from src.models.baselines.monte_carlo import enumerate_strategies

    result = evaluate_strategies(enumerate_strategies(params, 2, 8, 2), params, n_sims=500)
    best = result.iloc[0]
    assert best["n_stops"] == 1
    assert set(best["strategy"].replace("-", "")) - set("0123456789") == {"M", "H"}


def test_policy_name_and_validation() -> None:
    params = _params()
    assert ReactivePolicy("MEDIUM", "HARD", 8, 4, "SOFT").name == "M8-H w4 +S"
    ReactivePolicy("MEDIUM", "HARD", 8).validate(params)
    for bad in [
        ReactivePolicy("HARD", "HARD", 8),        # one compound
        ReactivePolicy("MEDIUM", "HARD", 20),     # plan_lap must be < total_laps
        ReactivePolicy("MEDIUM", "INTER", 8),     # no tyre model
        ReactivePolicy("MEDIUM", "HARD", 8, -1),  # negative window
    ]:
        with pytest.raises(ValueError):
            bad.validate(params)


def test_no_reaction_equals_static_one_stop() -> None:
    params = _params()
    sc, noise = _draws(5, 20, sc_laps=(3, 4, 15))
    reactive = simulate_reactive_py(ReactivePolicy("MEDIUM", "HARD", 8), params, sc, noise)
    np.testing.assert_allclose(reactive, _static(params, sc, noise, ("MEDIUM", 8), ("HARD", 12)))


def test_sc_inside_window_brings_the_stop_forward() -> None:
    params = _params()
    sc, noise = _draws(3, 20, sc_laps=(5, 6))  # window for plan lap 8, width 4: laps 4..7
    reactive = simulate_reactive_py(ReactivePolicy("MEDIUM", "HARD", 8, sc_window=4), params, sc, noise)
    # Pits on lap 5, the first SC lap in the window, at the cheap SC price.
    np.testing.assert_allclose(reactive, _static(params, sc, noise, ("MEDIUM", 5), ("HARD", 15)))


def test_sc_outside_window_keeps_the_plan() -> None:
    params = _params()
    sc, noise = _draws(3, 20, sc_laps=(2, 3))  # before the window opens on lap 4
    reactive = simulate_reactive_py(ReactivePolicy("MEDIUM", "HARD", 8, sc_window=4), params, sc, noise)
    np.testing.assert_allclose(reactive, _static(params, sc, noise, ("MEDIUM", 8), ("HARD", 12)))


def test_extra_stop_under_late_sc() -> None:
    params = _params()
    policy = ReactivePolicy("MEDIUM", "HARD", 6, extra_compound="SOFT", extra_min_age=5, extra_min_laps_left=3)
    sc, noise = _draws(3, 20, sc_laps=(14,))  # age on lap 14 = 8 >= 5, 6 laps left >= 3
    np.testing.assert_allclose(
        simulate_reactive_py(policy, params, sc, noise),
        _static(params, sc, noise, ("MEDIUM", 6), ("HARD", 8), ("SOFT", 6)),
    )


def test_extra_stop_needs_old_tyres_and_enough_laps() -> None:
    params = _params()
    policy = ReactivePolicy("MEDIUM", "HARD", 6, extra_compound="SOFT", extra_min_age=5, extra_min_laps_left=3)
    plain = ("MEDIUM", 6), ("HARD", 14)
    for sc_lap in (9, 18):  # lap 9: tyres only 3 laps old; lap 18: only 2 laps left
        sc, noise = _draws(2, 20, sc_laps=(sc_lap,))
        np.testing.assert_allclose(simulate_reactive_py(policy, params, sc, noise), _static(params, sc, noise, *plain))


def test_at_most_one_extra_stop() -> None:
    params = _params()
    policy = ReactivePolicy("MEDIUM", "HARD", 4, extra_compound="SOFT", extra_min_age=2, extra_min_laps_left=2)
    sc, noise = _draws(2, 20, sc_laps=(8, 15))  # two SCs that both qualify
    np.testing.assert_allclose(
        simulate_reactive_py(policy, params, sc, noise),
        _static(params, sc, noise, ("MEDIUM", 4), ("HARD", 4), ("SOFT", 12)),
    )


def test_enumerate_and_evaluate_policies() -> None:
    params = _params()
    policies = enumerate_reactive_policies(params, starts=("MEDIUM", "HARD"), plan_laps=[6, 10], sc_windows=(0, 4))
    assert len(policies) == 2 * 2 * 2  # (M,H) and (H,M) x 2 laps x 2 windows
    result = evaluate_policies(policies, params, n_sims=200, seed=3)
    assert list(result.columns) == ["policy", "mean_s", "std_s", "p05_s", "p95_s", "delta_to_best_s"]
    assert result["mean_s"].is_monotonic_increasing
    # Same draws as evaluate_strategies: a non-reacting policy matches its fixed strategy.
    static = evaluate_strategies([Strategy((("MEDIUM", 6), ("HARD", 14)))], params, n_sims=200, seed=3)
    reactive = result.set_index("policy").loc["M6-H w0", "mean_s"]
    assert reactive == pytest.approx(static["mean_s"].iloc[0])
