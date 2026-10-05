"""Tests for src.models.baselines.monte_carlo."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.baselines.monte_carlo import (
    Strategy,
    draw_safety_car,
    enumerate_strategies,
    evaluate_strategies,
    simulate_race_times,
    strategy_arrays,
)
from src.models.baselines.race_model import RaceParams, TyreModel


def _params(total_laps: int = 4, fuel: float = 0.0, **overrides) -> RaceParams:
    kwargs = {
        "base_lap_s": 90.0,
        "tyres": {"SOFT": TyreModel(-1.0, 0.5), "HARD": TyreModel(0.0, 0.1)},
        "pit_loss_s": 20.0,
        "lap_noise_s": 0.3,
        "sc_hazard_per_lap": 0.0,
        "sc_duration_laps": 3,
        "sc_lap_factor": 1.4,
        "sc_pit_loss_factor": 0.5,
        "fuel_effect_s_per_lap": fuel,
    }
    return RaceParams(total_laps=total_laps, **{**kwargs, **overrides})


S2_H2 = Strategy((("SOFT", 2), ("HARD", 2)))


def _no_randomness(n_sims: int, total_laps: int):
    return np.zeros((n_sims, total_laps), dtype=bool), np.zeros((n_sims, total_laps))


# --- Strategy / strategy_arrays (provided) ----------------------------------------

def test_strategy_name_and_validation() -> None:
    assert S2_H2.name == "S2-H2"
    assert S2_H2.n_stops == 1
    S2_H2.validate(4)
    with pytest.raises(ValueError):
        S2_H2.validate(5)  # doesn't cover the race
    with pytest.raises(ValueError):
        Strategy((("SOFT", 2), ("SOFT", 2))).validate(4)  # one compound only
    with pytest.raises(ValueError):
        Strategy((("SOFT", 2), ("MEDIUM", 2))).validate(4, compounds=("SOFT", "HARD"))


def test_strategy_arrays() -> None:
    params = _params(total_laps=7)
    compound_idx, tyre_age, pit = strategy_arrays(Strategy((("HARD", 3), ("SOFT", 2), ("HARD", 2))), params)
    assert params.compounds == ("SOFT", "HARD")
    assert compound_idx.tolist() == [1, 1, 1, 0, 0, 1, 1]
    assert tyre_age.tolist() == [1, 2, 3, 1, 2, 1, 2]
    assert pit.tolist() == [False, False, True, False, True, False, False]


# --- draw_safety_car ---------------------------------------------------------------

def test_safety_car_never_with_zero_hazard() -> None:
    mask = draw_safety_car(np.random.default_rng(0), 50, 60, 0.0, 4)
    assert mask.shape == (50, 60) and mask.dtype == bool
    assert not mask.any()


def test_safety_car_always_with_hazard_one() -> None:
    # Starts on lap 1; each time one ends, the next starts on the following lap.
    assert draw_safety_car(np.random.default_rng(0), 5, 13, 1.0, 4).all()


def test_safety_car_is_reproducible_from_seed() -> None:
    a = draw_safety_car(np.random.default_rng(42), 100, 60, 0.05, 4)
    b = draw_safety_car(np.random.default_rng(42), 100, 60, 0.05, 4)
    np.testing.assert_array_equal(a, b)


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of every run of True values in a 1-D boolean array."""
    padded = np.r_[False, row, False].astype(int)
    starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    return list(zip(starts, ends - starts))


def test_safety_car_periods_have_the_right_length() -> None:
    total, duration = 60, 4
    mask = draw_safety_car(np.random.default_rng(1), 500, total, 0.05, duration)
    for row in mask:
        for start, length in _runs(row):
            if start + length < total:  # runs cut off by the chequered flag can be shorter
                # back-to-back deployments merge into one run: a multiple of the duration
                assert length % duration == 0, (start, length)


def test_safety_car_long_run_fraction_matches_theory() -> None:
    # Renewal argument: a green spell lasts (1 - h) / h laps on average, then d SC laps,
    # so the long-run share of SC laps is d / (d + (1 - h) / h).
    h, d = 0.02, 4
    mask = draw_safety_car(np.random.default_rng(3), 2000, 1000, h, d)
    assert mask.mean() == pytest.approx(d / (d + (1 - h) / h), rel=0.03)


# --- simulate_race_times -----------------------------------------------------------

def test_simulate_by_hand() -> None:
    # SOFT ages 1,2: 89.5, 90.0 | HARD ages 1,2: 90.1, 90.2 | + one 20 s stop
    sc, noise = _no_randomness(3, 4)
    totals = simulate_race_times(S2_H2, _params(), sc, noise)
    assert totals.shape == (3,) and totals.dtype == np.float64
    np.testing.assert_allclose(totals, 379.8)


def test_simulate_adds_fuel_for_laps_remaining() -> None:
    sc, noise = _no_randomness(1, 4)
    totals = simulate_race_times(S2_H2, _params(fuel=0.06), sc, noise)
    np.testing.assert_allclose(totals, 379.8 + 0.06 * (3 + 2 + 1 + 0))


def test_simulate_safety_car_lap_and_cheap_stop() -> None:
    sc, noise = _no_randomness(2, 4)
    sc[1, 1] = True  # sim 2: SC on lap 2, which is the in-lap
    noise[:, 1] = 5.0  # noise must be ignored on SC laps
    totals = simulate_race_times(S2_H2, _params(), sc, noise)
    assert totals[0] == pytest.approx(379.8 + 5.0)
    # lap 2 becomes 90 * 1.4 = 126 instead of 90.0, and the stop costs 10 s instead of 20.
    assert totals[1] == pytest.approx(379.8 - 90.0 + 126.0 - 10.0)


def test_simulate_noise_is_additive_per_sim() -> None:
    rng = np.random.default_rng(0)
    sc, _ = _no_randomness(4, 4)
    noise = rng.normal(0, 0.3, size=(4, 4))
    totals = simulate_race_times(S2_H2, _params(), sc, noise)
    np.testing.assert_allclose(totals, 379.8 + noise.sum(axis=1))


def test_simulate_rejects_wrong_shapes() -> None:
    with pytest.raises(ValueError):
        simulate_race_times(S2_H2, _params(), *_no_randomness(2, 5))


# --- enumerate_strategies / evaluate_strategies (provided) --------------------------

def test_enumerate_strategies_are_all_legal() -> None:
    params = _params(total_laps=30)
    strategies = enumerate_strategies(params, max_stops=2, min_stint=8)
    assert len(strategies) == len({s.name for s in strategies})
    for s in strategies:
        s.validate(30, params.compounds)
        assert all(n >= 8 for _, n in s.stints)
    # 1-stop: 2 compound orders x stint lengths 8..22 -> 2 * 15
    assert sum(s.n_stops == 1 for s in strategies) == 30


def test_evaluate_strategies_common_random_numbers() -> None:
    params = _params(total_laps=40, sc_hazard_per_lap=0.03, fuel=0.06)
    a = Strategy((("SOFT", 15), ("HARD", 25)))
    b = Strategy((("HARD", 25), ("SOFT", 15)))
    result = evaluate_strategies([a, a, b], params, n_sims=500, seed=0)
    assert list(result.columns) == [
        "strategy", "n_stops", "mean_s", "std_s", "p05_s", "p95_s", "delta_to_best_s", "win_share"
    ]
    dup = result[result["strategy"] == a.name]
    assert dup["mean_s"].nunique() == 1  # identical strategy, identical random races
    assert result["delta_to_best_s"].iloc[0] == 0.0
    assert result["mean_s"].is_monotonic_increasing
    assert result["win_share"].sum() == pytest.approx(1.0)
