"""
Parity tests for the C++ simulator: it must match the Python reference number for number.

Build first:  .venv/bin/python -m src.cpp.build   (the tests are skipped if the module is missing)
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from src.cpp import sim
except ImportError as exc:  # pragma: no cover - depends on the local build
    pytest.skip(f"C++ module not built ({exc}); run `.venv/bin/python -m src.cpp.build`", allow_module_level=True)

from src.models.baselines.monte_carlo import (
    Strategy,
    draw_lap_noise,
    draw_safety_car,
    enumerate_strategies,
    simulate_race_times,
)
from src.models.baselines.race_model import reference_race
from src.models.baselines.reactive import ReactivePolicy, simulate_reactive_py

PARAMS = reference_race()


@pytest.fixture(scope="module")
def draws():
    """300 random races with plenty of Safety Cars, so the reactive rules actually fire."""
    rng = np.random.default_rng(7)
    sc = draw_safety_car(rng, 300, PARAMS.total_laps, 0.05, PARAMS.sc_duration_laps)
    return sc, draw_lap_noise(rng, 300, PARAMS)


POLICIES = [
    ReactivePolicy("MEDIUM", "HARD", 24),
    ReactivePolicy("MEDIUM", "HARD", 24, sc_window=10),
    ReactivePolicy("HARD", "SOFT", 35, sc_window=6, extra_compound="SOFT", extra_min_age=10, extra_min_laps_left=8),
    ReactivePolicy("SOFT", "HARD", 15, sc_window=5, extra_compound="MEDIUM", extra_min_age=12, extra_min_laps_left=12),
]


# --- static_core ---------------------------------------------------------------------

def test_static_matches_numpy_for_many_strategies(draws) -> None:
    sc, noise = draws
    strategies = enumerate_strategies(PARAMS, max_stops=2, min_stint=8, step=7)
    for strategy in strategies:
        np.testing.assert_allclose(
            sim.simulate_race_times_cpp(strategy, PARAMS, sc, noise),
            simulate_race_times(strategy, PARAMS, sc, noise),
            rtol=1e-12, err_msg=strategy.name,
        )


def test_static_returns_float64_per_sim(draws) -> None:
    sc, noise = draws
    out = sim.simulate_race_times_cpp(Strategy((("MEDIUM", 24), ("HARD", 33))), PARAMS, sc, noise)
    assert out.shape == (300,) and out.dtype == np.float64


# --- reactive_core -------------------------------------------------------------------

@pytest.mark.parametrize("policy", POLICIES, ids=lambda p: p.name)
def test_reactive_matches_python_reference(draws, policy) -> None:
    sc, noise = draws
    np.testing.assert_allclose(
        sim.simulate_reactive_cpp(policy, PARAMS, sc, noise),
        simulate_reactive_py(policy, PARAMS, sc, noise),
        rtol=1e-12,
    )


def test_reactive_batch_equals_individual_calls(draws) -> None:
    sc, noise = draws
    batch = sim.simulate_reactive_batch_cpp(POLICIES, PARAMS, sc, noise)
    assert batch.shape == (len(POLICIES), 300)
    for row, policy in zip(batch, POLICIES):
        np.testing.assert_allclose(row, sim.simulate_reactive_cpp(policy, PARAMS, sc, noise), rtol=1e-12)


# --- pybind11 layer (provided) -------------------------------------------------------

def test_wrong_shapes_raise_value_error(draws) -> None:
    sc, noise = draws
    with pytest.raises(ValueError):
        sim.simulate_reactive_cpp(POLICIES[0], PARAMS, sc[:, :-1], noise[:, :-1])
    with pytest.raises(ValueError):
        sim.simulate_race_times_cpp(Strategy((("MEDIUM", 24), ("HARD", 33))), PARAMS, sc[:10], noise)


def test_non_contiguous_input_is_handled(draws) -> None:
    # A Fortran-ordered copy is NOT C-contiguous: pybind11 must convert it, not misread it.
    sc, noise = draws
    expected = sim.simulate_reactive_cpp(POLICIES[1], PARAMS, sc, noise)
    got = sim.simulate_reactive_cpp(POLICIES[1], PARAMS, np.asfortranarray(sc), np.asfortranarray(noise))
    np.testing.assert_allclose(got, expected, rtol=1e-12)
