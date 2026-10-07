"""
Python-facing wrappers around the compiled module: same signatures as the Python
simulators, so they can be swapped in anywhere (e.g. ``evaluate_policies(simulate=...)``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from src.cpp import _pitwall_sim as _sim
from src.models.baselines.monte_carlo import Strategy, strategy_arrays
from src.models.baselines.race_model import RaceParams
from src.models.baselines.reactive import ReactivePolicy


def race_config(params: RaceParams) -> _sim.RaceConfig:
    cfg = _sim.RaceConfig()
    cfg.total_laps = params.total_laps
    cfg.base_lap_s = params.base_lap_s
    cfg.offsets = [params.tyres[c].offset_s for c in params.compounds]
    cfg.degs = [params.tyres[c].deg_s_per_lap for c in params.compounds]
    cfg.pit_loss_s = params.pit_loss_s
    cfg.sc_lap_factor = params.sc_lap_factor
    cfg.sc_pit_loss_factor = params.sc_pit_loss_factor
    cfg.fuel_effect_s_per_lap = params.fuel_effect_s_per_lap
    return cfg


def reactive_config(policy: ReactivePolicy, params: RaceParams) -> _sim.ReactiveConfig:
    policy.validate(params)
    idx = params.compounds.index
    pol = _sim.ReactiveConfig()
    pol.start, pol.second = idx(policy.start), idx(policy.second)
    pol.plan_lap, pol.sc_window = policy.plan_lap, policy.sc_window
    pol.extra_compound = -1 if policy.extra_compound is None else idx(policy.extra_compound)
    pol.extra_min_age, pol.extra_min_laps_left = policy.extra_min_age, policy.extra_min_laps_left
    return pol


def simulate_race_times_cpp(
    strategy: Strategy, params: RaceParams, sc_mask: NDArray[np.bool_], noise: NDArray[np.float64]
) -> NDArray[np.float64]:
    """C++ twin of ``monte_carlo.simulate_race_times``."""
    compound_idx, tyre_age, pit = strategy_arrays(strategy, params)
    return _sim.simulate_static(race_config(params), compound_idx, tyre_age, pit, sc_mask, noise)


def simulate_reactive_cpp(
    policy: ReactivePolicy, params: RaceParams, sc_mask: NDArray[np.bool_], noise: NDArray[np.float64]
) -> NDArray[np.float64]:
    """C++ twin of ``reactive.simulate_reactive_py``."""
    return _sim.simulate_reactive(race_config(params), reactive_config(policy, params), sc_mask, noise)


def simulate_reactive_batch_cpp(
    policies: Sequence[ReactivePolicy], params: RaceParams, sc_mask: NDArray[np.bool_], noise: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Many policies in ONE C++ call, shape ``(n_policies, n_sims)``."""
    configs = [reactive_config(p, params) for p in policies]
    return _sim.simulate_reactive_batch(race_config(params), configs, sc_mask, noise)
