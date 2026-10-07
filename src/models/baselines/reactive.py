"""
Reactive (rule-based) strategies: a one-stop plan that responds to Safety Cars.

A fixed strategy pits on fixed laps, so it only gets a cheap stop under the Safety
Car by luck (notebook 04, section 6). A reactive policy decides lap by lap:

    1. Planned stop: pit from `start` to `second` on lap `plan_lap`, or EARLIER if a
       Safety Car is out on any lap in [plan_lap - sc_window, plan_lap).
    2. Optional extra stop: after the planned stop, if a Safety Car is out, the tyres
       are at least `extra_min_age` laps old and at least `extra_min_laps_left` laps
       remain after this one, pit once more onto fresh `extra_compound` tyres.

Decisions depend on each simulated race's own Safety Cars, so every simulation takes
a different path. That branching is what NumPy cannot vectorise across laps, and
what the C++ port (src/cpp) makes fast. `simulate_reactive_py` below is the
specification: deliberately plain loops, slow, and easy to check.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.models.baselines.monte_carlo import draw_lap_noise, draw_safety_car
from src.models.baselines.race_model import RaceParams


@dataclass(frozen=True)
class ReactivePolicy:
    start: str
    second: str
    plan_lap: int
    sc_window: int = 0
    extra_compound: str | None = None
    extra_min_age: int = 15
    extra_min_laps_left: int = 10

    @property
    def name(self) -> str:
        """E.g. "M24-H w8 +S" (plan MEDIUM->HARD on lap 24, 8-lap SC window, extra stop onto SOFT)."""
        label = f"{self.start[0]}{self.plan_lap}-{self.second[0]} w{self.sc_window}"
        return label + (f" +{self.extra_compound[0]}" if self.extra_compound else "")

    def validate(self, params: RaceParams) -> None:
        """Raise ValueError unless the policy is legal for ``params``."""
        compounds = [self.start, self.second] + ([self.extra_compound] if self.extra_compound else [])
        if any(c not in params.compounds for c in compounds):
            raise ValueError(f"{self.name}: uses a compound without a tyre model")
        if self.start == self.second:
            raise ValueError(f"{self.name}: start and second compound must differ (two-compound rule)")
        if not 1 <= self.plan_lap < params.total_laps:
            raise ValueError(f"{self.name}: plan_lap must be in 1..{params.total_laps - 1}")
        if self.sc_window < 0 or self.extra_min_age < 1 or self.extra_min_laps_left < 1:
            raise ValueError(f"{self.name}: sc_window >= 0, extra_min_age >= 1, extra_min_laps_left >= 1")


def simulate_reactive_py(
    policy: ReactivePolicy,
    params: RaceParams,
    sc_mask: NDArray[np.bool_],
    noise: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Total race time of ``policy`` in each simulated race (pure-Python reference).

    Lap model identical to ``simulate_race_times``: a green lap costs
    base + offset[c] + deg[c] * age + fuel * (L - l) + noise, an SC lap costs
    base * sc_lap_factor, and an in-lap adds pit_loss (x sc_pit_loss_factor under SC).

    Per simulation, lap l = 1..L, with the lap's SC flag `sc`:
        1. add the lap time for the current compound and tyre age;
        2. decide whether lap l is an in-lap (rules in the module docstring); the
           planned stop is checked first, the extra stop only once it is done;
        3. if pitting: add the pit cost, switch compound, next lap's age = 1;
           otherwise age += 1.

    Returns:
        float64 array, shape ``(n_sims,)``.
    """
    policy.validate(params)
    n_sims, L = sc_mask.shape
    if noise.shape != sc_mask.shape or L != params.total_laps:
        raise ValueError(f"sc_mask {sc_mask.shape} and noise {noise.shape} must be (n_sims, {params.total_laps})")
    tyres = params.tyres
    totals = np.empty(n_sims)
    for s in range(n_sims):
        compound, age = policy.start, 1
        stopped = extra_done = False
        total = 0.0
        for lap in range(1, L + 1):
            sc = bool(sc_mask[s, lap - 1])
            if sc:
                total += params.base_lap_s * params.sc_lap_factor
            else:
                tyre = tyres[compound]
                total += (
                    params.base_lap_s + tyre.offset_s + tyre.deg_s_per_lap * age
                    + params.fuel_effect_s_per_lap * (L - lap) + noise[s, lap - 1]
                )

            next_compound = None
            if not stopped:
                in_window = policy.plan_lap - policy.sc_window <= lap < policy.plan_lap
                if lap == policy.plan_lap or (sc and in_window):
                    next_compound, stopped = policy.second, True
            elif (
                policy.extra_compound is not None
                and not extra_done
                and sc
                and age >= policy.extra_min_age
                and L - lap >= policy.extra_min_laps_left
            ):
                next_compound, extra_done = policy.extra_compound, True

            if next_compound is not None:
                total += params.pit_loss_s * (params.sc_pit_loss_factor if sc else 1.0)
                compound, age = next_compound, 1
            else:
                age += 1
        totals[s] = total
    return totals


def enumerate_reactive_policies(
    params: RaceParams,
    starts: Sequence[str] | None = None,
    plan_laps: Iterable[int] | None = None,
    sc_windows: Sequence[int] = (0, 4, 8, 12),
    extra_compounds: Sequence[str | None] = (None,),
) -> list[ReactivePolicy]:
    """Grid of legal policies: every (start, second) pair x plan lap x window x extra option."""
    compounds = params.compounds if starts is None else tuple(starts)
    laps = range(8, params.total_laps - 7) if plan_laps is None else plan_laps
    policies = []
    for (start, second), lap, window, extra in itertools.product(
        itertools.permutations(compounds, 2), laps, sc_windows, extra_compounds
    ):
        policies.append(ReactivePolicy(start, second, lap, window, extra))
    return policies


def evaluate_policies(
    policies: Iterable[ReactivePolicy],
    params: RaceParams,
    simulate: Callable[[ReactivePolicy, RaceParams, NDArray[np.bool_], NDArray[np.float64]], NDArray[np.float64]]
    = simulate_reactive_py,
    n_sims: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Simulate every policy on the same random races (same draws, in the same order,
    as ``evaluate_strategies``, so results are directly comparable) and summarise.

    Returns:
        One row per policy sorted by mean race time: policy, mean_s, std_s, p05_s,
        p95_s, delta_to_best_s.
    """
    policies = list(policies)
    rng = np.random.default_rng(seed)
    sc_mask = draw_safety_car(rng, n_sims, params.total_laps, params.sc_hazard_per_lap, params.sc_duration_laps)
    noise = draw_lap_noise(rng, n_sims, params)
    times = np.stack([simulate(p, params, sc_mask, noise) for p in policies])
    result = pd.DataFrame(
        {
            "policy": [p.name for p in policies],
            "mean_s": times.mean(axis=1),
            "std_s": times.std(axis=1),
            "p05_s": np.percentile(times, 5, axis=1),
            "p95_s": np.percentile(times, 95, axis=1),
        }
    )
    result["delta_to_best_s"] = result["mean_s"] - result["mean_s"].min()
    return result.sort_values("mean_s").reset_index(drop=True)
