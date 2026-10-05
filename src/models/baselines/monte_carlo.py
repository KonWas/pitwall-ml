"""
Static Monte Carlo strategy evaluation (Phase 4 baseline).

A "static" strategy is fixed before the race: compounds and stint lengths, e.g.
SOFT for 18 laps, then HARD for 39. The simulator plays it through thousands of
random races (random Safety Cars, random lap-to-lap noise) and reports the
distribution of total race time.

Randomness is drawn ONCE, separately from the simulation (draw_* functions), and
passed in as arrays. That gives two things:
    * common random numbers: every strategy faces the same Safety Cars and noise,
      so differences between strategies are not drowned in sampling noise;
    * a deterministic simulate_race_times(), so the C++ port can be checked against
      it number for number.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.models.baselines.race_model import RaceParams


@dataclass(frozen=True)
class Strategy:
    """Stints in order: ``((compound, n_laps), ...)``. The car pits at the end of every stint but the last."""

    stints: tuple[tuple[str, int], ...]

    @property
    def name(self) -> str:
        """Compact label, e.g. "S18-H39"."""
        return "-".join(f"{c[0]}{n}" for c, n in self.stints)

    @property
    def n_stops(self) -> int:
        return len(self.stints) - 1

    def validate(self, total_laps: int, compounds: Sequence[str] | None = None) -> None:
        """
        Raise ValueError unless the strategy is legal for a dry race of ``total_laps``:
        stints cover the race exactly, each stint >= 1 lap, at least two different
        compounds (F1 sporting regulations), and every compound in ``compounds``.
        """
        if sum(n for _, n in self.stints) != total_laps:
            raise ValueError(f"{self.name}: stints cover {sum(n for _, n in self.stints)} laps, race has {total_laps}")
        if any(n < 1 for _, n in self.stints):
            raise ValueError(f"{self.name}: every stint needs at least one lap")
        if len({c for c, _ in self.stints}) < 2:
            raise ValueError(f"{self.name}: a dry race must use at least two compounds")
        if compounds is not None and any(c not in compounds for c, _ in self.stints):
            raise ValueError(f"{self.name}: uses a compound without a tyre model")


def strategy_arrays(
    strategy: Strategy, params: RaceParams
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.bool_]]:
    """
    Per-lap description of a strategy, length ``params.total_laps``.

    Returns:
        ``compound_idx``: index into ``params.compounds`` for each lap;
        ``tyre_age``: laps on the current set, 1 on the first lap of a stint;
        ``pit``: True on the last lap of every stint except the final one (in-laps).
    """
    strategy.validate(params.total_laps, params.compounds)
    compound_idx = np.concatenate(
        [np.full(n, params.compounds.index(c), dtype=np.int64) for c, n in strategy.stints]
    )
    tyre_age = np.concatenate([np.arange(1, n + 1, dtype=np.float64) for _, n in strategy.stints])
    pit = np.zeros(params.total_laps, dtype=bool)
    pit[np.cumsum([n for _, n in strategy.stints])[:-1] - 1] = True
    return compound_idx, tyre_age, pit


def draw_lap_noise(rng: np.random.Generator, n_sims: int, params: RaceParams) -> NDArray[np.float64]:
    """Gaussian lap-to-lap noise in seconds, shape ``(n_sims, total_laps)``."""
    return rng.normal(0.0, params.lap_noise_s, size=(n_sims, params.total_laps))


def draw_safety_car(
    rng: np.random.Generator,
    n_sims: int,
    total_laps: int,
    hazard_per_lap: float,
    duration_laps: int,
) -> NDArray[np.bool_]:
    """
    Random Safety Car periods, shape ``(n_sims, total_laps)``, True = lap under SC.

    Rules, applied lap by lap (lap 1 first), independently per simulation:
        * if no SC is active, an SC starts on this lap with probability ``hazard_per_lap``;
        * an SC that starts on lap l covers laps l .. l + duration_laps - 1, cut off at
          the end of the race;
        * a new SC can start on the first lap after the previous one ends.

    Args:
        rng: NumPy random generator (all randomness comes from it).
        n_sims: Number of simulated races.
        total_laps: Laps per race.
        hazard_per_lap: Probability of a deployment on any lap without an active SC.
        duration_laps: Laps each deployment lasts (>= 1).

    Returns:
        Boolean array, shape ``(n_sims, total_laps)``.
    """
    # CONCEPT EXPLANATION: Vectorise across SIMULATIONS, not across laps. Lap l
    # depends on lap l-1 (is an SC still running?), so the lap loop has to stay;
    # but the 10,000 simulations are independent, so each lap step updates all of
    # them at once with array operations. A loop of ~60 laps is cheap.

    # HINT:
    # 1. u = rng.random((n_sims, total_laps)): one uniform draw per lap and sim, drawn
    #    up front. "SC starts" on lap l where (no SC active) & (u[:, l] < hazard).
    # 2. Keep remaining = np.zeros(n_sims, dtype=int): laps left of the current SC.
    # 3. For each lap: start where remaining == 0 and the coin flip says so ->
    #    set remaining to duration_laps there; the lap is SC where remaining > 0;
    #    then count remaining down by 1 (np.maximum(remaining - 1, 0)).
    u = rng.random((n_sims, total_laps))
    sc_mask = np.zeros((n_sims, total_laps), dtype=bool)
    remaining = np.zeros(n_sims, dtype=int)
    for lap in range(total_laps):
        start = (remaining == 0) & (u[:, lap] < hazard_per_lap)
        remaining[start] = duration_laps
        sc_mask[:, lap] = remaining > 0
        remaining = np.maximum(remaining - 1, 0)
    return sc_mask


def simulate_race_times(
    strategy: Strategy,
    params: RaceParams,
    sc_mask: NDArray[np.bool_],
    noise: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Total race time (seconds) of ``strategy`` in each simulated race. Deterministic.

    For lap l (1-based) of L, with c / a / pit from ``strategy_arrays``:

        green lap  = base_lap_s + offset[c] + deg[c] * a
                     + fuel_effect_s_per_lap * (L - l) + noise[s, l]
        SC lap     = base_lap_s * sc_lap_factor               (no noise, no tyre effect)
        pit cost   = pit_loss_s on in-laps, times sc_pit_loss_factor if that lap is SC

    Total = sum over laps of (lap time + pit cost).

    Args:
        strategy: Strategy to simulate (validated against params).
        params: Race model.
        sc_mask: Output of ``draw_safety_car``, shape ``(n_sims, L)``.
        noise: Output of ``draw_lap_noise``, shape ``(n_sims, L)``.

    Returns:
        float64 array, shape ``(n_sims,)``.
    """
    if sc_mask.shape != noise.shape or sc_mask.shape[1] != params.total_laps:
        raise ValueError(f"sc_mask {sc_mask.shape} and noise {noise.shape} must be (n_sims, {params.total_laps})")
    compound_idx, tyre_age, pit = strategy_arrays(strategy, params)
    offsets = np.array([params.tyres[c].offset_s for c in params.compounds])
    degs = np.array([params.tyres[c].deg_s_per_lap for c in params.compounds])

    laps_remaining = params.total_laps - np.arange(1, params.total_laps + 1)
    green = (
        params.base_lap_s
        + offsets[compound_idx]
        + degs[compound_idx] * tyre_age
        + params.fuel_effect_s_per_lap * laps_remaining
    )[None, :] + noise
    lap_time = np.where(sc_mask, params.base_lap_s * params.sc_lap_factor, green)
    pit_cost = pit * params.pit_loss_s * np.where(sc_mask, params.sc_pit_loss_factor, 1.0)
    return (lap_time + pit_cost).sum(axis=1)

def enumerate_strategies(
    params: RaceParams,
    max_stops: int = 2,
    min_stint: int = 8,
    step: int = 1,
) -> list[Strategy]:
    """
    Every legal 1..max_stops-stop strategy with stints of at least ``min_stint`` laps.

    Stint lengths move in multiples of ``step`` (the last stint takes whatever is
    left), so ``step=2`` roughly halves the count per pit stop.
    """
    total = params.total_laps
    strategies: list[Strategy] = []
    for n_stints in range(2, max_stops + 2):
        for compounds in itertools.product(params.compounds, repeat=n_stints):
            if len(set(compounds)) < 2:
                continue
            for lengths in itertools.product(range(min_stint, total, step), repeat=n_stints - 1):
                last = total - sum(lengths)
                if last >= min_stint:
                    strategies.append(Strategy(tuple(zip(compounds, (*lengths, last)))))
    return strategies


def evaluate_strategies(
    strategies: Iterable[Strategy],
    params: RaceParams,
    n_sims: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Simulate every strategy on the SAME random races and summarise.

    Returns:
        One row per strategy, sorted by mean race time, with columns: strategy,
        n_stops, mean_s, std_s, p05_s, p95_s, delta_to_best_s (mean minus the best
        mean) and win_share (fraction of simulated races in which it was fastest).
    """
    strategies = list(strategies)
    rng = np.random.default_rng(seed)
    sc_mask = draw_safety_car(rng, n_sims, params.total_laps, params.sc_hazard_per_lap, params.sc_duration_laps)
    noise = draw_lap_noise(rng, n_sims, params)
    times = np.stack([simulate_race_times(s, params, sc_mask, noise) for s in strategies])  # (n_strat, n_sims)
    wins = np.bincount(times.argmin(axis=0), minlength=len(strategies)) / n_sims
    result = pd.DataFrame(
        {
            "strategy": [s.name for s in strategies],
            "n_stops": [s.n_stops for s in strategies],
            "mean_s": times.mean(axis=1),
            "std_s": times.std(axis=1),
            "p05_s": np.percentile(times, 5, axis=1),
            "p95_s": np.percentile(times, 95, axis=1),
            "win_share": wins,
        }
    )
    result.insert(6, "delta_to_best_s", result["mean_s"] - result["mean_s"].min())
    return result.sort_values("mean_s").reset_index(drop=True)
