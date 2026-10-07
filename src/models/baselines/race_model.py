"""
Phase 4 race model: the handful of numbers a strategy simulator needs, estimated from data.

    lap time (green) = base_lap_s + offset[compound] + deg[compound] * tyre_age
                       + fuel_effect * laps_remaining + noise
    lap time (SC)    = base_lap_s * sc_lap_factor
    pit stop         = + pit_loss_s   (x sc_pit_loss_factor when the SC is out)

Why not plug the Phase 3 CatBoost model in here? A Monte Carlo study evaluates
millions of laps, and Phase 3 showed that beyond compound and tyre age the features
explain very little. A transparent parametric model is fast, easy to port to C++,
and its parameters can be sanity-checked by eye.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.build_features import FUEL_EFFECT_S_PER_LAP
from src.models.dataset import DRY_COMPOUNDS

# Fewer clean laps than this on a compound and its fit is mostly noise.
MIN_LAPS_PER_COMPOUND: int = 30


@dataclass(frozen=True)
class TyreModel:
    """Linear tyre model: lap-time offset on a new tyre and seconds lost per lap of age."""

    offset_s: float
    deg_s_per_lap: float


@dataclass(frozen=True)
class RaceParams:
    """Everything the simulator needs about one race. Defaults are 2023-season typical values."""

    total_laps: int
    base_lap_s: float
    tyres: dict[str, TyreModel]
    pit_loss_s: float = 22.0
    lap_noise_s: float = 0.3
    sc_hazard_per_lap: float = 0.013  # 2023: 18 deployments in 1,341 race laps
    sc_duration_laps: int = 4
    sc_lap_factor: float = 1.4  # SC laps are ~40 % slower than racing laps
    sc_pit_loss_factor: float = 0.5  # the field is slow, so a stop costs ~half
    fuel_effect_s_per_lap: float = FUEL_EFFECT_S_PER_LAP
    compounds: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        # Fixed compound order: index i in the simulator's arrays = compounds[i].
        object.__setattr__(self, "compounds", tuple(c for c in DRY_COMPOUNDS if c in self.tyres))


def fit_tyre_models(
    features: pd.DataFrame,
    min_laps: int = MIN_LAPS_PER_COMPOUND,
) -> tuple[float, dict[str, TyreModel]]:
    """
    Fit one linear tyre model per dry compound from ONE race's Phase 2 feature rows.

    Drivers differ in raw pace (a Red Bull is ~1 s faster than a Williams), and a
    plain pooled fit would mix that into the compound effects. So each lap is first
    expressed relative to its own driver:

        driver_ref   = median FuelCorrectedLapTime_s of that driver in this race
        rel_time     = FuelCorrectedLapTime_s - driver_ref
        per compound: rel_time ~ offset + deg * TyreLife     (least squares)
        base_lap_s   = median of driver_ref over drivers (a "typical" car)

    Compounds with fewer than ``min_laps`` rows, and non-dry compounds, are skipped.

    Args:
        features: One race's rows of the Phase 2 feature table (columns Driver,
            Compound, TyreLife, FuelCorrectedLapTime_s).
        min_laps: Minimum rows for a compound to get a model.

    Returns:
        ``(base_lap_s, {compound: TyreModel})`` with plain Python floats.
    """
    driver_ref = features.groupby("Driver", observed=True)["FuelCorrectedLapTime_s"].transform("median")
    results: dict[str, TyreModel] = {}
    for compound in DRY_COMPOUNDS:
        rows = features["Compound"] == compound
        if rows.sum() < min_laps:
            continue
        x = features.loc[rows, "TyreLife"].astype(float)
        y = (features.loc[rows, "FuelCorrectedLapTime_s"] - driver_ref[rows]).astype(float)
        slope, intercept = np.polyfit(x, y, deg=1)
        results[compound] = TyreModel(offset_s=float(intercept), deg_s_per_lap=float(slope))
    base_lap_s = float(features.groupby("Driver", observed=True)["FuelCorrectedLapTime_s"].median().median())
    return base_lap_s, results


def estimate_pit_loss(laps: pd.DataFrame) -> float:
    """
    Median time lost to one green-flag pit stop in ONE race, from its raw Phase 1 lap table.

    A stop spans two laps: the in-lap (``PitInTime`` set, lap N) and the out-lap
    (``PitOutTime`` set, lap N + 1, same driver). For each such pair:

        loss = (in_lap + out_lap) - 2 * driver's median green lap

    where a driver's green laps have ``TrackStatus == "1"`` and neither pit time set.
    Pairs where either lap is not fully green (stops under SC/VSC are much cheaper,
    the simulator models those separately) or either LapTime is missing are ignored.

    Args:
        laps: Raw lap table as written by Phase 1 (``load_laps_parquet``).

    Returns:
        Median loss in seconds over all valid stops, or NaN if there are none.
    """
    laps = laps.assign(lap_s=laps["LapTime"].dt.total_seconds())

    green = laps["TrackStatus"].eq("1").fillna(False) & laps["PitInTime"].isna() & laps["PitOutTime"].isna()
    green_ref = laps.loc[green].groupby("Driver", observed=True)["lap_s"].median()

    in_laps = laps.loc[laps["PitInTime"].notna()]
    out_laps = laps.loc[laps["PitOutTime"].notna() & (laps["LapNumber"] > 1)].copy()
    out_laps["LapNumber"] -= 1

    pairs = pd.merge(in_laps, out_laps, on=["Driver", "LapNumber"], suffixes=("_in", "_out"))
    both_green = pairs["TrackStatus_in"].eq("1").fillna(False) & pairs["TrackStatus_out"].eq("1").fillna(False)
    pairs = pairs.loc[both_green]

    # Driver is categorical: .map() returns a Categorical, cast it back to numbers.
    driver_ref = pairs["Driver"].map(green_ref).astype(float)
    loss = (pairs["lap_s_in"] + pairs["lap_s_out"] - 2 * driver_ref).dropna()
    return float(loss.median()) if len(loss) else float("nan")


def estimate_safety_car(races: Iterable[pd.DataFrame]) -> tuple[float, float]:
    """
    Safety Car deployment rate and mean duration, pooled over several races.

    A lap counts as an SC lap if ANY row with that LapNumber has "4" in its
    TrackStatus string. A deployment is a run of consecutive SC lap numbers.
    Pooling matters: a single race has 0-4 deployments, far too few to estimate a rate.

        hazard   = total deployments / total race laps   (race laps = max LapNumber)
        duration = total SC laps / total deployments

    Args:
        races: Raw Phase 1 lap tables, one per race.

    Returns:
        ``(hazard_per_lap, mean_duration_laps)`` as floats. With no deployments at
        all, ``(0.0, nan)``.
    """
    deployments_total = 0
    sc_laps_total = 0
    race_laps_total = 0
    for laps in races:
        is_sc = laps.groupby("LapNumber", observed=True)["TrackStatus"].agg(
            lambda s: s.str.contains("4", na=False).any()
        )
        starts = is_sc & ~is_sc.shift(fill_value=False)
        deployments_total += int(starts.sum())
        sc_laps_total += int(is_sc.sum())
        race_laps_total += int(laps["LapNumber"].max())
    if deployments_total == 0:
        return 0.0, float("nan")
    return float(deployments_total / race_laps_total), float(sc_laps_total / deployments_total)


def residual_noise_s(features: pd.DataFrame, base_lap_s: float, tyres: dict[str, TyreModel]) -> float:
    """
    Robust lap-to-lap noise level left over after the tyre models.

    Uses 1.4826 * median absolute deviation of the driver-relative residuals; the
    factor makes it equal the standard deviation for normal data, while one 10 s
    traffic lap cannot inflate it the way it inflates ``std()``.
    """
    rel = features["FuelCorrectedLapTime_s"] - features.groupby("Driver", observed=True)[
        "FuelCorrectedLapTime_s"
    ].transform("median")
    rows = features["Compound"].astype(str).isin(tyres)
    comp = features.loc[rows, "Compound"].astype(str)
    fitted = comp.map({c: t.offset_s for c, t in tyres.items()}) + comp.map(
        {c: t.deg_s_per_lap for c, t in tyres.items()}
    ) * features.loc[rows, "TyreLife"].astype(float)
    resid = (rel[rows] - fitted).dropna()
    return float(1.4826 * np.median(np.abs(resid - resid.median())))


def race_params_from_data(
    features: pd.DataFrame,
    raw_laps: pd.DataFrame,
    *,
    sc_hazard_per_lap: float,
    sc_duration_laps: float,
    total_laps: int | None = None,
) -> RaceParams:
    """
    Assemble ``RaceParams`` for one race from its feature rows and raw lap table.

    The Safety Car numbers come from the whole season (``estimate_safety_car``),
    everything else from this race. A NaN pit loss falls back to the default.
    """
    base, tyres = fit_tyre_models(features)
    pit_loss = estimate_pit_loss(raw_laps)
    kwargs = {} if np.isnan(pit_loss) else {"pit_loss_s": pit_loss}
    return RaceParams(
        total_laps=total_laps or int(raw_laps["LapNumber"].max()),
        base_lap_s=base,
        tyres=tyres,
        lap_noise_s=residual_noise_s(features, base, tyres),
        sc_hazard_per_lap=sc_hazard_per_lap,
        sc_duration_laps=max(1, round(sc_duration_laps)) if not np.isnan(sc_duration_laps) else 4,
        **kwargs,
    )


def reference_race() -> RaceParams:
    """
    Hand-set, Bahrain-like race with Pirelli-style tyre behaviour.

    Race data cannot reliably identify compound pace: teams pick strategies that
    make the compounds roughly equal, and drivers manage their tyres instead of
    running at the limit (see notebook 04, section 7). So the C++ port and the RL
    agent are developed on this clean, known race model, and the data-fitted
    parameters stay as an honest measure of the sim-to-real gap.

    SOFT is fastest when new and wears fastest, HARD the opposite; the best fixed
    strategy is a MEDIUM-HARD one-stop, about 5 s ahead of the best two-stop.
    """
    return RaceParams(
        total_laps=57,
        base_lap_s=96.0,
        tyres={
            "SOFT": TyreModel(offset_s=-0.8, deg_s_per_lap=0.12),
            "MEDIUM": TyreModel(offset_s=-0.4, deg_s_per_lap=0.075),
            "HARD": TyreModel(offset_s=0.0, deg_s_per_lap=0.045),
        },
        pit_loss_s=22.5,
        lap_noise_s=0.4,
        sc_hazard_per_lap=0.0136,
        sc_duration_laps=4,
    )
