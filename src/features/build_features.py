"""
Domain feature engineering for PitWallML.

Input is a lap table as written (`src.data.ingestion.build_lap_dataset`,
read back with `src.data.ingestion.load_laps_parquet`): one row per driver-lap,
with timing, tyre and weather columns.

Leakage convention used throughout this module:
    The modelling target is `LapTime_s` of lap t. Every feature on the row of
    lap t may only use information available when lap t STARTS: completed
    laps < t, the tyre fitted, and weather sampled at the lap start. Anything
    computed from lap t's own LapTime (e.g. its fuel-corrected time) is a
    transformed target, never a feature.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from src.data.ingestion import filter_clean_laps

# Typical F1 fuel effect: ~0.06 s of lap time per lap's worth of fuel on board.
FUEL_EFFECT_S_PER_LAP: float = 0.06

# ISA sea-level air density (15 °C, 1013.25 hPa, dry), used to normalise downforce.
RHO_ISA_KG_M3: float = 1.225

# Gas constants for dry air and water vapour, J/(kg·K).
R_DRY_AIR: float = 287.058
R_WATER_VAPOUR: float = 461.495

# Rows are grouped per tyre stint for every pace/degradation feature.
STINT_KEYS: tuple[str, ...] = ("Driver", "Stint")

TARGET_COLUMN: str = "LapTime_s"

# Must never contain TARGET_COLUMN oranything derived from the current lap's own lap time.
FEATURE_COLUMNS: list[str] = [
    "LapNumber",
    "TyreLife",
    "Compound",
    "AirTemp",
    "TrackTemp",
    "TrackTempDelta_C",
    "AirDensity_kgm3",
    "DownforceIndex",
    "RollingPace_s",
    "TireDegIndex_s_per_lap",
]


def to_seconds(durations: pd.Series) -> pd.Series:
    """
    Convert a `timedelta64` Series (e.g. `LapTime`) to float seconds. NaT -> NaN.
    """
    return durations.dt.total_seconds()


def fuel_corrected_lap_time(
    lap_time_s: pd.Series,
    lap_number: pd.Series,
    total_laps: int,
    fuel_effect_s_per_lap: float = FUEL_EFFECT_S_PER_LAP,
) -> pd.Series:
    """
    Express each lap time as if the car were running on an empty tank.

    A car on lap n of total_laps still carries fuel for
    total_laps - n laps, which costs fuel_effect_s_per_lap seconds per
    lap of fuel. Removing that penalty isolates tyre degradation from fuel burn.

    This is a transformed TARGET (it uses the lap's own time), not a feature.

    Args:
        lap_time_s: Lap times in seconds (NaN allowed, propagates).
        lap_number: Lap numbers aligned with lap_time_s (nullable ints allowed).
        total_laps: Scheduled race distance in laps.
        fuel_effect_s_per_lap: Seconds of lap time per lap of fuel carried.

    Returns:
        Fuel-corrected lap times in seconds, same index as lap_time_s.
    """
    laps_remaining = total_laps - lap_number
    corrected = lap_time_s - fuel_effect_s_per_lap * laps_remaining
    return corrected.astype("float64")


def lagged_rolling_mean(
    laps: pd.DataFrame,
    value_col: str,
    window: int,
    min_periods: int = 1,
    group_cols: Sequence[str] = STINT_KEYS,
    order_col: str = "LapNumber",
) -> pd.Series:
    """
    Rolling mean of value_col over the PREVIOUS window rows of each group.

    Within each group (default: one tyre stint of one driver), rows are ordered
    by order_col. The value for a row is the mean of up to window rows
    strictly before it, so the row's own value is never included (no leakage).
    Rows with fewer than min_periods earlier rows get NaN. Windows never
    cross group boundaries.

    Example (one group, window=2, min_periods=1):
        values [90, 91, 92, 93] -> [NaN, 90.0, 90.5, 91.5]

    Args:
        laps: Lap table in any row order.
        value_col: Numeric column to average.
        window: Number of previous rows in the window.
        min_periods: Minimum previous rows required for a non-NaN result.
        group_cols: Columns defining independent groups.
        order_col: Column giving chronological order within a group.

    Returns:
        float64 Series with the SAME index (labels and order) as laps.
        The input DataFrame must not be modified.
    """
    sorted_laps = laps.sort_values(list(group_cols) + [order_col])
    g = sorted_laps.groupby(list(group_cols), observed=True, sort=False)[value_col]
    lagged = g.shift(1)
    rolled = lagged.groupby([sorted_laps[c] for c in group_cols], observed=True, sort=False).rolling(
        window, min_periods=min_periods
    ).mean()
    rolled = rolled.reset_index(level=list(group_cols), drop=True)
    return rolled.reindex(laps.index).astype("float64")

def tire_degradation_index(
    laps: pd.DataFrame,
    pace_col: str = "FuelCorrectedLapTime_s",
    tyre_age_col: str = "TyreLife",
    window: int = 5,
    min_periods: int = 3,
    group_cols: Sequence[str] = STINT_KEYS,
    order_col: str = "LapNumber",
) -> pd.Series:
    """
    Tire degradation index: rolling least-squares slope of pace vs. tyre age.

    For each row, fit pace = a + b * tyre_age on the previous window
    laps of the same stint (current lap excluded) and return b, in seconds
    lost per lap of tyre age. Rows with fewer than min_periods previous laps,
    or whose window has zero variance in tyre age, get NaN.

    Example: a stint where fuel-corrected pace is exactly 90 + 0.1 * TyreLife
    yields 0.1 for every row with at least min_periods earlier laps.

    Args:
        laps: Lap table in any row order, containing pace_col and tyre_age_col.
        pace_col: Lap pace column (seconds), normally fuel-corrected.
        tyre_age_col: Tyre age in laps.
        window: Number of previous laps in the regression window.
        min_periods: Minimum previous laps for a non-NaN slope (>= 2).
        group_cols: Columns defining one stint.
        order_col: Column giving chronological order within a stint.

    Returns:
        float64 Series with the same index as laps. Input not modified.
    """
    laps_copy = laps.copy()
    laps_copy["x"] = laps_copy[tyre_age_col].astype("float64")
    laps_copy["y"] = laps_copy[pace_col].astype("float64")
    laps_copy["xy"] = laps_copy["x"] * laps_copy["y"]
    laps_copy["xx"] = laps_copy["x"] * laps_copy["x"]
    var_x = lagged_rolling_mean(laps_copy, "xx", window, min_periods, group_cols, order_col) - lagged_rolling_mean(laps_copy, "x", window, min_periods, group_cols, order_col) ** 2
    cov_xy = lagged_rolling_mean(laps_copy, "xy", window, min_periods, group_cols, order_col) - lagged_rolling_mean(laps_copy, "x", window, min_periods, group_cols, order_col) * lagged_rolling_mean(laps_copy, "y", window, min_periods, group_cols, order_col)
    slope = cov_xy / var_x
    slope = slope.where(~np.isclose(var_x, 0), np.nan)
    return slope.reindex(laps.index).astype("float64")


def track_temp_delta(
    laps: pd.DataFrame,
    temp_col: str = "TrackTemp",
    group_cols: Sequence[str] = STINT_KEYS,
    order_col: str = "LapNumber",
) -> pd.Series:
    """
    Change in track temperature since the start of the current tyre stint.

    For each row: temp_col minus the first non-NaN temp_col of its
    stint (stint chronology given by order_col). Rows whose own temperature
    is NaN stay NaN. Uses only weather sampled at or before each lap's start,
    so it is leakage-free.

    Args:
        laps: Lap table in any row order.
        temp_col: Temperature column (°C).
        group_cols: Columns defining one stint.
        order_col: Column giving chronological order within a stint.

    Returns:
        float64 Series (°C) with the same index as laps. Input not modified.
    """
    laps_copy = laps.sort_values(list(group_cols) + [order_col])
    starting_temp = laps_copy.groupby(list(group_cols), observed=True)[temp_col].transform("first")
    return (laps_copy[temp_col] - starting_temp).reindex(laps.index).astype("float64")


def air_density(
    air_temp_c: ArrayLike,
    pressure_hpa: ArrayLike,
    humidity_pct: ArrayLike,
) -> NDArray[np.float64]:
    """
    Moist-air density (kg/m3) from weather readings, element-wise.

    Downforce and drag scale linearly with air density, so this is the
    physical driver of aero grip (e.g. Mexico City at ~2,200 m sits ~20 % below
    sea level). Computed as dry-air + water-vapour partial densities (ideal gas)

        T_K   = air_temp_c + 273.15
        p_sat = 6.1078 * 10 ** (7.5 * air_temp_c / (air_temp_c + 237.3))    [hPa, Tetens]
        p_v   = humidity_pct / 100 * p_sat                                  [hPa]
        p_d   = pressure_hpa - p_v                                          [hPa]
        rho   = (p_d * 100) / (R_DRY_AIR * T_K) + (p_v * 100) / (R_WATER_VAPOUR * T_K)

    Args:
        air_temp_c: Air temperature, °C.
        pressure_hpa: Station air pressure, hPa (= mbar, FastF1's Pressure unit).
        humidity_pct: Relative humidity, 0-100 %.

    Returns:
        float64 array broadcast from the inputs. NaN inputs propagate to NaN.
        Dry air at 15 °C / 1013.25 hPa gives ≈ RHO_ISA_KG_M3 (1.225).
    """
    air_temp_c = np.asarray(air_temp_c, dtype=np.float64)
    pressure_hpa = np.asarray(pressure_hpa, dtype=np.float64)
    humidity_pct = np.asarray(humidity_pct, dtype=np.float64)

    t_k = air_temp_c + 273.15
    p_sat = 6.1078 * 10 ** (7.5 * air_temp_c / (air_temp_c + 237.3))
    p_v = humidity_pct / 100 * p_sat
    p_d = pressure_hpa - p_v
    rho = (p_d * 100) / (R_DRY_AIR * t_k) + (p_v * 100) / (R_WATER_VAPOUR * t_k)
    return rho.astype("float64")


def micro_sector_deltas(
    distance_m: ArrayLike,
    time_s: ArrayLike,
    ref_distance_m: ArrayLike,
    ref_time_s: ArrayLike,
    n_sectors: int = 25,
) -> NDArray[np.float64]:
    """
    Time lost (+) or gained (-) in each of `n_sectors` equal-length micro-sectors.

    Both laps are telemetry traces: cumulative distance (m) and elapsed time (s)
    since the lap start. The two traces have different sample positions, 
    so both are resampled onto the same distance grid
    `np.linspace(0, L, n_sectors + 1)`, where L is the shorter of the two
    lap distances.

    Returns `lap_sector_times - ref_sector_times`; the sum equals the lap-time
    difference over `[0, L]`.

    Args:
        distance_m: Lap's cumulative distance, non-decreasing.
        time_s: Lap's elapsed time at each distance sample.
        ref_distance_m: Reference lap's cumulative distance, non-decreasing.
        ref_time_s: Reference lap's elapsed time.
        n_sectors: Number of equal-distance micro-sectors (>= 1).

    Returns:
        float64 array of shape `(n_sectors,)`.

    Raises:
        ValueError: If `n_sectors < 1`, array lengths mismatch, or a distance
            trace decreases anywhere.
    """
    if n_sectors < 1:
        raise ValueError(f"n_sectors must be >= 1, got {n_sectors}")
    d, t = np.asarray(distance_m, dtype=np.float64), np.asarray(time_s, dtype=np.float64)
    rd, rt = np.asarray(ref_distance_m, dtype=np.float64), np.asarray(ref_time_s, dtype=np.float64)
    if d.shape != t.shape or rd.shape != rt.shape:
        raise ValueError("distance and time arrays must have the same shape")
    if np.any(np.diff(d) < 0) or np.any(np.diff(rd) < 0):
        raise ValueError("distance traces must be non-decreasing")

    L = min(d[-1], rd[-1])
    edges = np.linspace(0.0, L, n_sectors + 1)
    t_at_edges = np.interp(edges, d, t)
    rt_at_edges = np.interp(edges, rd, rt)
    lap_durations = np.diff(t_at_edges)
    ref_durations = np.diff(rt_at_edges)
    return (lap_durations - ref_durations).astype("float64")


def build_features(
    laps: pd.DataFrame,
    total_laps: int,
    *,
    window: int = 5,
    min_periods: int = 3,
    green_flag_only: bool = True,
    fuel_effect_s_per_lap: float = FUEL_EFFECT_S_PER_LAP,
) -> pd.DataFrame:
    """
    Full Phase 2 pipeline for one race session's lap table.

    Steps:
        1. Weather/context features on ALL laps (so a stint's start temperature
           is taken from its real first lap, even if that lap is later dropped).
        2. Keep only clean laps (`filter_clean_laps`) and, optionally,
           green-flag laps (`TrackStatus == "1"`).
        3. Target (`LapTime_s`), its fuel-corrected version, and lagged pace
           / degradation features computed over the remaining laps.

    Args:
        laps: Lap table as returned by `src.data.ingestion.load_laps_parquet`.
        total_laps: Scheduled race distance in laps.
        window: Rolling window (laps) for pace and degradation features.
        min_periods: Minimum previous laps for the degradation index.
        green_flag_only: Drop laps run under any yellow / SC / VSC / red status.
        fuel_effect_s_per_lap: Fuel effect used for the correction.

    Returns:
        A new DataFrame with every input column plus `TARGET_COLUMN`,
        `FuelCorrectedLapTime_s` and all engineered `FEATURE_COLUMNS`.
        Sorted by `Driver` then `LapNumber` with a fresh 0..n-1 index.
        The input is not modified.
    """
    out = laps.copy()
    out["TrackTempDelta_C"] = track_temp_delta(out)
    out["AirDensity_kgm3"] = air_density(out["AirTemp"], out["Pressure"], out["Humidity"])
    out["DownforceIndex"] = out["AirDensity_kgm3"] / RHO_ISA_KG_M3

    out = filter_clean_laps(out)
    if green_flag_only:
        # TrackStatus is nullable "string": <NA> == "1" is <NA>, treat as not green.
        out = out.loc[out["TrackStatus"].eq("1").fillna(False)].reset_index(drop=True)

    out[TARGET_COLUMN] = to_seconds(out["LapTime"])
    out["FuelCorrectedLapTime_s"] = fuel_corrected_lap_time(
        out[TARGET_COLUMN], out["LapNumber"], total_laps, fuel_effect_s_per_lap
    )
    out["RollingPace_s"] = lagged_rolling_mean(out, "FuelCorrectedLapTime_s", window)
    out["TireDegIndex_s_per_lap"] = tire_degradation_index(
        out, window=window, min_periods=min_periods
    )
    return out.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
