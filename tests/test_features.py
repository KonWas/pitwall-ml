"""Tests for src.features.build_features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import LAPS_DTYPES, get_laps, get_weather, load_session, merge_weather
from src.features.build_features import (
    FEATURE_COLUMNS,
    RHO_ISA_KG_M3,
    TARGET_COLUMN,
    air_density,
    build_features,
    fuel_corrected_lap_time,
    lagged_rolling_mean,
    micro_sector_deltas,
    tire_degradation_index,
    track_temp_delta,
)

TOTAL_LAPS = 12
FUEL = 0.06


def _as_parquet_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast like save_laps_parquet does, so tests see the dtypes Phase 2 really gets."""
    return df.astype({c: t for c, t in LAPS_DTYPES.items() if c in df.columns})


def _stint_frame(values: list[float], driver: str = "VER", stint: int = 1) -> pd.DataFrame:
    """One stint with LapNumber 1..n and a numeric column ``v``."""
    n = len(values)
    return pd.DataFrame(
        {"Driver": [driver] * n, "Stint": [stint] * n, "LapNumber": range(1, n + 1), "v": values}
    )


@pytest.fixture
def synthetic_race() -> pd.DataFrame:
    """
    12-lap race, rows shuffled, dtypes as loaded from Parquet.

    Fuel-corrected pace = base + deg * TyreLife exactly, so degradation slopes are known:
        VER stint 1: laps 1-7, TyreLife 3..9 (used set), deg 0.10, lap 7 = pit-in.
        VER stint 2: laps 8-12, TyreLife 1..5, deg 0.05, lap 8 = pit-out.
        HAM stint 1: laps 1-12, TyreLife 1..12, deg 0.08, lap 6 under Safety Car.
    TrackTemp rises 0.5 °C per lap from 40 °C.
    """
    rows = []
    for driver, stint, laps, tyre_life, base, deg, compound in [
        ("VER", 1, range(1, 8), range(3, 10), 90.0, 0.10, "SOFT"),
        ("VER", 2, range(8, 13), range(1, 6), 89.5, 0.05, "HARD"),
        ("HAM", 1, range(1, 13), range(1, 13), 90.5, 0.08, "MEDIUM"),
    ]:
        for lap, life in zip(laps, tyre_life):
            corrected = base + deg * life
            rows.append(
                {
                    "Driver": driver,
                    "Team": "Red Bull Racing" if driver == "VER" else "Mercedes",
                    "LapNumber": float(lap),
                    "LapTime": pd.Timedelta(seconds=corrected + FUEL * (TOTAL_LAPS - lap)),
                    "Stint": float(stint),
                    "Compound": compound,
                    "TyreLife": float(life),
                    "PitInTime": pd.Timedelta(seconds=700) if (driver, lap) == ("VER", 7) else pd.NaT,
                    "PitOutTime": pd.Timedelta(seconds=720) if (driver, lap) == ("VER", 8) else pd.NaT,
                    "TrackStatus": "4" if (driver, lap) == ("HAM", 6) else "1",
                    "AirTemp": 25.0,
                    "Humidity": 40.0,
                    "Pressure": 1010.0,
                    "Rainfall": False,
                    "TrackTemp": 40.0 + 0.5 * (lap - 1),
                }
            )
    df = pd.DataFrame(rows)
    df["PitInTime"] = pd.to_timedelta(df["PitInTime"])
    df["PitOutTime"] = pd.to_timedelta(df["PitOutTime"])
    return _as_parquet_dtypes(df).sample(frac=1.0, random_state=0)


def _row(df: pd.DataFrame, driver: str, lap: int) -> pd.Series:
    match = df.loc[(df["Driver"] == driver) & (df["LapNumber"] == lap)]
    assert len(match) == 1, f"expected exactly one row for {driver} lap {lap}"
    return match.iloc[0]


# --- fuel_corrected_lap_time -------------------------------------------------------

def test_fuel_correction_removes_fuel_for_remaining_laps() -> None:
    lap_time = pd.Series([100.0, 99.0, 98.0, np.nan], index=[10, 11, 12, 13])
    lap_number = pd.Series([1, 2, 3, 4], index=[10, 11, 12, 13], dtype="Int16")
    result = fuel_corrected_lap_time(lap_time, lap_number, total_laps=4, fuel_effect_s_per_lap=0.06)
    assert result.dtype == np.float64
    assert result.index.tolist() == [10, 11, 12, 13]
    np.testing.assert_allclose(result.to_numpy()[:3], [100.0 - 0.18, 99.0 - 0.12, 98.0 - 0.06])
    assert np.isnan(result.iloc[3])


def test_fuel_correction_last_lap_unchanged() -> None:
    result = fuel_corrected_lap_time(pd.Series([95.0]), pd.Series([57]), total_laps=57)
    assert result.iloc[0] == pytest.approx(95.0)


# --- lagged_rolling_mean -----------------------------------------------------------

def test_lagged_rolling_mean_excludes_current_row() -> None:
    result = lagged_rolling_mean(_stint_frame([90.0, 91.0, 92.0, 93.0]), "v", window=2)
    np.testing.assert_allclose(result.to_numpy(), [np.nan, 90.0, 90.5, 91.5])


def test_lagged_rolling_mean_respects_min_periods() -> None:
    result = lagged_rolling_mean(_stint_frame([90.0, 91.0, 92.0, 93.0]), "v", window=2, min_periods=2)
    np.testing.assert_allclose(result.to_numpy(), [np.nan, np.nan, 90.5, 91.5])


def test_lagged_rolling_mean_does_not_cross_groups() -> None:
    df = pd.concat(
        [
            _stint_frame([90.0, 91.0, 92.0], driver="VER", stint=1),
            _stint_frame([80.0, 81.0], driver="VER", stint=2),
            _stint_frame([70.0, 71.0], driver="HAM", stint=1),
        ],
        ignore_index=True,
    )
    result = lagged_rolling_mean(df, "v", window=3)
    np.testing.assert_allclose(
        result.to_numpy(), [np.nan, 90.0, 90.5, np.nan, 80.0, np.nan, 70.0]
    )


def test_lagged_rolling_mean_aligns_to_shuffled_input() -> None:
    df = _stint_frame([90.0, 91.0, 92.0, 93.0]).astype({"Driver": "category", "LapNumber": "Int16"})
    df.index = [100, 101, 102, 103]
    shuffled = df.loc[[102, 100, 103, 101]]
    before = shuffled.copy()
    result = lagged_rolling_mean(shuffled, "v", window=2)
    assert result.index.tolist() == [102, 100, 103, 101]
    np.testing.assert_allclose(result.to_numpy(), [90.5, np.nan, 91.5, 90.0])
    pd.testing.assert_frame_equal(shuffled, before)


def test_lagged_rolling_mean_has_no_lookahead() -> None:
    df = _stint_frame([90.0, 91.0, 92.0, 93.0, 94.0])
    base = lagged_rolling_mean(df, "v", window=3)
    df.loc[2, "v"] = 500.0  # change lap 3
    perturbed = lagged_rolling_mean(df, "v", window=3)
    # Laps 1-3 must not see lap 3's value; lap 4 must.
    np.testing.assert_allclose(perturbed.to_numpy()[:3], base.to_numpy()[:3])
    assert perturbed.iloc[3] != base.iloc[3]


# --- tire_degradation_index --------------------------------------------------------

def _linear_stint(n: int, deg: float, first_tyre_life: int = 1) -> pd.DataFrame:
    life = np.arange(first_tyre_life, first_tyre_life + n, dtype=float)
    df = _stint_frame((90.0 + deg * life).tolist())
    return df.assign(TyreLife=life).rename(columns={"v": "FuelCorrectedLapTime_s"})


def test_tire_degradation_index_recovers_linear_slope() -> None:
    result = tire_degradation_index(_linear_stint(8, deg=0.1), window=5, min_periods=3)
    assert result.iloc[:3].isna().all()
    np.testing.assert_allclose(result.iloc[3:].to_numpy(), 0.1, rtol=1e-6)


def test_tire_degradation_index_independent_of_tyre_age_offset() -> None:
    result = tire_degradation_index(_linear_stint(8, deg=0.07, first_tyre_life=15), window=4)
    np.testing.assert_allclose(result.iloc[3:].to_numpy(), 0.07, rtol=1e-6)


def test_tire_degradation_index_zero_variance_is_nan_not_inf() -> None:
    df = _linear_stint(6, deg=0.1).assign(TyreLife=5.0)
    result = tire_degradation_index(df, window=5, min_periods=3)
    assert not np.isinf(result.to_numpy()).any()
    assert result.isna().all()


def test_tire_degradation_index_has_no_lookahead() -> None:
    df = _linear_stint(8, deg=0.1)
    base = tire_degradation_index(df, window=5, min_periods=3)
    df.loc[5, "FuelCorrectedLapTime_s"] += 3.0  # lap 6 gets a big lock-up
    perturbed = tire_degradation_index(df, window=5, min_periods=3)
    np.testing.assert_allclose(perturbed.iloc[:6].to_numpy(), base.iloc[:6].to_numpy())
    assert perturbed.iloc[6] > base.iloc[6]


# --- track_temp_delta --------------------------------------------------------------

def test_track_temp_delta_relative_to_stint_start() -> None:
    df = pd.DataFrame(
        {
            "Driver": ["VER"] * 4 + ["HAM"] * 2,
            "Stint": [1, 1, 2, 2, 1, 1],
            "LapNumber": [1, 2, 3, 4, 1, 2],
            "TrackTemp": [np.nan, 41.0, 44.0, 43.0, 40.0, np.nan],
        }
    ).astype({"Driver": "category", "Stint": "Int8", "LapNumber": "Int16"})
    shuffled = df.iloc[[3, 0, 5, 2, 4, 1]]
    result = track_temp_delta(shuffled)
    assert result.index.tolist() == shuffled.index.tolist()
    # VER stint 1 starts at its first NON-NaN temp (41.0); NaN rows stay NaN.
    expected = pd.Series([np.nan, 0.0, 0.0, -1.0, 0.0, np.nan]).loc[shuffled.index]
    np.testing.assert_allclose(result.to_numpy(), expected.to_numpy())


# --- air_density -------------------------------------------------------------------

def test_air_density_dry_isa_reference() -> None:
    rho = air_density(15.0, 1013.25, 0.0)
    assert float(rho) == pytest.approx(RHO_ISA_KG_M3, abs=1e-3)


def test_air_density_physical_trends() -> None:
    base = float(air_density(25.0, 1010.0, 40.0))
    assert float(air_density(25.0, 1010.0, 90.0)) < base  # humid air is lighter
    assert float(air_density(35.0, 1010.0, 40.0)) < base  # hot air is lighter
    assert float(air_density(25.0, 780.0, 40.0)) < 0.8 * base  # Mexico City altitude


def test_air_density_is_vectorized_and_propagates_nan() -> None:
    temps = pd.Series([15.0, 25.0, np.nan])
    rho = air_density(temps, 1013.25, [0.0, 50.0, 50.0])
    assert isinstance(rho, np.ndarray)
    assert rho.shape == (3,)
    assert rho.dtype == np.float64
    assert rho[0] == pytest.approx(RHO_ISA_KG_M3, abs=1e-3)
    assert np.isnan(rho[2])


# --- micro_sector_deltas -----------------------------------------------------------

def _trace(step_m: float, slow_from_m: float | None, length_m: float = 5000.0):
    """Distance/time trace at 50 m/s, dropping to 40 m/s after ``slow_from_m``."""
    d = np.linspace(0.0, length_m, int(length_m / step_m) + 1)
    t = d / 50.0
    if slow_from_m is not None:
        t = np.where(d <= slow_from_m, t, slow_from_m / 50.0 + (d - slow_from_m) / 40.0)
    return d, t


def test_micro_sector_deltas_identical_laps_are_zero() -> None:
    d, t = _trace(10.0, None)
    rd, rt = _trace(14.0, None)  # different sampling grid, same physics
    deltas = micro_sector_deltas(d, t, rd, rt, n_sectors=20)
    assert deltas.shape == (20,)
    np.testing.assert_allclose(deltas, 0.0, atol=1e-9)


def test_micro_sector_deltas_localise_time_loss() -> None:
    d, t = _trace(10.0, slow_from_m=2500.0)
    rd, rt = _trace(10.0, None)
    deltas = micro_sector_deltas(d, t, rd, rt, n_sectors=10)
    # 500 m sectors: 10.0 s at 50 m/s vs 12.5 s at 40 m/s.
    np.testing.assert_allclose(deltas[:5], 0.0, atol=1e-9)
    np.testing.assert_allclose(deltas[5:], 2.5, atol=1e-9)
    assert deltas.sum() == pytest.approx(t[-1] - rt[-1])


def test_micro_sector_deltas_use_shorter_lap_distance() -> None:
    d, t = _trace(10.0, None, length_m=5100.0)  # lap trace runs past the line
    rd, rt = _trace(10.0, None, length_m=5000.0)
    deltas = micro_sector_deltas(d, t, rd, rt, n_sectors=10)
    np.testing.assert_allclose(deltas, 0.0, atol=1e-9)


@pytest.mark.parametrize(
    ("distance", "n_sectors"),
    [([0.0, 10.0, 5.0], 5), ([0.0, 10.0, 20.0], 0)],
    ids=["decreasing-distance", "zero-sectors"],
)
def test_micro_sector_deltas_validates_input(distance: list[float], n_sectors: int) -> None:
    with pytest.raises(ValueError):
        micro_sector_deltas(distance, [0.0, 1.0, 2.0], [0.0, 10.0, 20.0], [0.0, 1.0, 2.0], n_sectors)


# --- build_features (end-to-end) ---------------------------------------------------

def test_feature_columns_exclude_target() -> None:
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert "FuelCorrectedLapTime_s" not in FEATURE_COLUMNS
    assert "LapTime" not in FEATURE_COLUMNS


def test_build_features_output_shape(synthetic_race: pd.DataFrame) -> None:
    before = synthetic_race.copy()
    features = build_features(synthetic_race, total_laps=TOTAL_LAPS)
    pd.testing.assert_frame_equal(synthetic_race, before)
    assert set(FEATURE_COLUMNS) | {TARGET_COLUMN} <= set(features.columns)
    assert features.index.tolist() == list(range(len(features)))
    # Pit-in (VER 7), pit-out (VER 8) and Safety Car (HAM 6) laps are dropped.
    assert len(features) == 21
    assert features["PitInTime"].isna().all() and features["PitOutTime"].isna().all()
    assert features["Driver"].astype(str).tolist() == ["HAM"] * 11 + ["VER"] * 10
    assert features["LapNumber"].tolist()[11:] == [1, 2, 3, 4, 5, 6, 9, 10, 11, 12]


def test_build_features_keeps_sc_laps_when_asked(synthetic_race: pd.DataFrame) -> None:
    features = build_features(synthetic_race, total_laps=TOTAL_LAPS, green_flag_only=False)
    assert len(features) == 22


def test_build_features_values(synthetic_race: pd.DataFrame) -> None:
    f = build_features(synthetic_race, total_laps=TOTAL_LAPS, window=5, min_periods=3)
    assert _row(f, "VER", 1)[TARGET_COLUMN] == pytest.approx(90.3 + FUEL * 11)
    assert _row(f, "VER", 2)["RollingPace_s"] == pytest.approx(90.3)
    # Degradation per stint; never computed from another stint's laps.
    for lap in (1, 2, 3, 9, 10, 11):
        assert np.isnan(_row(f, "VER", lap)["TireDegIndex_s_per_lap"])
    for lap in (4, 5, 6):
        assert _row(f, "VER", lap)["TireDegIndex_s_per_lap"] == pytest.approx(0.10)
    assert _row(f, "VER", 12)["TireDegIndex_s_per_lap"] == pytest.approx(0.05)
    for lap in (4, 5, 7, 12):
        assert _row(f, "HAM", lap)["TireDegIndex_s_per_lap"] == pytest.approx(0.08)
    # Stint 2 starts on the (dropped) pit-out lap 8 at 43.5 °C.
    assert _row(f, "VER", 6)["TrackTempDelta_C"] == pytest.approx(2.5)
    assert _row(f, "VER", 12)["TrackTempDelta_C"] == pytest.approx(2.0)
    assert f["DownforceIndex"].between(0.9, 1.0).all()


def test_build_features_has_no_target_leakage(synthetic_race: pd.DataFrame) -> None:
    base = build_features(synthetic_race, total_laps=TOTAL_LAPS)
    leaked = synthetic_race.copy()
    ham_lap8 = (leaked["Driver"] == "HAM") & (leaked["LapNumber"] == 8)
    leaked.loc[ham_lap8, "LapTime"] += pd.Timedelta(seconds=10)
    perturbed = build_features(leaked, total_laps=TOTAL_LAPS)

    numeric = [c for c in FEATURE_COLUMNS if c != "Compound"]
    up_to_8 = (base["Driver"] == "HAM") & (base["LapNumber"] <= 8)
    pd.testing.assert_frame_equal(perturbed.loc[up_to_8, numeric], base.loc[up_to_8, numeric])
    # ...but later laps do see it, so the features really use past laps.
    assert _row(perturbed, "HAM", 9)["RollingPace_s"] > _row(base, "HAM", 9)["RollingPace_s"]


@pytest.mark.network
def test_build_features_on_real_data() -> None:
    session = load_session(2023, "Bahrain", "R", weather=True)
    laps = _as_parquet_dtypes(merge_weather(get_laps(session), get_weather(session)))
    f = build_features(laps, total_laps=57)
    assert len(f) > 600
    assert f["DownforceIndex"].between(0.85, 1.05).all()
    deg = f["TireDegIndex_s_per_lap"].dropna()
    assert len(deg) > 0.5 * len(f)
    assert -0.3 < deg.median() < 0.3
