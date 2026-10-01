"""Tests for src.data.ingestion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.ingestion import (
    LAP_COLUMNS,
    LAPS_DTYPES,
    build_lap_dataset,
    enable_cache,
    filter_clean_laps,
    get_laps,
    get_weather,
    load_laps_parquet,
    load_session,
    merge_weather,
    save_laps_parquet,
)

td = pd.Timedelta


@pytest.fixture(scope="module")
def bahrain_2023_session():
    """2023 Bahrain GP race with laps + weather (cached after first run)."""
    return load_session(2023, "Bahrain", "R", weather=True)


@pytest.fixture(scope="module")
def bahrain_2023_laps(bahrain_2023_session) -> pd.DataFrame:
    return get_laps(bahrain_2023_session)


@pytest.fixture(scope="module")
def bahrain_2023_weather(bahrain_2023_session) -> pd.DataFrame:
    return get_weather(bahrain_2023_session)


@pytest.fixture
def synthetic_laps() -> pd.DataFrame:
    """Small hand-built lap table covering every filtering case."""
    # Times in seconds; None -> NaT. Same timedelta64[ns] dtype FastF1 uses.
    # Lap 3 = pit-in lap, lap 4 = pit-out lap, HAM lap 1 = no lap time recorded.
    return pd.DataFrame(
        {
            "Driver": ["VER", "VER", "VER", "VER", "VER", "HAM"],
            "LapNumber": [1.0, 2.0, 3.0, 4.0, 5.0, 1.0],
            "LapTime": pd.to_timedelta([97.1, 96.4, 118.0, 119.5, 95.9, None], unit="s"),
            "PitInTime": pd.to_timedelta([None, None, 300, None, None, None], unit="s"),
            "PitOutTime": pd.to_timedelta([None, None, None, 360, None, None], unit="s"),
        }
    )


# --- Infrastructure -----------------------------------------------------------

def test_enable_cache_creates_directory(tmp_path: Path) -> None:
    cache_dir = tmp_path / "nested" / "cache"
    result = enable_cache(cache_dir)
    assert result == cache_dir.resolve()
    assert cache_dir.is_dir()


# --- Lap timing fetch (network on first run, cache afterwards) ------------------

@pytest.mark.network
def test_fetch_lap_timing_data(bahrain_2023_laps: pd.DataFrame) -> None:
    laps = bahrain_2023_laps
    assert list(laps.columns) == LAP_COLUMNS
    assert len(laps) > 900  # 20 drivers x 57 laps, minus DNFs
    assert laps["Driver"].nunique() == 20
    assert pd.api.types.is_timedelta64_dtype(laps["LapTime"])
    assert laps["LapTime"].notna().sum() > 0.9 * len(laps)
    # Verstappen won from pole; his fastest lap should be in the 1:3x range.
    ver_best = laps.loc[laps["Driver"] == "VER", "LapTime"].min()
    assert td(seconds=90) < ver_best < td(seconds=100)


# --- filter_clean_laps (student implementation) ----------------------------------

def test_filter_clean_laps_removes_pit_and_missing_laps(synthetic_laps: pd.DataFrame) -> None:
    clean = filter_clean_laps(synthetic_laps)
    assert isinstance(clean, pd.DataFrame)
    assert clean["LapNumber"].tolist() == [1.0, 2.0, 5.0]
    assert clean["Driver"].tolist() == ["VER", "VER", "VER"]


def test_filter_clean_laps_resets_index(synthetic_laps: pd.DataFrame) -> None:
    clean = filter_clean_laps(synthetic_laps)
    assert clean.index.tolist() == list(range(len(clean)))


def test_filter_clean_laps_does_not_mutate_input(synthetic_laps: pd.DataFrame) -> None:
    before = synthetic_laps.copy()
    filter_clean_laps(synthetic_laps)
    pd.testing.assert_frame_equal(synthetic_laps, before)


@pytest.mark.network
def test_filter_clean_laps_on_real_data(bahrain_2023_laps: pd.DataFrame) -> None:
    clean = filter_clean_laps(bahrain_2023_laps)
    assert clean["PitInTime"].isna().all()
    assert clean["PitOutTime"].isna().all()
    assert clean["LapTime"].notna().all()
    assert 0 < len(clean) < len(bahrain_2023_laps)


# --- merge_weather (student implementation) --------------------------------------

@pytest.fixture
def synthetic_weather() -> pd.DataFrame:
    """Samples at t = 0, 60, 120, 600 s (gap 120 -> 600 s simulates a sensor dropout)."""
    return pd.DataFrame(
        {
            "Time": pd.to_timedelta([0, 60, 120, 600], unit="s"),
            "AirTemp": [30.0, 31.0, 32.0, 33.0],
            "TrackTemp": [40.0, 41.0, 42.0, 43.0],
        }
    )


@pytest.fixture
def synthetic_laps_for_weather() -> pd.DataFrame:
    """Deliberately NOT sorted by LapStartTime or by Driver/LapNumber."""
    return pd.DataFrame(
        {
            "Driver": ["VER", "HAM", "VER", "HAM"],
            "LapNumber": [2.0, 2.0, 1.0, 1.0],
            #  VER L2 = exactly on a sample; HAM L2 = 280 s after last sample;
            #  VER L1 = between samples;     HAM L1 = 1 s before the 60 s sample.
            "LapStartTime": pd.to_timedelta([120, 400, 30, 59], unit="s"),
        }
    )


def test_merge_weather_uses_latest_past_sample(
    synthetic_laps_for_weather: pd.DataFrame, synthetic_weather: pd.DataFrame
) -> None:
    merged = merge_weather(synthetic_laps_for_weather, synthetic_weather)
    assert merged["Driver"].tolist() == ["HAM", "HAM", "VER", "VER"]
    assert merged["LapNumber"].tolist() == [1.0, 2.0, 1.0, 2.0]
    # HAM L1 at 59 s must get the 0 s sample, NOT the 60 s one (lookahead).
    # HAM L2 at 400 s: last sample is 280 s old > tolerance -> NaN.
    assert merged["AirTemp"].tolist()[0] == 30.0
    assert pd.isna(merged["AirTemp"].tolist()[1])
    assert merged["AirTemp"].tolist()[2:] == [30.0, 32.0]
    assert merged["TrackTemp"].tolist()[2:] == [40.0, 42.0]


def test_merge_weather_columns_and_index(
    synthetic_laps_for_weather: pd.DataFrame, synthetic_weather: pd.DataFrame
) -> None:
    merged = merge_weather(synthetic_laps_for_weather, synthetic_weather)
    assert list(merged.columns) == ["Driver", "LapNumber", "LapStartTime", "AirTemp", "TrackTemp"]
    assert merged.index.tolist() == [0, 1, 2, 3]


def test_merge_weather_wider_tolerance_fills_gap(
    synthetic_laps_for_weather: pd.DataFrame, synthetic_weather: pd.DataFrame
) -> None:
    merged = merge_weather(synthetic_laps_for_weather, synthetic_weather, tolerance=td(minutes=10))
    assert merged["AirTemp"].tolist() == [30.0, 32.0, 30.0, 32.0]


def test_merge_weather_does_not_mutate_inputs(
    synthetic_laps_for_weather: pd.DataFrame, synthetic_weather: pd.DataFrame
) -> None:
    laps_before = synthetic_laps_for_weather.copy()
    weather_before = synthetic_weather.copy()
    merge_weather(synthetic_laps_for_weather, synthetic_weather)
    pd.testing.assert_frame_equal(synthetic_laps_for_weather, laps_before)
    pd.testing.assert_frame_equal(synthetic_weather, weather_before)


@pytest.mark.network
def test_merge_weather_on_real_data(
    bahrain_2023_laps: pd.DataFrame, bahrain_2023_weather: pd.DataFrame
) -> None:
    merged = merge_weather(bahrain_2023_laps, bahrain_2023_weather)
    assert len(merged) == len(bahrain_2023_laps)
    assert "Time" not in merged.columns
    # Samples arrive every ~60 s for the whole race, so every lap gets weather.
    assert merged["TrackTemp"].notna().all()
    assert merged["TrackTemp"].between(15, 60).all()


def test_merge_weather_keeps_laps_with_missing_start_time(
    synthetic_laps_for_weather: pd.DataFrame, synthetic_weather: pd.DataFrame
) -> None:
    glitch = pd.DataFrame(
        {"Driver": ["ALO"], "LapNumber": [1.0], "LapStartTime": pd.to_timedelta([None], unit="s")}
    )
    laps = pd.concat([synthetic_laps_for_weather, glitch], ignore_index=True)
    merged = merge_weather(laps, synthetic_weather)
    # One row per input lap; the NaT-start lap is kept with NaN weather.
    assert len(merged) == 5
    assert merged["Driver"].tolist() == ["ALO", "HAM", "HAM", "VER", "VER"]
    assert list(merged.columns) == ["Driver", "LapNumber", "LapStartTime", "AirTemp", "TrackTemp"]
    assert merged.loc[0, ["AirTemp", "TrackTemp"]].isna().all()
    assert merged["AirTemp"].tolist()[3:] == [30.0, 32.0]
    assert merged.index.tolist() == [0, 1, 2, 3, 4]


# --- Parquet persistence (student implementation) ---------------------------------

@pytest.fixture
def synthetic_merged_laps() -> pd.DataFrame:
    """Merged-lap shape with the dtypes merge_weather really produces (object/float)."""
    return pd.DataFrame(
        {
            "Driver": ["VER", "VER", "HAM"],
            "LapNumber": [1.0, 2.0, 1.0],
            "LapTime": pd.to_timedelta([97.1, 96.4, None], unit="s"),
            "Compound": ["SOFT", "SOFT", "HARD"],
            "TrackStatus": ["1", "14", "1"],
            "Rainfall": pd.Series([False, True, None], dtype=object),
            "WindDirection": [176.0, 182.0, None],
            "TrackTemp": [40.0, 41.5, None],
        }
    )


def test_save_laps_parquet_round_trip(tmp_path: Path, synthetic_merged_laps: pd.DataFrame) -> None:
    path = save_laps_parquet(synthetic_merged_laps, tmp_path / "out" / "laps.parquet")
    assert Path(path).is_file()
    loaded = load_laps_parquet(path)
    assert str(loaded["Driver"].dtype) == "category"
    assert str(loaded["Compound"].dtype) == "category"
    assert str(loaded["TrackStatus"].dtype) == "string"
    assert str(loaded["LapNumber"].dtype) == "Int16"
    assert str(loaded["Rainfall"].dtype) == "boolean"
    assert str(loaded["WindDirection"].dtype) == "Int16"
    assert str(loaded["LapTime"].dtype) == "timedelta64[ns]"
    assert loaded["Rainfall"].tolist()[:2] == [False, True]
    assert loaded["Rainfall"].isna().tolist() == [False, False, True]
    assert loaded["LapTime"].tolist()[0] == td(seconds=97.1)
    assert loaded.index.tolist() == [0, 1, 2]


def test_save_laps_parquet_does_not_mutate_input(
    tmp_path: Path, synthetic_merged_laps: pd.DataFrame
) -> None:
    before = synthetic_merged_laps.copy()
    save_laps_parquet(synthetic_merged_laps, tmp_path / "laps.parquet")
    pd.testing.assert_frame_equal(synthetic_merged_laps, before)


@pytest.mark.network
def test_save_laps_parquet_on_real_data(
    tmp_path: Path, bahrain_2023_laps: pd.DataFrame, bahrain_2023_weather: pd.DataFrame
) -> None:
    merged = merge_weather(bahrain_2023_laps, bahrain_2023_weather)
    path = save_laps_parquet(merged, tmp_path / "bahrain_2023_R.parquet")
    loaded = load_laps_parquet(path)
    expected = merged.astype({c: t for c, t in LAPS_DTYPES.items() if c in merged.columns})
    pd.testing.assert_frame_equal(loaded, expected)


# --- End-to-end pipeline ----------------------------------------------------------

@pytest.mark.network
def test_build_lap_dataset(tmp_path: Path) -> None:
    path = build_lap_dataset(2023, "Bahrain", "R", out_dir=tmp_path)
    assert path == tmp_path / "2023_bahrain_R.parquet"
    loaded = load_laps_parquet(path)
    assert len(loaded) > 900
    assert {"LapTime", "Compound", "TrackTemp", "Rainfall"} <= set(loaded.columns)
