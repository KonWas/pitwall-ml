"""Tests for src.features.season_features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import dataset_path, load_laps_parquet, save_laps_parquet
from src.features.build_features import FUEL_EFFECT_S_PER_LAP, build_features
from src.features.season_features import (
    CATEGORICAL_COLUMNS,
    build_features_for_files,
    build_season_features,
    infer_total_laps,
    parse_dataset_path,
)


def _make_race(drivers: dict[str, str], n_laps: int, compound: str, base: float) -> pd.DataFrame:
    """One-stint race per driver: clean green laps, pace = base + 0.1 * TyreLife."""
    rows = []
    for driver, team in drivers.items():
        for lap in range(1, n_laps + 1):
            rows.append(
                {
                    "Driver": driver,
                    "Team": team,
                    "LapNumber": float(lap),
                    "LapTime": pd.Timedelta(seconds=base + 0.1 * lap),
                    "Stint": 1.0,
                    "Compound": compound,
                    "TyreLife": float(lap),
                    "TrackStatus": "1",
                    "AirTemp": 25.0,
                    "Humidity": 40.0,
                    "Pressure": 1010.0,
                    "TrackTemp": 40.0,
                }
            )
    df = pd.DataFrame(rows)
    # No pit stops: all-NaT columns, typed as timedelta like FastF1's.
    no_pit = pd.Series(pd.NaT, index=df.index, dtype="timedelta64[ns]")
    return df.assign(PitInTime=no_pit, PitOutTime=no_pit)


@pytest.fixture
def race_files(tmp_path: Path) -> list[Path]:
    """Two races written exactly as Phase 1 writes them, with different drivers/compounds."""
    bahrain = _make_race({"VER": "Red Bull Racing", "HAM": "Mercedes"}, 10, "SOFT", 95.0)
    saudi = _make_race({"VER": "Red Bull Racing", "LEC": "Ferrari"}, 8, "HARD", 90.0)
    return [
        save_laps_parquet(bahrain, dataset_path(2023, "Bahrain Grand Prix", "R", tmp_path)),
        save_laps_parquet(saudi, dataset_path(2023, "Saudi Arabian Grand Prix", "R", tmp_path)),
    ]


# --- parse_dataset_path -------------------------------------------------------------

def test_parse_dataset_path() -> None:
    assert parse_dataset_path("data/processed/2023_saudi_arabian_R.parquet") == (2023, "saudi_arabian", "R")
    with pytest.raises(ValueError):
        parse_dataset_path("notes.parquet")


# --- infer_total_laps ---------------------------------------------------------------

def test_infer_total_laps_returns_plain_int() -> None:
    laps = pd.DataFrame({"LapNumber": pd.array([3, 57, None, 12], dtype="Int16")})
    result = infer_total_laps(laps)
    assert result == 57
    assert type(result) is int


@pytest.mark.parametrize(
    "lap_numbers",
    [pd.array([None, None], dtype="Int16"), pd.array([], dtype="Int16")],
    ids=["all-missing", "empty"],
)
def test_infer_total_laps_rejects_tables_without_laps(lap_numbers) -> None:
    with pytest.raises(ValueError):
        infer_total_laps(pd.DataFrame({"LapNumber": lap_numbers}))


# --- build_features_for_files -------------------------------------------------------

def test_stacks_races_with_race_columns(race_files: list[Path]) -> None:
    f = build_features_for_files(race_files)
    assert len(f) == 2 * 9 + 2 * 7  # lap 1 (standing start) dropped for every driver
    assert f["RaceId"].unique().tolist() == ["2023_bahrain", "2023_saudi_arabian"]
    assert set(f["Year"]) == {2023}
    assert set(f["Event"]) == {"bahrain", "saudi_arabian"}
    assert set(f["Session"]) == {"R"}
    assert f.index.tolist() == list(range(len(f)))


def test_sorted_by_race_driver_lap(race_files: list[Path]) -> None:
    f = build_features_for_files(race_files[::-1])  # input order must not matter
    expected = f.sort_values(["RaceId", "Driver", "LapNumber"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(f, expected)
    assert f["Driver"].astype(str).tolist()[:10] == ["HAM"] * 9 + ["VER"]


def test_categorical_columns_survive_concat(race_files: list[Path]) -> None:
    f = build_features_for_files(race_files)
    for col in CATEGORICAL_COLUMNS:
        assert isinstance(f[col].dtype, pd.CategoricalDtype), f"{col} is {f[col].dtype}"
    assert set(f["Driver"].cat.categories) == {"VER", "HAM", "LEC"}
    assert set(f["Compound"].cat.categories) == {"SOFT", "HARD"}


def test_matches_single_race_build_features(race_files: list[Path]) -> None:
    f = build_features_for_files(race_files)
    single = build_features(load_laps_parquet(race_files[1]), total_laps=8)
    stacked = f[f["RaceId"] == "2023_saudi_arabian"].reset_index(drop=True)
    for col in ["LapTime_s", "FuelCorrectedLapTime_s", "RollingPace_s", "TireDegIndex_s_per_lap"]:
        np.testing.assert_allclose(stacked[col].to_numpy(), single[col].to_numpy())


def test_no_leakage_across_races(race_files: list[Path]) -> None:
    f = build_features_for_files(race_files)
    ver_saudi = f[(f["RaceId"] == "2023_saudi_arabian") & (f["Driver"] == "VER")]
    # VER's first Saudi lap must not see his Bahrain laps (same Driver + Stint keys).
    assert np.isnan(ver_saudi["RollingPace_s"].iloc[0])
    assert ver_saudi["RollingPace_s"].iloc[1] == pytest.approx(ver_saudi["FuelCorrectedLapTime_s"].iloc[0])


def test_total_laps_override(race_files: list[Path]) -> None:
    default = build_features_for_files(race_files)
    overridden = build_features_for_files(race_files, total_laps={"2023_bahrain": 20})
    bahrain_lap2 = (overridden["RaceId"] == "2023_bahrain") & (overridden["LapNumber"] == 2)
    expected = 95.2 - FUEL_EFFECT_S_PER_LAP * (20 - 2)
    assert overridden.loc[bahrain_lap2, "FuelCorrectedLapTime_s"].tolist() == pytest.approx([expected] * 2)
    # The other race keeps its inferred distance.
    saudi = default["RaceId"] == "2023_saudi_arabian"
    np.testing.assert_allclose(
        overridden.loc[saudi, "FuelCorrectedLapTime_s"], default.loc[saudi, "FuelCorrectedLapTime_s"]
    )


def test_passes_window_through(race_files: list[Path]) -> None:
    f3 = build_features_for_files(race_files, window=3)
    f5 = build_features_for_files(race_files, window=5)
    assert not np.allclose(f3["RollingPace_s"], f5["RollingPace_s"], equal_nan=True)


# --- build_season_features ----------------------------------------------------------

def test_build_season_features_writes_parquet(race_files: list[Path], tmp_path: Path) -> None:
    out = build_season_features(tmp_path, tmp_path / "out" / "features.parquet")
    assert out.is_file()
    loaded = pd.read_parquet(out)
    assert len(loaded) == 32
    assert isinstance(loaded["Driver"].dtype, pd.CategoricalDtype)


def test_build_season_features_filters_session_type(race_files: list[Path], tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_season_features(tmp_path, tmp_path / "q.parquet", session_type="Q")
