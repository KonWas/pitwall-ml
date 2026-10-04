"""Tests for src.models.dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.dataset import (
    MODEL_FEATURES,
    TARGET_COLUMN,
    add_pace_delta_target,
    chronological_race_split,
    feature_matrix,
    load_model_table,
)


# --- add_pace_delta_target (student) ----------------------------------------------

@pytest.fixture
def small_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Driver": ["VER"] * 4,
            "LapNumber": [1, 2, 3, 4],
            "FuelCorrectedLapTime_s": [90.0, 91.0, 92.0, 93.0],
            "RollingPace_s": [np.nan, 90.0, 90.5, np.nan],  # stint start / gap -> no target
        }
    )


def test_pace_delta_values_and_dropped_rows(small_features: pd.DataFrame) -> None:
    out = add_pace_delta_target(small_features)
    assert out[TARGET_COLUMN].tolist() == [1.0, 1.5]
    assert out["LapNumber"].tolist() == [2, 3]
    assert out.index.tolist() == [0, 1]
    assert {"Driver", "FuelCorrectedLapTime_s", "RollingPace_s"} <= set(out.columns)


def test_pace_delta_does_not_mutate_input(small_features: pd.DataFrame) -> None:
    before = small_features.copy()
    add_pace_delta_target(small_features)
    pd.testing.assert_frame_equal(small_features, before)


# --- chronological_race_split (student) -------------------------------------------

def test_split_assigns_races_by_calendar_order(model_table: pd.DataFrame) -> None:
    shuffled = model_table.sample(frac=1.0, random_state=1)  # row order must not matter
    split = chronological_race_split(shuffled, n_val=2, n_calib=1, n_test=2)
    rounds = {name: sorted(part["RoundNumber"].unique()) for name, part in vars(split).items()}
    assert rounds == {"train": [1, 2, 3], "val": [4, 5], "calib": [6], "test": [7, 8]}


def test_split_has_no_temporal_overlap(model_table: pd.DataFrame) -> None:
    split = chronological_race_split(model_table, n_val=2, n_calib=2, n_test=2)
    parts = [split.train, split.val, split.calib, split.test]
    for earlier, later in zip(parts, parts[1:]):
        assert earlier["RoundNumber"].max() < later["RoundNumber"].min()
    race_sets = [set(p["RaceId"]) for p in parts]
    assert sum(len(s) for s in race_sets) == len(set().union(*race_sets))


def test_split_keeps_every_row_once_with_fresh_index(model_table: pd.DataFrame) -> None:
    split = chronological_race_split(model_table, n_val=2, n_calib=2, n_test=2)
    parts = [split.train, split.val, split.calib, split.test]
    assert sum(len(p) for p in parts) == len(model_table)
    for p in parts:
        assert p.index.tolist() == list(range(len(p)))


def test_split_requires_a_training_race(model_table: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        chronological_race_split(model_table, n_val=3, n_calib=3, n_test=2)  # 8 races, needs 9


def test_split_races_helper_lists_calendar_order(model_table: pd.DataFrame) -> None:
    split = chronological_race_split(model_table, n_val=2, n_calib=1, n_test=2)
    assert split.races()["test"] == ["2023_r02", "2023_r01"]  # rounds 7, 8


# --- load_model_table / feature_matrix ---------------------------------------------

def test_load_model_table_filters_dry_and_adds_round(tmp_path: Path, model_table: pd.DataFrame) -> None:
    features = model_table.drop(columns=["RoundNumber", TARGET_COLUMN]).assign(Year=2023)
    features.loc[features.index[:5], "Compound"] = "INTERMEDIATE"
    path = tmp_path / "features_R.parquet"
    features.to_parquet(path, index=False)
    calendar = model_table[["RaceId", "RoundNumber"]].drop_duplicates()

    table = load_model_table(path, calendar=calendar)
    assert set(table["Compound"]) <= {"SOFT", "MEDIUM", "HARD"}
    assert table["RoundNumber"].is_monotonic_increasing
    np.testing.assert_allclose(
        table[TARGET_COLUMN], table["FuelCorrectedLapTime_s"] - table["RollingPace_s"]
    )

    with pytest.raises(ValueError, match="not in calendar"):
        load_model_table(path, calendar=calendar.iloc[1:])


def test_feature_matrix_plain_dtypes(model_table: pd.DataFrame) -> None:
    table = model_table.astype({"LapNumber": "Int16", "Compound": "category"})
    table.loc[0, "LapNumber"] = pd.NA
    X, y = feature_matrix(table)
    assert list(X.columns) == MODEL_FEATURES
    assert X["LapNumber"].dtype == "float64" and np.isnan(X.loc[0, "LapNumber"])
    assert X["Compound"].map(type).eq(str).all()
    assert y.dtype == "float64"
