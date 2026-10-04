"""
Phase 3 modelling table: target definition, chronological race splits, feature matrices.

Prediction task (one-step-ahead, within a stint):
    At the START of lap t, predict how much slower (+) or faster (-) lap t will be
    than the driver's recent pace on this set of tyres:

        PaceDelta_s(t) = FuelCorrectedLapTime_s(t) - RollingPace_s(t)

    RollingPace_s(t) is the mean fuel-corrected time of the previous laps of the
    same stint (Phase 2, lag-safe), so it is known before lap t starts. Predicted
    absolute lap time = RollingPace_s + predicted delta + fuel effect.

Leakage rules carried over from Phase 2: features only use information available
when lap t starts; splits are by whole race, in calendar order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.features.season_features import FEATURES_DIR

TARGET_COLUMN: str = "PaceDelta_s"

# Wet/intermediate laps mix drying-track effects (deltas of -60 s) into the target;
# they are a different prediction problem, so Phase 3 models dry running only.
DRY_COMPOUNDS: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD")

# RollingPace_s is excluded on purpose: it is an absolute, track-specific level
# (Monaco ~75 s, Spa ~108 s). The target is already relative to it, and trees
# cannot extrapolate to a level they never saw in training.
NUMERIC_FEATURES: list[str] = [
    "LapNumber",
    "TyreLife",
    "AirTemp",
    "TrackTemp",
    "TrackTempDelta_C",
    "AirDensity_kgm3",
    "TireDegIndex_s_per_lap",
]
CATEGORICAL_FEATURES: list[str] = ["Compound"]
MODEL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class RaceSplit:
    """Four disjoint, chronologically ordered subsets of the modelling table."""

    train: pd.DataFrame  # fit model parameters
    val: pd.DataFrame  # early stopping / model selection
    calib: pd.DataFrame  # conformal calibration ONLY (never seen in fitting)
    test: pd.DataFrame  # final, untouched evaluation

    def races(self) -> dict[str, list[str]]:
        """RaceIds per subset, in calendar order (handy for printing/plots)."""
        return {
            name: part.sort_values("RoundNumber")["RaceId"].unique().tolist()
            for name, part in vars(self).items()
        }


def add_pace_delta_target(table: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``TARGET_COLUMN`` and drop rows where it cannot be computed.

    ``PaceDelta_s = FuelCorrectedLapTime_s - RollingPace_s``. RollingPace_s is
    NaN on the first lap of every stint (no previous laps), so those rows have
    no target and are removed.

    Args:
        table: Phase 2 feature table (needs both source columns).

    Returns:
        A new DataFrame with the target column, only rows with a non-NaN
        target, and a fresh 0..n-1 index. The input is not modified.
    """
    table_c = table.copy()
    table_c[TARGET_COLUMN] = table_c["FuelCorrectedLapTime_s"] - table_c["RollingPace_s"]
    table_c = table_c.dropna(subset=[TARGET_COLUMN])
    return table_c.reset_index(drop=True)


def chronological_race_split(
    table: pd.DataFrame,
    n_val: int = 3,
    n_calib: int = 3,
    n_test: int = 4,
    round_col: str = "RoundNumber",
) -> RaceSplit:
    """
    Split by whole races in calendar order: train < val < calib < test.

    Races are ordered by ``round_col``. The LAST ``n_test`` races form the test
    set, the ``n_calib`` before them the calibration set, the ``n_val`` before
    those the validation set, and every earlier race the training set. Every
    row of a race lands in exactly one subset.

    Args:
        table: Modelling table with ``round_col``.
        n_val: Number of validation races.
        n_calib: Number of calibration races.
        n_test: Number of test races.
        round_col: Column giving each race's position in the calendar.

    Returns:
        A ``RaceSplit``; every part has a fresh 0..n-1 index.

    Raises:
        ValueError: If fewer than ``n_val + n_calib + n_test + 1`` races exist
            (the training set must contain at least one race).
    """
    rounds = sorted(table[round_col].unique())
    if len(rounds) < n_val + n_calib + n_test + 1:
        raise ValueError(
            f"Not enough races for split: {len(rounds)} < "
            f"{n_val + n_calib + n_test + 1}"
        )
    test_rounds = rounds[-n_test:]
    calib_rounds = rounds[-(n_test + n_calib):-n_test]
    val_rounds = rounds[-(n_test + n_calib + n_val):-(n_test + n_calib)]
    train_rounds = rounds[: -(n_test + n_calib + n_val)]
    return RaceSplit(
        train=table[table[round_col].isin(train_rounds)].reset_index(drop=True),
        val=table[table[round_col].isin(val_rounds)].reset_index(drop=True),
        calib=table[table[round_col].isin(calib_rounds)].reset_index(drop=True),
        test=table[table[round_col].isin(test_rounds)].reset_index(drop=True),
    )


def load_model_table(
    features_path: Path | str = FEATURES_DIR / "features_R.parquet",
    calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Load the Phase 2 feature table and turn it into the Phase 3 modelling table.

    Steps: keep ``DRY_COMPOUNDS`` laps, attach ``RoundNumber`` from the season
    calendar, then :func:`add_pace_delta_target`.

    Args:
        features_path: Parquet file written by ``build_season_features``.
        calendar: Output of ``season_calendar`` (one or more seasons). If None,
            it is fetched (from the FastF1 cache) for every Year in the table.

    Returns:
        Modelling table sorted by RoundNumber, Driver, LapNumber, fresh index.

    Raises:
        ValueError: If a RaceId in the table is missing from the calendar.
    """
    table = pd.read_parquet(features_path)
    table = table.loc[table["Compound"].astype(str).isin(DRY_COMPOUNDS)]
    if calendar is None:
        from src.data.season import season_calendar  # FastF1 import only when needed

        calendar = pd.concat([season_calendar(int(y)) for y in table["Year"].unique()])
    missing = set(table["RaceId"]) - set(calendar["RaceId"])
    if missing:
        raise ValueError(f"RaceIds not in calendar: {sorted(missing)}")
    table = table.merge(calendar[["RaceId", "RoundNumber"]], on="RaceId", how="left")
    table = add_pace_delta_target(table)
    return table.sort_values(["RoundNumber", "Driver", "LapNumber"]).reset_index(drop=True)


def feature_matrix(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Model inputs and target in plain dtypes every library accepts.

    Numeric features become float64 (nullable <NA> -> NaN), categorical features
    become str. CatBoost, sklearn and the MLP wrapper all take this X directly.

    Returns:
        ``(X, y)``: X has exactly ``MODEL_FEATURES`` columns, y is float64.
    """
    X = table[MODEL_FEATURES].copy()
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].astype("float64")
    X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].astype(str)
    return X, table[TARGET_COLUMN].astype("float64")
