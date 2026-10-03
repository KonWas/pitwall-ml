"""
Season-level feature dataset: run build_features on every race file and stack the results.

Features are always computed race by race, BEFORE stacking. Every rolling feature
is grouped by (Driver, Stint), and those keys repeat across races (VER stint 1 exists
in every race), so computing them on the stacked table would let one race's laps
leak into the next race's windows.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from src.data.ingestion import PROCESSED_DIR, PROJECT_ROOT, load_laps_parquet
from src.data.season import _FILENAME_RE
from src.features.build_features import build_features

logger = logging.getLogger(__name__)

FEATURES_DIR: Path = PROJECT_ROOT / "data" / "features"

# Columns that must come out of the stack as `category`, whatever races were combined.
CATEGORICAL_COLUMNS: list[str] = ["Driver", "Team", "Compound"]


def parse_dataset_path(path: Path | str) -> tuple[int, str, str]:
    """
    Split a Phase 1 file name into its parts:
    "2023_saudi_arabian_R.parquet" -> (2023, "saudi_arabian", "R").

    Raises:
        ValueError: If the name doesn't follow the dataset_path() pattern.
    """
    match = re.search(_FILENAME_RE, Path(path).name)
    if match is None:
        raise ValueError(f"not a lap dataset file name: {Path(path).name}")
    return int(match.group(1)), match.group(2), match.group(3)


def infer_total_laps(laps: pd.DataFrame) -> int:
    """
    Race distance in laps, taken from a race's lap table.

    The winner completes the full scheduled distance, so the highest LapNumber in
    the table equals the scheduled race length. That number is public before the
    race, so using it in the fuel correction is not leakage. Exception: races
    shortened by a red flag or rain (e.g. 2021 Belgian GP) end early; pass the
    real distance via the total_laps override in build_features_for_files.

    Args:
        laps: One race's lap table (LapNumber may be nullable Int16 with <NA>).

    Returns:
        The race distance as a plain Python int.

    Raises:
        ValueError: If the table has no valid LapNumber at all.
    """
    max_lap = laps["LapNumber"].max()
    if pd.isna(max_lap):
        raise ValueError("No valid LapNumber found in the table.")
    return int(max_lap)


def build_features_for_files(
    paths: Sequence[Path | str],
    *,
    total_laps: Mapping[str, int] | None = None,
    window: int = 5,
    min_periods: int = 3,
    green_flag_only: bool = True,
) -> pd.DataFrame:
    """
    Build features for each race file separately, then stack them into one table.

    For each file:
        1. Load it with load_laps_parquet.
        2. Work out RaceId = "<year>_<event>" (e.g. "2023_bahrain") from the name.
        3. Race distance = total_laps[RaceId] if given, else infer_total_laps().
        4. Run build_features with window / min_periods / green_flag_only.
        5. Add columns Year (int), Event (str), Session (str) and RaceId (str).

    Args:
        paths: Lap dataset files written by Phase 1.
        total_laps: Optional race distance overrides, keyed by RaceId.
        window: Passed to build_features.
        min_periods: Passed to build_features.
        green_flag_only: Passed to build_features.

    Returns:
        One DataFrame with all races, sorted by RaceId, Driver, LapNumber, with a
        fresh 0..n-1 index. Every CATEGORICAL_COLUMNS column is `category` dtype,
        with categories covering all races.
    """
    total_laps = total_laps or {}

    frames = []
    for p in paths:
        year, event, session = parse_dataset_path(p)
        laps = load_laps_parquet(p)
        race_id = f"{year}_{event}"
        if race_id in total_laps:
            race_total_laps = total_laps[race_id]
        else:
            race_total_laps = infer_total_laps(laps)
        features = build_features(
            laps,
            total_laps=race_total_laps,
            window=window,
            min_periods=min_periods,
            green_flag_only=green_flag_only,
        ).assign(Year=year, Event=event, Session=session, RaceId=race_id)
        frames.append(features)
    result = pd.concat(frames, ignore_index=True)
    for col in CATEGORICAL_COLUMNS:
        result[col] = result[col].astype("category")
    result = result.sort_values(["RaceId", "Driver", "LapNumber"]).reset_index(drop=True)
    return result


def build_season_features(
    data_dir: Path | str = PROCESSED_DIR,
    out_path: Path | str = FEATURES_DIR / "features_R.parquet",
    *,
    session_type: str = "R",
    total_laps: Mapping[str, int] | None = None,
    window: int = 5,
    min_periods: int = 3,
    green_flag_only: bool = True,
) -> Path:
    """
    Build features for every `session_type` file in data_dir and write one Parquet file.

    Only race sessions make sense for the fuel correction, hence the "R" default.

    Args:
        data_dir: Directory with Phase 1 lap datasets.
        out_path: Destination Parquet file (parent directories are created).
        session_type: Which session files to include.
        total_laps: Optional race distance overrides, keyed by RaceId.
        window: Passed to build_features.
        min_periods: Passed to build_features.
        green_flag_only: Passed to build_features.

    Returns:
        The path written to.

    Raises:
        FileNotFoundError: If data_dir has no files for session_type.
    """
    paths = sorted(Path(data_dir).glob(f"*_{session_type}.parquet"))
    if not paths:
        raise FileNotFoundError(f"no *_{session_type}.parquet files in {data_dir}")
    features = build_features_for_files(
        paths,
        total_laps=total_laps,
        window=window,
        min_periods=min_periods,
        green_flag_only=green_flag_only,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, engine="pyarrow", index=False)
    logger.info("Wrote %d feature rows from %d races to %s", len(features), len(paths), out_path)
    return out_path
