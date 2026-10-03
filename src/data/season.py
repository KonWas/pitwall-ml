"""
Season-level dataset building (FastF1 -> Parquet) and lazy multi-race loading (Polars).
"""

from __future__ import annotations

import logging
from pathlib import Path

import fastf1
import polars as pl

from src.data.ingestion import PROCESSED_DIR, build_lap_dataset, dataset_path, enable_cache

logger = logging.getLogger(__name__)

# Matches the basename written by dataset_path(): "<year>_<event_slug>_<session>.parquet".
_FILENAME_RE = r"(\d{4})_(.+)_([A-Z0-9]+)\.parquet$"


def build_season(
    year: int,
    session_type: str = "R",
    out_dir: Path | str = PROCESSED_DIR,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """
    Run `build_lap_dataset()` for every event of a season.

    Events whose file already exists are skipped unless `overwrite` is set.
    An event that fails (e.g. data not published, cancelled race) is logged
    and skipped, so one bad weekend doesn't abort the whole season.

    Args:
        year: Championship season.
        session_type: Session to build for each event, e.g. "R" or "Q".
        out_dir: Directory for the output files.
        overwrite: Rebuild files that already exist.

    Returns:
        Paths of all dataset files present for the season (built or skipped).
    """
    enable_cache()
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    paths: list[Path] = []
    for _, event in schedule.iterrows():
        path = dataset_path(year, event["EventName"], session_type, out_dir)
        if path.exists() and not overwrite:
            logger.info("Skipping %s (exists)", path.name)
            paths.append(path)
            continue
        try:
            paths.append(build_lap_dataset(year, event["RoundNumber"], session_type, out_dir))
        except Exception:
            logger.exception("Failed to build %s %s %s", year, event["EventName"], session_type)
    return paths


def scan_laps(data_dir: Path | str = PROCESSED_DIR) -> pl.LazyFrame:
    """
    Lazily scan every lap dataset in `data_dir` as one table.

    Nothing is read from disk here: the result is a query plan that runs on
    `.collect()`. `Year`, `Event` and `Session` columns are derived
    from each row's source filename.

    Args:
        data_dir: Directory containing files written by `dataset_path()`.

    Returns:
        A `pl.LazyFrame` over all files.
    """
    return (
        pl.scan_parquet(Path(data_dir) / "*.parquet", include_file_paths="SourceFile")
        .with_columns(
            Year=pl.col("SourceFile").str.extract(_FILENAME_RE, 1).cast(pl.Int16),
            Event=pl.col("SourceFile").str.extract(_FILENAME_RE, 2),
            Session=pl.col("SourceFile").str.extract(_FILENAME_RE, 3),
        )
        .drop("SourceFile")
    )


def compound_pace_summary(laps: pl.LazyFrame) -> pl.DataFrame:
    """
    Median green-flag racing pace per event and tyre compound.

    A lap counts only if it is a clean racing lap under green flag:
    `PitInTime` and `PitOutTime` are null, `LapTime` is not null, and
    `TrackStatus == "1"` (green for the whole lap).

    Args:
        laps: Lazy lap table as returned by `scan_laps()`.

    Returns:
        One row per (Event, Compound) with columns:
            * `Event` (String), `Compound` (String)
            * `Laps` (UInt32): number of qualifying laps
            * `MedianLapTime_s` (Float64): median lap time in seconds
        Sorted by `Event` then `Compound`.
    """
    filtered = laps.filter(
        pl.col("PitInTime").is_null(),
        pl.col("PitOutTime").is_null(),
        pl.col("LapTime").is_not_null(),
        pl.col("TrackStatus") == "1",
    )
    grouped = filtered.group_by("Event", "Compound").agg(
        Laps=pl.len(),
        MedianLapTime_s=(pl.col("LapTime").dt.total_milliseconds() / 1000).median(),
    )
    grouped = grouped.with_columns(Compound=pl.col("Compound").cast(pl.String))
    return grouped.sort("Event", "Compound").collect()
