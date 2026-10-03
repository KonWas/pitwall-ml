"""
FastF1 data ingestion for PitWallML.

Responsibilities:
    * Initialise the FastF1 on-disk HTTP cache (so repeated runs don't re-download).
    * Load a race weekend session and expose its lap timing table.
    * Clean the lap table down to representative "racing" laps.
"""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.core import Session

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR: Path = PROJECT_ROOT / "data" / "cache"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

# Columns of FastF1's Laps table that downstream phases rely on.
LAP_COLUMNS: list[str] = [
    "Driver",
    "Team",
    "LapNumber",
    "LapStartTime",
    "LapTime",
    "Stint",
    "Compound",
    "TyreLife",
    "PitInTime",
    "PitOutTime",
    "TrackStatus",
    "IsAccurate",
]

# Target on-disk dtypes for lap datasets. Columns not listed keep their dtype
# (floats and timedeltas are already what we want). Capitalised names are
# pandas' nullable dtypes: they can hold <NA> without degrading to object/float.
LAPS_DTYPES: dict[str, str] = {
    "Driver": "category",
    "Team": "category",
    "Compound": "category",
    "TrackStatus": "string",
    "LapNumber": "Int16",
    "Stint": "Int8",
    "IsAccurate": "boolean",
    "Rainfall": "boolean",
    "WindDirection": "Int16",
}

# Weather is sampled ~once per minute; older samples than this count as missing.
WEATHER_TOLERANCE: pd.Timedelta = pd.Timedelta(minutes=2)


def enable_cache(cache_dir: Path | str = DEFAULT_CACHE_DIR) -> Path:
    """
    Create the cache directory (if needed) and enable the FastF1 HTTP cache in it.

    Args:
        cache_dir: Directory where FastF1 stores raw API responses.

    Returns:
        The resolved cache directory path.
    """
    cache_path = Path(cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))
    logger.info("FastF1 cache enabled at %s", cache_path)
    return cache_path


def load_session(
    year: int,
    event: str | int,
    session_type: str = "R",
    *,
    telemetry: bool = False,
    weather: bool = False,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> Session:
    """
    Load a FastF1 session with lap timing data.

    Args:
        year: Championship season, e.g. 2023.
        event: Grand Prix name ("Bahrain") or round number (1).
        session_type: "FP1", "FP2", "FP3", "Q", "S", or "R".
        telemetry: Also load car telemetry / position data (slow, large).
        weather: Also load weather data.
        cache_dir: FastF1 cache directory.

    Returns:
        A loaded `fastf1.core.Session`.
    """
    enable_cache(cache_dir)
    session = fastf1.get_session(year, event, session_type)
    session.load(laps=True, telemetry=telemetry, weather=weather, messages=False)
    return session


def get_laps(session: Session) -> pd.DataFrame:
    """
    Extract the lap timing table from a loaded session as a plain DataFrame.

    Args:
        session: A session already loaded via `load_session()`.

    Returns:
        One row per driver-lap, restricted to `LAP_COLUMNS`.
    """
    return pd.DataFrame(session.laps)[LAP_COLUMNS].reset_index(drop=True)


def get_weather(session: Session) -> pd.DataFrame:
    """
    Extract the weather table from a session loaded with `weather=True`.

    Args:
        session: A session already loaded via `load_session()`.

    Returns:
        One row per weather sample (~1/min). `Time` is session time
        (timedelta), the same clock as `LapStartTime` in the lap table.
    """
    return pd.DataFrame(session.weather_data).reset_index(drop=True)


def filter_clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Remove laps that don't represent true racing pace.

    A lap is dropped if any of the following hold:
        * It is a pit-in lap   (`PitInTime` is set).
        * It is a pit-out lap  (`PitOutTime` is set).
        * It has no recorded lap time (`LapTime` is NaT).

    Args:
        laps: Lap table as returned by `get_laps()`.

    Returns:
        A new DataFrame containing only clean laps, with a fresh 0..n-1 index.
        The input DataFrame must not be modified.
    """
    mask = (
        laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
        & laps["LapTime"].notna()
    )
    return laps.loc[mask].reset_index(drop=True)


def merge_weather(
    laps: pd.DataFrame,
    weather: pd.DataFrame,
    tolerance: pd.Timedelta = WEATHER_TOLERANCE,
) -> pd.DataFrame:
    """
    Attach to each lap the most recent weather sample taken at or before its start.

    Laps are matched on `LapStartTime` (laps) <-> `Time` (weather). A lap
    whose latest earlier sample is more than `tolerance` old gets NaN weather.
    Weather recorded *after* a lap started must never be used (no lookahead).

    Args:
        laps: Lap table as returned by `get_laps()` (any row order).
        weather: Weather table as returned by `get_weather()` (any row order).
        tolerance: Maximum allowed age of the matched weather sample.

    Returns:
        A new DataFrame with one row per input lap: all lap columns plus every
        weather column except `Time`. Sorted by `Driver` then `LapNumber`,
        with a fresh 0..n-1 index. Inputs must not be modified.
    """
    mask = laps["LapStartTime"].notna()
    valid_laps = laps.loc[mask]
    invalid_laps = laps.loc[~mask]
    laps_sorted = valid_laps.sort_values("LapStartTime")
    weather_sorted = weather.sort_values("Time")
    merged = pd.merge_asof(
        laps_sorted,
        weather_sorted,
        left_on="LapStartTime",
        right_on="Time",
        direction="backward",
        tolerance=tolerance,
    )
    merged = merged.drop(columns=["Time"])
    merged = pd.concat([merged, invalid_laps])
    merged = merged.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    return merged


def save_laps_parquet(laps: pd.DataFrame, path: Path | str) -> Path:
    """
    Cast a lap table to `LAPS_DTYPES` and write it to a Parquet file.

    Only the `LAPS_DTYPES` entries whose column exists in `laps` are applied,
    so this works both before and after `merge_weather()`. Parent
    directories are created if missing. The index is not stored.

    Args:
        laps: Lap table (optionally with weather columns).
        path: Destination `.parquet` file.

    Returns:
        The path written to.
    """
    dtypes_to_apply = {col: dtype for col, dtype in LAPS_DTYPES.items() if col in laps.columns}
    laps = laps.astype(dtypes_to_apply)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    laps.to_parquet(path, engine="pyarrow", index=False, compression="snappy")
    return path


def load_laps_parquet(path: Path | str) -> pd.DataFrame:
    """
    Read a lap table written by `save_laps_parquet()`.

    Args:
        path: Source `.parquet` file.

    Returns:
        The lap table with the dtypes it was saved with.
    """
    return pd.read_parquet(path, engine="pyarrow")


def dataset_path(
    year: int, event_name: str, session_type: str, out_dir: Path | str = PROCESSED_DIR
) -> Path:
    """
    File path for a session dataset: `"Saudi Arabian Grand Prix"` ->
    `<out_dir>/2023_saudi_arabian_R.parquet`. Accents are stripped (São Paulo -> sao_paulo).
    """
    ascii_name = unicodedata.normalize("NFKD", event_name).encode("ascii", "ignore").decode()
    slug = ascii_name.lower().replace(" grand prix", "").replace(" ", "_")
    return Path(out_dir) / f"{year}_{slug}_{session_type}.parquet"


def build_lap_dataset(
    year: int,
    event: str | int,
    session_type: str = "R",
    out_dir: Path | str = PROCESSED_DIR,
) -> Path:
    """
    Full Phase 1 pipeline for one session: load -> laps + weather -> merge -> Parquet.

    All laps are kept (pit laps, SC laps, lap 1); filtering is a downstream choice.

    Args:
        year: Championship season.
        event: Grand Prix name or round number.
        session_type: "FP1", "FP2", "FP3", "Q", "S", or "R".
        out_dir: Directory for the output file.

    Returns:
        Path of the written file, e.g. `data/processed/2023_bahrain_R.parquet`.
    """
    session = load_session(year, event, session_type, weather=True)
    merged = merge_weather(get_laps(session), get_weather(session))
    path = dataset_path(year, session.event["EventName"], session_type, out_dir)
    save_laps_parquet(merged, path)
    logger.info("Wrote %d laps to %s", len(merged), path)
    return path
