"""Tests for src.data.season."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

import src.data.season as season
from src.data.ingestion import save_laps_parquet
from src.data.season import build_season, compound_pace_summary, scan_laps


def _laps(rows: list[tuple]) -> pd.DataFrame:
    """rows: (Compound, LapTime_s | None, PitIn_s | None, PitOut_s | None, TrackStatus)."""
    compound, lap, pit_in, pit_out, status = zip(*rows)
    return pd.DataFrame(
        {
            "Driver": ["VER"] * len(rows),
            "LapNumber": [float(i + 1) for i in range(len(rows))],
            "LapTime": pd.to_timedelta(list(lap), unit="s"),
            "PitInTime": pd.to_timedelta(list(pit_in), unit="s"),
            "PitOutTime": pd.to_timedelta(list(pit_out), unit="s"),
            "Compound": list(compound),
            "TrackStatus": list(status),
        }
    )


@pytest.fixture
def season_dir(tmp_path: Path) -> Path:
    """Two race files written exactly the way the real pipeline writes them."""
    bahrain = _laps(
        [
            ("SOFT", 95.0, None, None, "1"),
            ("SOFT", 96.0, None, None, "1"),
            ("SOFT", 97.0, None, None, "1"),
            ("SOFT", 120.0, 3000, None, "1"),   # pit-in lap      -> excluded
            ("SOFT", 99.0, None, None, "14"),   # Safety Car lap  -> excluded
            ("SOFT", None, None, None, "1"),    # no lap time     -> excluded
            ("HARD", 98.0, None, None, "1"),
            ("HARD", 100.0, None, None, "1"),
        ]
    )
    saudi = _laps(
        [
            ("MEDIUM", 110.0, None, 500, "1"),  # pit-out lap     -> excluded
            ("MEDIUM", 90.0, None, None, "1"),
            ("MEDIUM", 91.0, None, None, "1"),
        ]
    )
    save_laps_parquet(bahrain, tmp_path / "2023_bahrain_R.parquet")
    save_laps_parquet(saudi, tmp_path / "2023_saudi_arabian_R.parquet")
    return tmp_path


# --- scan_laps -------------------------------------------------------------------

def test_scan_laps_is_lazy_and_tags_source(season_dir: Path) -> None:
    lf = scan_laps(season_dir)
    assert isinstance(lf, pl.LazyFrame)
    df = lf.collect()
    assert len(df) == 11
    assert "SourceFile" not in df.columns
    assert sorted(df["Event"].unique().to_list()) == ["bahrain", "saudi_arabian"]
    assert df["Year"].unique().to_list() == [2023]
    assert df["Session"].unique().to_list() == ["R"]
    assert df.schema["LapTime"] == pl.Duration("ns")


# --- build_season (no network: schedule and builder are faked) --------------------

def test_build_season_skips_existing_and_survives_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule = pd.DataFrame(
        {"RoundNumber": [1, 2, 3], "EventName": ["Bahrain Grand Prix", "Monaco Grand Prix", "Las Vegas Grand Prix"]}
    )
    monkeypatch.setattr(season, "enable_cache", lambda: None)
    monkeypatch.setattr(season.fastf1, "get_event_schedule", lambda year, include_testing: schedule)
    (tmp_path / "2023_bahrain_R.parquet").touch()  # already built
    built: list[int] = []

    def fake_build(year: int, rnd: int, session_type: str, out_dir: Path) -> Path:
        built.append(rnd)
        if rnd == 2:
            raise RuntimeError("data not available")
        return Path(out_dir) / f"round{rnd}.parquet"

    monkeypatch.setattr(season, "build_lap_dataset", fake_build)
    paths = build_season(2023, "R", tmp_path)
    assert built == [2, 3]  # round 1 skipped
    assert [p.name for p in paths] == ["2023_bahrain_R.parquet", "round3.parquet"]


# --- compound_pace_summary (student implementation) -------------------------------

def test_compound_pace_summary_values(season_dir: Path) -> None:
    out = compound_pace_summary(scan_laps(season_dir))
    assert isinstance(out, pl.DataFrame)
    assert out.rows() == [
        ("bahrain", "HARD", 2, 99.0),
        ("bahrain", "SOFT", 3, 96.0),
        ("saudi_arabian", "MEDIUM", 2, 90.5),
    ]


def test_compound_pace_summary_schema(season_dir: Path) -> None:
    out = compound_pace_summary(scan_laps(season_dir))
    assert out.schema == pl.Schema(
        {"Event": pl.String, "Compound": pl.String, "Laps": pl.UInt32, "MedianLapTime_s": pl.Float64}
    )


@pytest.mark.network
def test_compound_pace_summary_on_real_data(tmp_path: Path) -> None:
    from src.data.ingestion import build_lap_dataset

    build_lap_dataset(2023, "Bahrain", "R", out_dir=tmp_path)
    out = compound_pace_summary(scan_laps(tmp_path))
    assert set(out["Compound"]) <= {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
    assert out["Laps"].sum() > 500
    assert out["MedianLapTime_s"].is_between(90, 105).all()
