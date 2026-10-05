"""Tests for src.models.baselines.race_model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.baselines.race_model import (
    RaceParams,
    TyreModel,
    estimate_pit_loss,
    estimate_safety_car,
    fit_tyre_models,
    race_params_from_data,
)

# Known truth for the synthetic race: rel_time = offset + deg * TyreLife.
TRUE_TYRES = {"SOFT": TyreModel(-0.6, 0.10), "HARD": TyreModel(0.4, 0.04)}


def _feature_rows(driver_pace: dict[str, float], laps_per_stint: int = 20) -> pd.DataFrame:
    """Each driver: one SOFT stint then one HARD stint; pace = driver level + tyre model."""
    rows = []
    for driver, level in driver_pace.items():
        for compound in ("SOFT", "HARD"):
            tyre = TRUE_TYRES[compound]
            for age in range(1, laps_per_stint + 1):
                rows.append(
                    {
                        "Driver": driver,
                        "Compound": compound,
                        "TyreLife": float(age),
                        "FuelCorrectedLapTime_s": level + tyre.offset_s + tyre.deg_s_per_lap * age,
                    }
                )
    return pd.DataFrame(rows).astype({"Driver": "category", "Compound": "category"})


# --- fit_tyre_models ---------------------------------------------------------------

def test_fit_tyre_models_recovers_degradation_despite_driver_pace() -> None:
    features = _feature_rows({"VER": 90.0, "SAR": 91.5, "HAM": 90.4})
    _, tyres = fit_tyre_models(features)
    assert set(tyres) == {"SOFT", "HARD"}
    for comp, truth in TRUE_TYRES.items():
        assert tyres[comp].deg_s_per_lap == pytest.approx(truth.deg_s_per_lap, abs=1e-9)
    # Absolute offsets shift with the driver reference; the SOFT-vs-HARD gap must not.
    gap = tyres["SOFT"].offset_s - tyres["HARD"].offset_s
    assert gap == pytest.approx(TRUE_TYRES["SOFT"].offset_s - TRUE_TYRES["HARD"].offset_s, abs=1e-9)


def test_fit_tyre_models_reproduces_single_driver_exactly() -> None:
    features = _feature_rows({"VER": 90.0})
    base, tyres = fit_tyre_models(features, min_laps=10)  # one driver: 20 laps per compound
    predicted = base + features["Compound"].astype(str).map(
        {c: t.offset_s for c, t in tyres.items()}
    ) + features["Compound"].astype(str).map({c: t.deg_s_per_lap for c, t in tyres.items()}) * features["TyreLife"]
    np.testing.assert_allclose(predicted, features["FuelCorrectedLapTime_s"], atol=1e-9)


def test_fit_tyre_models_base_is_median_driver_and_plain_floats() -> None:
    features = _feature_rows({"VER": 90.0, "SAR": 91.5, "HAM": 90.4})
    base, tyres = fit_tyre_models(features)
    driver_medians = features.groupby("Driver", observed=True)["FuelCorrectedLapTime_s"].median()
    assert base == pytest.approx(driver_medians.median())
    assert type(base) is float
    assert all(type(t.offset_s) is float and type(t.deg_s_per_lap) is float for t in tyres.values())


def test_fit_tyre_models_skips_thin_and_wet_compounds() -> None:
    dry = _feature_rows({"VER": 90.0}, laps_per_stint=20).astype({"Driver": str, "Compound": str})
    wet = pd.DataFrame(
        {"Driver": ["VER"] * 40, "Compound": ["INTERMEDIATE"] * 40,
         "TyreLife": np.arange(1.0, 41.0), "FuelCorrectedLapTime_s": [100.0] * 40}
    )
    features = pd.concat([dry, wet], ignore_index=True)
    _, tyres = fit_tyre_models(features, min_laps=15)
    assert set(tyres) == {"SOFT", "HARD"}
    _, tyres = fit_tyre_models(features, min_laps=25)  # 20 laps per compound < 25
    assert tyres == {}


# --- estimate_pit_loss -------------------------------------------------------------

def _raw_laps() -> pd.DataFrame:
    """
    VER: green laps of 90 s, a green stop (in-lap 100 s on lap 4, out-lap 105 s on lap 5)
         -> loss 25 s; and a stop under SC on laps 8/9 (must be ignored).
    HAM: green laps of 91 s, a green stop on laps 3/4 with in 100 s, out 102 s -> loss 20 s.
         Lap 1 has PitOutTime (pit-lane start): not a stop.
    """
    t = pd.Timedelta
    rows = []
    for lap in range(1, 11):
        status = "4" if lap in (8, 9) else "1"
        lap_time = {4: 100.0, 5: 105.0, 8: 120.0, 9: 125.0}.get(lap, 90.0)
        rows.append(("VER", lap, lap_time, t(seconds=1000) if lap in (4, 8) else pd.NaT,
                     t(seconds=1100) if lap in (5, 9) else pd.NaT, status))
        ham_time = {3: 100.0, 4: 102.0}.get(lap, 91.0)
        rows.append(("HAM", lap, ham_time, t(seconds=900) if lap == 3 else pd.NaT,
                     t(seconds=950) if lap in (1, 4) else pd.NaT, "1"))
    df = pd.DataFrame(rows, columns=["Driver", "LapNumber", "LapTime_s", "PitInTime", "PitOutTime", "TrackStatus"])
    df["LapTime"] = pd.to_timedelta(df.pop("LapTime_s"), unit="s")
    for col in ("PitInTime", "PitOutTime"):
        df[col] = df[col].astype("timedelta64[ns]")
    return df.astype({"Driver": "category", "LapNumber": "Int16", "TrackStatus": "string"}).sample(frac=1.0, random_state=1)


def test_estimate_pit_loss_median_of_green_stops() -> None:
    # Stops: VER 25 s, HAM 20 s (VER's SC stop and HAM's lap-1 pit-lane start excluded).
    assert estimate_pit_loss(_raw_laps()) == pytest.approx(22.5)


def test_estimate_pit_loss_without_green_stops_is_nan() -> None:
    laps = _raw_laps()
    laps = laps[laps["Driver"] == "HAM"].assign(TrackStatus=pd.array(["2"] * 10, dtype="string"))
    assert np.isnan(estimate_pit_loss(laps))


# --- estimate_safety_car -----------------------------------------------------------

def _race(n_laps: int, sc_laps: set[int]) -> pd.DataFrame:
    """Two drivers; SC laps flagged on ONE driver's rows only ("any row" rule), with <NA>s."""
    rows = []
    for lap in range(1, n_laps + 1):
        rows.append({"LapNumber": lap, "Driver": "VER", "TrackStatus": "14" if lap in sc_laps else "1"})
        rows.append({"LapNumber": lap, "Driver": "HAM", "TrackStatus": None})
    return pd.DataFrame(rows).astype({"LapNumber": "Int16", "TrackStatus": "string"})


def test_estimate_safety_car_pools_races() -> None:
    races = [
        _race(50, {10, 11, 12, 30, 31}),  # 2 deployments, 5 SC laps
        _race(60, set()),                 # none
        _race(40, {1, 2, 3, 4, 5, 6, 7}), # 1 deployment from lap 1, 7 SC laps
    ]
    hazard, duration = estimate_safety_car(races)
    assert hazard == pytest.approx(3 / 150)
    assert duration == pytest.approx(12 / 3)


def test_estimate_safety_car_race_ending_under_sc() -> None:
    # A deployment still running at the chequered flag never "ends" inside the race.
    hazard, duration = estimate_safety_car([_race(50, {20, 21, 48, 49, 50})])
    assert hazard == pytest.approx(2 / 50)
    assert duration == pytest.approx(5 / 2)
    assert type(hazard) is float and type(duration) is float


def test_estimate_safety_car_without_deployments() -> None:
    hazard, duration = estimate_safety_car([_race(50, set())])
    assert hazard == 0.0
    assert np.isnan(duration)


# --- race_params_from_data ---------------------------------------------------------

def test_race_params_from_data_assembles_everything() -> None:
    params = race_params_from_data(
        _feature_rows({"VER": 90.0, "HAM": 90.5}), _raw_laps(), sc_hazard_per_lap=0.02, sc_duration_laps=3.6
    )
    assert isinstance(params, RaceParams)
    assert params.total_laps == 10
    assert params.compounds == ("SOFT", "HARD")
    assert params.pit_loss_s == pytest.approx(22.5)
    assert params.sc_duration_laps == 4
    assert params.lap_noise_s == pytest.approx(0.0, abs=1e-9)  # synthetic data is noise-free
