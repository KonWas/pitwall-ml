"""Shared fixtures for the Phase 3 model tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

TRUE_DEG_S_PER_LAP: dict[str, float] = {"SOFT": 0.12, "MEDIUM": 0.07, "HARD": 0.04}


def make_model_table(
    n_races: int = 8,
    n_drivers: int = 4,
    n_stints: int = 2,
    laps_per_stint: int = 15,
    noise_s: float = 0.15,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Synthetic Phase 3 modelling table with a known signal:

        PaceDelta_s = deg(Compound) * (TyreLife - 3) + 0.03 * TrackTempDelta_C + noise

    RaceIds sort alphabetically in REVERSE calendar order ("2023_r08" is round 1),
    so any split that orders by name instead of RoundNumber fails the tests.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for r in range(n_races):
        # Race-level weather drawn independently of the round: no trend over the
        # season, so test races stay inside the training feature range.
        air_temp, track_temp = rng.normal(25, 3), rng.normal(38, 5)
        for d in range(n_drivers):
            for s in range(n_stints):
                compound = rng.choice(list(TRUE_DEG_S_PER_LAP))
                tyre_life = np.arange(1, laps_per_stint + 1, dtype=float)
                temp_delta = np.cumsum(rng.normal(0, 0.3, laps_per_stint))
                deg_index = TRUE_DEG_S_PER_LAP[compound] + rng.normal(0, 0.02, laps_per_stint)
                deg_index[:3] = np.nan  # Phase 2: needs >= 3 previous laps
                delta = (
                    TRUE_DEG_S_PER_LAP[compound] * (tyre_life - 3)
                    + 0.03 * temp_delta
                    + rng.normal(0, noise_s, laps_per_stint)
                )
                rolling = 90.0 + r + rng.normal(0, 0.1, laps_per_stint)
                frames.append(
                    pd.DataFrame(
                        {
                            "RaceId": f"2023_r{n_races - r:02d}",
                            "RoundNumber": r + 1,
                            "Driver": f"D{d:02d}",
                            "Stint": s + 1,
                            "LapNumber": s * laps_per_stint + tyre_life,
                            "Compound": compound,
                            "TyreLife": tyre_life,
                            "AirTemp": air_temp + rng.normal(0, 0.5, laps_per_stint),
                            "TrackTemp": track_temp + temp_delta,
                            "TrackTempDelta_C": temp_delta,
                            "AirDensity_kgm3": 1.17 + rng.normal(0, 0.005, laps_per_stint),
                            "TireDegIndex_s_per_lap": deg_index,
                            "RollingPace_s": rolling,
                            "FuelCorrectedLapTime_s": rolling + delta,
                            "PaceDelta_s": delta,
                        }
                    )
                )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def model_table() -> pd.DataFrame:
    return make_model_table()
