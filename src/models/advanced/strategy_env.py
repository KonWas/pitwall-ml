"""
Phase 4d: pit-stop strategy as a reinforcement-learning problem.

One episode = one race, one step = one lap. Before lap l is finished the agent sees
the race state (lap, tyre age, compound, whether the Safety Car is out, stops so
far) and decides whether lap l is an in-lap, and onto which compound:

    action 0      stay out
    action k >= 1 pit, fit a new set of params.compounds[k - 1]

The lap model is exactly the one used by the Monte Carlo simulators (fixed,
reactive, C++), so an agent's race times are directly comparable with theirs on
the same random races.

`RaceState` holds the dynamics for N races at once (NumPy arrays of length N).
The Gymnasium env uses N = 1 for training; `rollout` uses N = thousands to
evaluate any policy (scripted or learned) on the same races as the baselines.
"""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from src.models.baselines.monte_carlo import Strategy, draw_lap_noise, draw_safety_car
from src.models.baselines.race_model import RaceParams, reference_race
from src.models.baselines.reactive import ReactivePolicy

# Finishing a dry race on one compound is a disqualification in F1. A large time
# penalty teaches the agent the rule without hard-coding it into the action space.
DSQ_PENALTY_S: float = 100.0


class RaceState:
    """
    State and dynamics of N simulated races, advanced one lap at a time.

    Attributes (arrays of length N unless noted):
        lap: next lap to be driven, 1-based (int, shared by all races).
        compound: index into params.compounds of the tyres on the car.
        age: laps on the current set, 1 on the first lap of a stint.
        stops: pit stops made so far.
        used: (N, n_compounds) which compounds have been raced.
        total: race time so far, seconds (penalty included at the end).
    """

    def __init__(
        self,
        params: RaceParams,
        sc_mask: NDArray[np.bool_],
        noise: NDArray[np.float64],
        start_compound: str = "MEDIUM",
    ) -> None:
        if sc_mask.shape != noise.shape or sc_mask.ndim != 2 or sc_mask.shape[1] != params.total_laps:
            raise ValueError(f"sc_mask {sc_mask.shape} and noise {noise.shape} must be (n, {params.total_laps})")
        self.params = params
        self.sc_mask, self.noise = sc_mask, noise
        n, k = sc_mask.shape[0], len(params.compounds)
        self.offsets = np.array([params.tyres[c].offset_s for c in params.compounds])
        self.degs = np.array([params.tyres[c].deg_s_per_lap for c in params.compounds])
        start = params.compounds.index(start_compound)
        self.lap = 1
        self.compound = np.full(n, start, dtype=np.int64)
        self.age = np.ones(n, dtype=np.int64)
        self.stops = np.zeros(n, dtype=np.int64)
        self.used = np.zeros((n, k), dtype=bool)
        self.used[:, start] = True
        self.total = np.zeros(n)

    @property
    def n(self) -> int:
        return len(self.compound)

    @property
    def done(self) -> bool:
        return self.lap > self.params.total_laps

    @property
    def sc_now(self) -> NDArray[np.bool_]:
        """Is the Safety Car out on the lap about to be decided (all False once the race is over)?"""
        if self.done:
            return np.zeros(self.n, dtype=bool)
        return self.sc_mask[:, self.lap - 1]

    def observe(self) -> NDArray[np.float32]:
        """
        Observation per race, shape (N, 4 + n_compounds + 2), every entry in [0, 1]:

            [lap / L, laps remaining / L, tyre age / L, one-hot compound...,
             SC out, min(stops, 3) / 3, two compounds used]
        """
        L = self.params.total_laps
        lap = min(self.lap, L)
        k = len(self.params.compounds)
        return np.column_stack(
            [
                np.full(self.n, lap / L),
                np.full(self.n, (L - lap) / L),
                np.minimum(self.age, L) / L,
                np.eye(k)[self.compound],
                self.sc_now,
                np.minimum(self.stops, 3) / 3,
                self.used.sum(axis=1) >= 2,
            ]
        ).astype(np.float32)

    def step(self, actions: NDArray[np.int64]) -> NDArray[np.float64]:
        """
        Drive the current lap in every race, apply the pit decisions, return the
        seconds spent (lap time + pit cost, + DSQ penalty after the final lap).
        """
        if self.done:
            raise RuntimeError("race is over; create a new RaceState")
        p, L, l = self.params, self.params.total_laps, self.lap
        # A scalar means the same action in every race.
        actions = np.broadcast_to(np.asarray(actions, dtype=np.int64), (self.n,))
        sc = self.sc_mask[:, l - 1]
        green = (
            p.base_lap_s + self.offsets[self.compound] + self.degs[self.compound] * self.age
            + p.fuel_effect_s_per_lap * (L - l) + self.noise[:, l - 1]
        )
        seconds = np.where(sc, p.base_lap_s * p.sc_lap_factor, green)
        pit = actions > 0
        seconds = seconds + np.where(pit, p.pit_loss_s * np.where(sc, p.sc_pit_loss_factor, 1.0), 0.0)

        new = actions - 1
        self.compound = np.where(pit, new, self.compound)
        self.age = np.where(pit, 1, self.age + 1)
        self.stops = self.stops + pit
        self.used[np.flatnonzero(pit), new[pit]] = True
        self.lap += 1
        if self.done:
            seconds = seconds + np.where(self.used.sum(axis=1) < 2, DSQ_PENALTY_S, 0.0)
        self.total = self.total + seconds
        return seconds


Policy = Callable[[RaceState], NDArray[np.int64]]


def rollout(
    policy: Policy,
    params: RaceParams,
    sc_mask: NDArray[np.bool_],
    noise: NDArray[np.float64],
    start_compound: str = "MEDIUM",
) -> NDArray[np.float64]:
    """Total race time of ``policy`` in each race, shape (N,). All races advance together."""
    state = RaceState(params, sc_mask, noise, start_compound)
    while not state.done:
        state.step(policy(state))
    return state.total


# --- Scripted policies (baselines expressed in the same action space) -----------------

def fixed_strategy_policy(strategy: Strategy, params: RaceParams) -> Policy:
    """Pit exactly at the end of each stint of a fixed strategy."""
    strategy.validate(params.total_laps, params.compounds)
    plan = np.zeros(params.total_laps + 1, dtype=np.int64)  # plan[lap] = action on that lap
    end = 0
    for (_, length), (nxt, _) in zip(strategy.stints[:-1], strategy.stints[1:]):
        end += length
        plan[end] = params.compounds.index(nxt) + 1
    return lambda state: np.full(state.n, plan[state.lap])


def reactive_policy(policy: ReactivePolicy, params: RaceParams) -> Policy:
    """The rule-based ReactivePolicy, written against RaceState (start compound set by rollout)."""
    policy.validate(params)
    second = params.compounds.index(policy.second) + 1
    extra = None if policy.extra_compound is None else params.compounds.index(policy.extra_compound) + 1
    L = params.total_laps

    def act(state: RaceState) -> NDArray[np.int64]:
        lap, sc = state.lap, state.sc_now
        in_window = policy.plan_lap - policy.sc_window <= lap < policy.plan_lap
        first = (state.stops == 0) & ((lap == policy.plan_lap) | (sc & in_window))
        actions = np.where(first, second, 0)
        if extra is not None:
            late = (
                (state.stops == 1) & sc & (state.age >= policy.extra_min_age)
                & (L - lap >= policy.extra_min_laps_left)
            )
            actions = np.where(late, extra, actions)
        return actions

    return act


# --- Gymnasium environment -------------------------------------------------------------

class PitStopEnv(gym.Env):
    """
    Single-race Gymnasium environment for training.

    Reward per lap = -(seconds spent - base_lap_s) / reward_scale. Subtracting a
    constant per lap doesn't change which policy is best (every race has the same
    number of laps), but it centres rewards near 0, and the scale keeps a pit stop
    around -2 and an SC lap around -4: magnitudes neural networks learn well from.

    reset(options={"sc_mask": (L,) bools, "noise": (L,) floats}) replays a given race
    instead of drawing a new one (used to compare against baselines on fixed races).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: RaceParams | None = None,
        start_compound: str = "MEDIUM",
        reward_scale: float = 10.0,
    ) -> None:
        self.params = params or reference_race()
        self.start_compound = start_compound
        self.reward_scale = reward_scale
        k = len(self.params.compounds)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(k + 6,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(k + 1)
        self.state: RaceState | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        p = self.params
        if options and "sc_mask" in options:
            sc = np.asarray(options["sc_mask"], dtype=bool).reshape(1, p.total_laps)
            noise = np.asarray(options["noise"], dtype=np.float64).reshape(1, p.total_laps)
        else:
            sc = draw_safety_car(self.np_random, 1, p.total_laps, p.sc_hazard_per_lap, p.sc_duration_laps)
            noise = draw_lap_noise(self.np_random, 1, p)
        self.state = RaceState(p, sc, noise, self.start_compound)
        return self.state.observe()[0], {}

    def step(self, action):
        seconds = float(self.state.step(np.array([int(action)]))[0])
        reward = -(seconds - self.params.base_lap_s) / self.reward_scale
        terminated = self.state.done
        info = {"race_time_s": float(self.state.total[0])} if terminated else {}
        return self.state.observe()[0], reward, terminated, False, info
