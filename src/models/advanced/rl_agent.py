"""
PPO agent for PitStopEnv (stable-baselines3) and helpers to evaluate it like the baselines.

PPO (Proximal Policy Optimization) learns a stochastic policy pi(action | state) with a
neural network, from many simulated races. Each update nudges the policy towards
actions that turned out better than expected, but clips the step size so one lucky
batch of races cannot wreck what was learned: the "proximal" part.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from src.data.ingestion import PROJECT_ROOT
from src.models.advanced.strategy_env import PitStopEnv, Policy, RaceState, rollout
from src.models.baselines.race_model import RaceParams, reference_race

MODELS_DIR: Path = PROJECT_ROOT / "data" / "models"


def linear_schedule(initial: float):
    """Learning rate that decays linearly to 0 over training (SB3 passes progress_remaining: 1 -> 0)."""
    return lambda progress_remaining: initial * progress_remaining


# Chosen from a 4-way comparison (notebook 06, section 2): with a constant 3e-4 learning rate
# and small batches the greedy policy found a good one-stop, then drifted away from it.
# Bigger batches average out the noisy race outcomes; a decaying rate lets it settle.
DEFAULT_PPO_PARAMS: dict[str, Any] = {
    "n_steps": 2048,          # per env, per update: 8 envs x 2048 = 16,384 laps (~290 races)
    "batch_size": 1024,
    "n_epochs": 10,
    "learning_rate": linear_schedule(3e-4),
    "gamma": 1.0,             # fixed-length episodes: every second counts the same
    "gae_lambda": 0.95,
    "ent_coef": 0.001,        # a little exploration, not enough to keep it pitting at random
    "clip_range": 0.2,
    "policy_kwargs": {"net_arch": [128, 128]},
}


def agent_policy(model: PPO) -> Policy:
    """Deterministic (greedy) policy of a trained agent, usable with ``rollout`` on N races at once."""
    return lambda state: model.predict(state.observe(), deterministic=True)[0]


class RaceTimeEvalCallback(BaseCallback):
    """Every ``eval_freq`` env steps, record the greedy policy's mean race time on fixed races."""

    def __init__(self, params: RaceParams, sc_mask: NDArray[np.bool_], noise: NDArray[np.float64],
                 eval_freq: int = 20_000, start_compound: str = "MEDIUM") -> None:
        super().__init__()
        self.params, self.sc_mask, self.noise = params, sc_mask, noise
        self.eval_freq, self.start_compound = eval_freq, start_compound
        self.timesteps: list[int] = []
        self.mean_race_time_s: list[float] = []
        self._next = 0

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            times = rollout(agent_policy(self.model), self.params, self.sc_mask, self.noise, self.start_compound)
            self.timesteps.append(self.num_timesteps)
            self.mean_race_time_s.append(float(times.mean()))
            self._next += self.eval_freq
        return True


def train_ppo(
    params: RaceParams | None = None,
    total_timesteps: int = 6_000_000,
    n_envs: int = 8,
    seed: int = 0,
    start_compound: str = "MEDIUM",
    callback: BaseCallback | None = None,
    **ppo_overrides: Any,
) -> PPO:
    """Train PPO on random races of ``params`` (default: the reference race)."""
    params = params or reference_race()
    env = make_vec_env(
        PitStopEnv, n_envs=n_envs, seed=seed,
        env_kwargs={"params": params, "start_compound": start_compound},
    )
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu", **{**DEFAULT_PPO_PARAMS, **ppo_overrides})
    model.learn(total_timesteps=total_timesteps, callback=callback)
    return model


def pit_decisions(policy: Policy, params: RaceParams, sc_mask, noise, start_compound: str = "MEDIUM"):
    """
    Replay ``policy`` and record every pit stop: arrays (race index, lap, new compound
    index, SC out on that lap). Used to visualise what the agent learned.
    """
    state = RaceState(params, sc_mask, noise, start_compound)
    rows = []
    while not state.done:
        actions = np.asarray(policy(state))
        sc, lap = state.sc_now.copy(), state.lap
        for s in np.flatnonzero(actions > 0):
            rows.append((s, lap, actions[s] - 1, sc[s]))
        state.step(actions)
    return np.array(rows, dtype=np.int64).reshape(-1, 4)
