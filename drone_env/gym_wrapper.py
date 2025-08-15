# drone_env/gym_wrapper.py
from __future__ import annotations

import math
from typing import Dict, Any, List, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import EnvConfig
from .environment import DroneEnvironment


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


class DefenderRLWrapper(gym.Env):
    """
    Single-agent Gymnasium wrapper that controls ALL defenders at once.

    Actions
    -------
    Flat vector of length 3 * n_defenders, where each consecutive triple is
    [ax, ay, az] in [-1, 1] for defender i. The *environment* will scale to
    physical acceleration using cfg.def_amax (this wrapper does NOT scale).

    Observations
    ------------
    Fixed-size vector:
      - For each defender:  (x,y)/arena_half, z/h_max, (vx,vy,vz)/def_vmax  -> 6 floats
      - For each attacker:  (x,y)/arena_half, z/h_max, (vx,vy,vz)/att_speed -> 6 floats
      - Progress: step_idx / max_steps -> 1 float
    = 6*n_def + 6*n_att + 1

    Reward
    ------
    Must be provided by the environment in `events["reward"]`.
    If missing, raises a RuntimeError.

    Termination
    -----------
    - terminated: base breached (as reported by env)
    - truncated: step limit reached (cfg.max_steps)

    Rendering
    ---------
    Set render_mode="human" to see GUI. The DroneEnvironment handles GUI rendering
    internally; this wrapper's render() is a no-op.
    """

    metadata = {"render_modes": ["none", "human"], "render_fps": 60}

    def __init__(
        self,
        cfg: EnvConfig | None = None,
        render_mode: str = "none",
        seed: int = 0,
    ):
        super().__init__()
        self.cfg = cfg or EnvConfig()

        # Avoid early terminations for RL training
        self.cfg.end_on_perimeter_near = False
        self.cfg.end_on_near = False

        self.render_mode = render_mode
        self.seed_val = int(seed)

        # Underlying simulation
        self.env = DroneEnvironment(self.cfg, render_mode=self.render_mode, seed=self.seed_val)
        self.state_cache: Dict[str, Any] | None = None
        self.info_cache: Dict[str, Any] | None = None

        # Team sizes (target shapes for fixed obs)
        self.n_def = int(self.cfg.n_defenders)
        self.n_att = int(self.cfg.n_attackers)

        # Observation space
        obs_dim = 6 * self.n_def + 6 * self.n_att + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3 * self.n_def,), dtype=np.float32)

        # Normalization constants for obs
        self._pos_xy_scale = float(max(1e-6, self.cfg.arena_half))
        self._pos_z_scale = float(max(1e-6, self.cfg.h_max))
        self._v_def_scale = float(max(1e-6, self.cfg.def_vmax))
        self._v_att_scale = float(max(1e-6, self.cfg.att_speed))

        self._steps = 0
        self._warned_short_lists = False

    # ---------------- Gym API ----------------

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed_val = int(seed)
        state = self.env.reset(seed=self.seed_val)
        self.state_cache = state
        self.info_cache = {}
        self._steps = 0
        return self._get_obs(state), self.info_cache

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        act_dict = self._action_to_dict(action)

        # Step environment
        state, events = self.env.step(act_dict)
        self.state_cache, self.info_cache = state, events
        self._steps += 1

        # Require reward from environment
        if "reward" not in events:
            raise RuntimeError("Environment did not provide 'reward' in events dict.")

        reward = float(events["reward"])

        # Termination / truncation
        terminated = bool(events.get("base_breached", False))
        truncated = bool(self._steps >= int(getattr(self.cfg, "max_steps", 1000)))

        obs = self._get_obs(state)
        return obs, reward, terminated, truncated, dict(events)

    def render(self):
        pass  # Env handles rendering

    def close(self):
        self.env.close()

    # --------------- Helpers ---------------

    def _action_to_dict(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """Split flat action into 3D per-defender commands in [-1,1]."""
        need = 3 * self.n_def
        if action.size < need:
            action = np.pad(action, (0, need - action.size))
        elif action.size > need:
            action = action[:need]

        out: Dict[str, np.ndarray] = {}
        for i in range(self.n_def):
            a = action[3 * i : 3 * (i + 1)]
            out[f"def_{i}"] = np.clip(a, -1.0, 1.0).astype(np.float32)
        return out

    def _get_obs(self, state: Dict[str, Any]) -> np.ndarray:
        """Build a fixed-size observation vector, padding with zeros if needed."""
        base = np.asarray(self.cfg.base_pos, dtype=np.float32)

        def encode_agent(item: Dict[str, Any], vscale: float) -> np.ndarray:
            pos = np.asarray(item["pos"], dtype=np.float32)
            vel = np.asarray(item["vel"], dtype=np.float32)
            rel = pos - base
            x, y, z = rel.tolist()
            vx, vy, vz = vel.tolist()
            return np.array(
                [
                    x / self._pos_xy_scale,
                    y / self._pos_xy_scale,
                    z / self._pos_z_scale,
                    vx / vscale,
                    vy / vscale,
                    vz / vscale,
                ],
                dtype=np.float32,
            )

        # Defenders
        dvecs: List[np.ndarray] = []
        dlist = state.get("defenders", [])
        for i in range(min(self.n_def, len(dlist))):
            dvecs.append(encode_agent(dlist[i], self._v_def_scale))
        for _ in range(self.n_def - len(dvecs)):
            dvecs.append(np.zeros(6, dtype=np.float32))

        # Attackers
        avecs: List[np.ndarray] = []
        alist = state.get("attackers", [])
        for j in range(min(self.n_att, len(alist))):
            avecs.append(encode_agent(alist[j], self._v_att_scale))
        for _ in range(self.n_att - len(avecs)):
            avecs.append(np.zeros(6, dtype=np.float32))

        # Progress
        progress = np.array(
            [_clamp(self._steps / max(1, int(getattr(self.cfg, "max_steps", 1000))), 0.0, 1.0)],
            dtype=np.float32,
        )

        # Warn once if actual counts < configured counts
        if not self._warned_short_lists and (len(dlist) < self.n_def or len(alist) < self.n_att):
            self._warned_short_lists = True

        return np.concatenate(dvecs + avecs + [progress], dtype=np.float32)
