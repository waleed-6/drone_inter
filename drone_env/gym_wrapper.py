# drone_env/gym_wrapper.py
import math
from typing import Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import EnvConfig
from .environment import DroneEnvironment

def _clamp(x, lo, hi):
    return float(max(lo, min(hi, x)))

class DefenderRLWrapper(gym.Env):
    """
    Single-agent Gymnasium wrapper that controls ALL defenders at once.
    Action: concatenated accelerations for N defenders (3 each), in range [-1, +1] -> scaled by def_amax.
    Observation: flattened defenders and attackers states (relative to base), plus step progress.
    Reward: catch bonuses, breach penalties, distance shaping, small control penalty, alive bonus.
    """
    metadata = {"render_modes": ["none", "human"], "render_fps": 60}

    def __init__(self,
                 cfg: EnvConfig | None = None,
                 render_mode: str = "none",
                 seed: int = 0):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        # For RL training, avoid ending on "near" to keep episodes meaningful
        self.cfg.end_on_perimeter_near = False
        self.cfg.end_on_near = False
        self.render_mode = render_mode
        self.seed_val = int(seed)

        # Underlying sim
        self.env = DroneEnvironment(self.cfg, render_mode=self.render_mode, seed=self.seed_val)
        self.state_cache = None  # last state dict
        self.info_cache = None   # last info dict

        # Spaces
        self.n_def = int(self.cfg.n_defenders)
        self.n_att = int(self.cfg.n_attackers)

        # Observation = for each defender: (x,y)/arena, z/hmax, (vx,vy,vz)/vmax  -> 6 * n_def
        #             + for each attacker: (x,y)/arena, z/hmax, (vx,vy,vz)/att_speed -> 6 * n_att
        #             + step_progress in [0,1]
        obs_dim = (6 * self.n_def) + (6 * self.n_att) + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action = 3D accel per defender, in [-1,+1], wrapper scales by def_amax
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3 * self.n_def,), dtype=np.float32)

        # Normalization constants
        self._pos_xy_scale = float(self.cfg.arena_half)
        self._pos_z_scale  = float(self.cfg.h_max)
        self._v_def_scale  = max(1e-6, float(self.cfg.def_vmax))
        self._v_att_scale  = max(1e-6, float(self.cfg.att_speed))

        # reward weights
        self.R_CATCH   = 10.0
        self.R_BREACH  = -100.0
        self.W_NEAREST = 3.0     # shaping: reward larger nearest distance to base
        self.W_EFFORT  = 0.05    # control penalty weight
        self.R_ALIVE   = 0.01    # tiny per-step bonus

        self._steps = 0

    # ------------- Gym API -------------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            self.seed_val = int(seed)
        self.state_cache = self.env.reset(seed=self.seed_val)
        self.info_cache = {}
        self._steps = 0
        return self._get_obs(self.state_cache), self.info_cache

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        act_dict = self._action_to_dict(action)
        state, events = self.env.step(act_dict)
        self.state_cache, self.info_cache = state, events
        self._steps += 1

        # Reward shaping
        reward = 0.0
        # + catch bonus
        reward += self.R_CATCH * float(len(events.get("caught", [])))
        # - breach penalty
        if events.get("base_breached", False):
            reward += self.R_BREACH
        # + encourage attackers far from base (nearest distance normalized)
        nearest = float(events.get("nearest_dist", float("inf")))
        if math.isfinite(nearest):
            norm_near = _clamp(nearest / self.cfg.arena_half, 0.0, 1.0)  # 0..1
            reward += self.W_NEAREST * norm_near
        # - small action effort (mean squared accel per defender)
        effort = float(np.mean(np.square(action)))
        reward -= self.W_EFFORT * effort
        # + alive bonus
        reward += self.R_ALIVE

        # Termination
        # We only terminate on breach or time limit (wrapper managed),
        # because we disabled near/perimeter early termination in __init__.
        terminated = bool(events.get("base_breached", False))
        truncated  = bool(self._steps >= self.cfg.max_steps)

        obs = self._get_obs(state)
        return obs, float(reward), terminated, truncated, dict(events)

    def render(self):
        # The DroneEnvironment handles all GUI rendering itself when render_mode="human".
        pass

    def close(self):
        self.env.close()

    # ------------- helpers -------------
    def _action_to_dict(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        amax = float(self.cfg.def_amax)
        out: Dict[str, np.ndarray] = {}
        for i in range(self.n_def):
            a = action[3*i:3*(i+1)]
            # scale from [-1,1] to [-def_amax, +def_amax]
            u = np.clip(a, -1.0, 1.0) * amax
            out[f"def_{i}"] = u.astype(np.float32)
        return out

    def _get_obs(self, state: Dict[str, Any]) -> np.ndarray:
        base = np.array(self.cfg.base_pos, dtype=np.float32)

        def proc_agent(item, vscale):
            pos = np.asarray(item["pos"], dtype=np.float32)
            vel = np.asarray(item["vel"], dtype=np.float32)
            rel = pos - base
            x, y, z = rel.tolist()
            vx, vy, vz = vel.tolist()
            return np.array([
                x / self._pos_xy_scale,
                y / self._pos_xy_scale,
                z / self._pos_z_scale,
                vx / vscale,
                vy / vscale,
                vz / vscale,
            ], dtype=np.float32)

        # defenders
        dvecs = []
        for i in range(self.n_def):
            dvecs.append(proc_agent(state["defenders"][i], self._v_def_scale))
        # attackers
        avecs = []
        for j in range(self.n_att):
            avecs.append(proc_agent(state["attackers"][j], self._v_att_scale))

        progress = np.array([_clamp(self._steps / max(1, self.cfg.max_steps), 0.0, 1.0)], dtype=np.float32)
        obs = np.concatenate(dvecs + avecs + [progress], dtype=np.float32)
        return obs
