# drone_env/agents.py
# Defenders: acceleration-controlled with damping and speed cap.
# Attackers: immediate constant-speed pursuit straight toward the base (no delay logic).

from __future__ import annotations
import numpy as np
from .math_utils import unit, clip_speed, clamp_box

# In your Defender agent class
class DefenderAgent:
    def __init__(self, name, pos, vel, body_id):
        self.name = name
        self.pos  = np.asarray(pos, np.float32)
        self.vel  = np.asarray(vel, np.float32)
        self.body_id = int(body_id)
        self.last_action = np.zeros(3, np.float32)

    def step(self, a: np.ndarray, cfg):
        a = np.asarray(a, np.float32)
        if a.shape != (3,):
            aa = np.zeros(3, np.float32); aa[:min(3, a.size)] = a[:min(3, a.size)]
            a = aa
        self.last_action = np.clip(a, -1.0, 1.0)  # store raw command in [-1,1]

        dt   = float(cfg.dt)
        amax = float(cfg.def_amax)
        vmax = float(cfg.def_vmax)
        damp = float(cfg.v_damp)

        acc = self.last_action * amax
        self.vel = (self.vel + acc * dt) * damp
        s = float(np.linalg.norm(self.vel))
        if s > vmax and s > 1e-9:
            self.vel = self.vel / s * vmax
        self.pos = self.pos + self.vel * dt


class AttackerAgent:
    def __init__(self, name: str, pos: np.ndarray, vel: np.ndarray, body_id: int):
        self.name = name
        self.pos = pos.astype(np.float32)
        self.vel = vel.astype(np.float32)
        self.body_id = int(body_id)

    def step(self, cfg, base: np.ndarray) -> None:
        """
        Immediate pursuit: every step (including the first), point at base and move
        at constant speed cfg.att_speed. No delay, no ramp-up.
        """
        direction = unit(np.asarray(base, dtype=np.float32) - self.pos)
        self.vel = direction * float(cfg.att_speed)

        # Integrate position (attackers have no damping; constant speed)
        self.pos = self.pos + self.vel * float(cfg.dt)

        # Keep inside arena/altitude bounds
        self.pos, self.vel = clamp_box(
            self.pos, self.vel,
            float(cfg.arena_half),
            float(cfg.h_min),
            float(cfg.h_max),
        )
