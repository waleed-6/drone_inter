# drone_env/agents.py
# Defenders: acceleration-controlled with damping and speed cap.
# Attackers: immediate constant-speed pursuit straight toward the base (no delay logic).

from __future__ import annotations
import numpy as np
from .math_utils import unit, clip_speed, clamp_box

class DefenderAgent:
    def __init__(self, name: str, pos: np.ndarray, vel: np.ndarray, body_id: int):
        self.name = name
        self.pos = pos.astype(np.float32)
        self.vel = vel.astype(np.float32)
        self.body_id = int(body_id)

    def step(self, accel: np.ndarray, cfg) -> None:
        # Clip acceleration to max
        a = np.asarray(accel, dtype=np.float32)
        amax = float(cfg.def_amax)
        an = float(np.linalg.norm(a))
        if an > amax and an > 1e-8:
            a = a * (amax / an)

        # Integrate velocity with damping; cap speed
        self.vel = self.vel + a * float(cfg.dt)
        self.vel *= float(cfg.v_damp)
        self.vel = clip_speed(self.vel, float(cfg.def_vmax))

        # Integrate position
        self.pos = self.pos + self.vel * float(cfg.dt)

        # Keep inside arena bounds (XY) and altitude band (Z)
        self.pos, self.vel = clamp_box(
            self.pos, self.vel,
            float(cfg.arena_half),
            float(cfg.h_min),
            float(cfg.h_max),
        )

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
