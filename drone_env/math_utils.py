# drone_env/math_utils.py
# Math helpers used across the env & agents.

from __future__ import annotations
import numpy as np

def unit(v: np.ndarray) -> np.ndarray:
    """Return v / ||v|| (safe for zero)."""
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)

def clip_speed(v: np.ndarray, vmax: float) -> np.ndarray:
    """Clamp vector speed to <= vmax (preserve direction)."""
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    vmax = float(vmax)
    if n > vmax and n > 1e-8:
        v = v * (vmax / n)
    return v.astype(np.float32)

def clamp_box(pos: np.ndarray,
              vel: np.ndarray,
              arena_half: float,
              h_min: float,
              h_max: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep (pos, vel) inside an axis-aligned box:
      x,y in [-arena_half, +arena_half]
      z    in [h_min, h_max]
    If we hit a boundary, clamp position and zero the outward velocity component.
    """
    p = np.asarray(pos, dtype=np.float32).copy()
    v = np.asarray(vel, dtype=np.float32).copy()
    H = float(arena_half)
    zmin = float(h_min)
    zmax = float(h_max)

    # X bound
    if p[0] > H:
        p[0] = H
        if v[0] > 0: v[0] = 0.0
    elif p[0] < -H:
        p[0] = -H
        if v[0] < 0: v[0] = 0.0

    # Y bound
    if p[1] > H:
        p[1] = H
        if v[1] > 0: v[1] = 0.0
    elif p[1] < -H:
        p[1] = -H
        if v[1] < 0: v[1] = 0.0

    # Z bound
    if p[2] > zmax:
        p[2] = zmax
        if v[2] > 0: v[2] = 0.0
    elif p[2] < zmin:
        p[2] = zmin
        if v[2] < 0: v[2] = 0.0

    return p.astype(np.float32), v.astype(np.float32)

def rand_on_arena_edge(rng: np.random.Generator,
                       arena_half: float,
                       z: float = 0.0) -> np.ndarray:
    """
    Pick a random point on the perimeter of the square arena (at altitude z).
    """
    H = float(arena_half)
    side = int(rng.integers(0, 4))  # 0:+X, 1:-X, 2:+Y, 3:-Y
    t = float(rng.uniform(-H, H))
    if side == 0:
        x, y = +H, t
    elif side == 1:
        x, y = -H, t
    elif side == 2:
        x, y = t, +H
    else:
        x, y = t, -H
    return np.array([x, y, float(z)], dtype=np.float32)
