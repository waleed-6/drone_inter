import numpy as np

def circle_positions(n, center_xy=(0.0, 0.0), radius=8.0, z=2.0, phase_deg=0.0):
    cx, cy = center_xy
    angles = np.linspace(0.0, 2*np.pi, n, endpoint=False) + np.deg2rad(phase_deg)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    zs = np.full(n, z, dtype=np.float32)
    return np.stack([xs, ys, zs], axis=1).astype(np.float32)

def grid_positions(n, center_xy=(0.0, 0.0), rows=2, cols=3, spacing=4.0, z=2.0):
    rows = max(1, rows); cols = max(1, cols)
    total = rows * cols
    m = min(n, total)
    cx, cy = center_xy
    xs = np.linspace(-(cols-1)/2.0, (cols-1)/2.0, cols) * spacing + cx
    ys = np.linspace(-(rows-1)/2.0, (rows-1)/2.0, rows) * spacing + cy
    grid = np.array([(x,y,z) for y in ys for x in xs], dtype=np.float32)[:m]
    if m < n:
        extra = np.tile(grid[-1], (n-m,1))
        grid = np.vstack([grid, extra])
    return grid

def v_positions(n, center_xy=(0.0, 0.0), angle_deg=40.0, arm_spacing=3.0, z=2.0):
    """Two symmetric arms forming a V centered at center_xy, opening at angle_deg."""
    cx, cy = center_xy
    ang = np.deg2rad(angle_deg / 2.0)
    right_dir = np.array([ np.cos(ang),  np.sin(ang)], dtype=np.float32)
    left_dir  = np.array([ np.cos(ang), -np.sin(ang)], dtype=np.float32)
    pts = [np.array([cx, cy, z], dtype=np.float32)]  # tip
    k = 1
    while len(pts) < n:
        for d in (right_dir, left_dir):
            if len(pts) >= n: break
            offset = d * (k * arm_spacing)
            pts.append(np.array([cx, cy, z], dtype=np.float32) + np.array([offset[0], offset[1], 0.0], dtype=np.float32))
        k += 1
    return np.stack(pts, axis=0)

def line_positions(n, center_xy=(0.0, 0.0), dir_deg=0.0, spacing=3.0, z=2.0):
    """Line centered at center_xy, pointing at dir_deg (0° = +X)."""
    cx, cy = center_xy
    ang = np.deg2rad(dir_deg)
    fwd = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
    start = -((n-1)/2.0) * spacing
    pts = []
    for i in range(n):
        off = (start + i*spacing) * fwd
        pts.append([cx + off[0], cy + off[1], z])
    return np.array(pts, dtype=np.float32)
