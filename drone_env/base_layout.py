import pybullet as p

def _box(half_extents, rgba):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
    return col, vis

def _cyl(radius, height, rgba):
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=rgba)
    return col, vis

def build_ground(cfg):
    p.loadURDF("plane.urdf")
    col, vis = _box([cfg.arena_half, cfg.arena_half, 0.05], cfg.ground_color)
    p.createMultiBody(0.0, col, vis, [0,0,-0.05])

def build_perimeter_walls(cfg):
    hx, hy = cfg.base_half_x, cfg.base_half_y
    t = cfg.wall_thickness
    h = cfg.wall_height
    wall_col, wall_vis = _box([hx, t/2, h/2], cfg.wall_color)
    side_col, side_vis = _box([t/2, hy, h/2], cfg.wall_color)

    y_top =  hy + t/2
    y_bot = -hy - t/2
    x_r   =  hx + t/2
    x_l   = -hx - t/2

    # North/South
    p.createMultiBody(0.0, wall_col, wall_vis, [0, y_top, h/2])
    p.createMultiBody(0.0, wall_col, wall_vis, [0, y_bot, h/2])
    # East/West
    p.createMultiBody(0.0, side_col, side_vis, [x_r, 0, h/2])
    p.createMultiBody(0.0, side_col, side_vis, [x_l, 0, h/2])

def build_corner_towers(cfg):
    hx, hy = cfg.base_half_x, cfg.base_half_y
    r = 1.2
    h = cfg.wall_height + 2.5
    col, vis = _cyl(r, h, cfg.tower_color)
    z = h/2
    offs = 1.2
    for sx in (-1, 1):
        for sy in (-1, 1):
            x = sx*(hx - offs)
            y = sy*(hy - offs)
            p.createMultiBody(0.0, col, vis, [x, y, z])

def build_runway(cfg):
    length = cfg.base_half_x * 1.8
    width  = 6.0
    thickness = 0.05
    col, vis = _box([length/2, width/2, thickness/2], cfg.asphalt_color)
    p.createMultiBody(0.0, col, vis, [-4.0, 0.0, thickness/2 - 0.01])

def build_hangars(cfg):
    hx, hy, hz = 6.0, 4.0, 3.2
    base_y = -cfg.base_half_y + hy + 1.5
    xs = [-cfg.base_half_x/2, cfg.base_half_x/2]
    for x in xs:
        col, vis = _box([hx, hy, hz], cfg.hangar_color)
        p.createMultiBody(0.0, col, vis, [x, base_y, hz])

def build_helipad(cfg):
    radius = 4.0
    thickness = 0.07
    col, vis = _cyl(radius, thickness, cfg.concrete_color)
    p.createMultiBody(0.0, col, vis, [cfg.base_pos[0]+10.0, cfg.base_pos[1]+6.0, thickness/2 - 0.015])
    # 'H' marking (three thin boxes)
    h_col, h_vis = _box([0.2, 2.2, 0.02], (1,1,1,1))
    p.createMultiBody(0.0, h_col, h_vis, [cfg.base_pos[0]+8.8,  cfg.base_pos[1]+6.0, thickness+0.01])
    p.createMultiBody(0.0, h_col, h_vis, [cfg.base_pos[0]+11.2, cfg.base_pos[1]+6.0, thickness+0.01])
    mid_col, mid_vis = _box([1.3, 0.2, 0.02], (1,1,1,1))
    p.createMultiBody(0.0, mid_col, mid_vis,[cfg.base_pos[0]+10.0, cfg.base_pos[1]+6.0, thickness+0.01])

def build_base_core_marker(cfg):
    # translucent disc to show breach zone
    col, vis = _cyl(cfg.base_radius, 0.02, (1.0, 0.2, 0.2, 0.25))
    p.createMultiBody(0.0, col, vis, [*cfg.base_pos])

def build_military_base(cfg):
    build_ground(cfg)
    build_perimeter_walls(cfg)
    build_corner_towers(cfg)
    build_runway(cfg)
    build_hangars(cfg)
    build_helipad(cfg)
    build_base_core_marker(cfg)
