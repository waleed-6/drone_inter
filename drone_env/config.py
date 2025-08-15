from dataclasses import dataclass

@dataclass
class EnvConfig:
    # Timing
    dt: float = 1.0/60.0
    max_steps: int = 1500

    # Bounds
    arena_half: float = 60.0
    h_min: float = 0.5
    h_max: float = 25.0

    # Team sizes
    n_defenders: int = 5
    n_attackers: int = 5

    # Defender dynamics
    def_amax: float = 30.0
    def_vmax: float = 20.0
    v_damp: float = 1

    # Attacker dynamics
    att_speed: float = 20.0

    # Core/breach
    base_pos: tuple = (0.0, 0.0, 0.6)
    base_radius: float = 3.0
    catch_radius: float = 0.8

    # Business rules
    perimeter_near_margin: float = 1.0
    end_on_perimeter_near: bool = True
    near_margin: float = 2.0
    end_on_near: bool = True
    end_on_breach: bool = True

    # Base rectangle + walls
    base_half_x: float = 20.0
    base_half_y: float = 14.0
    wall_height: float = 3.0
    wall_thickness: float = 0.5

    # ---------- Formation (KEEP CIRCLE) ----------
    def_spawn_formation: str = "circle"   # ensure circular layout
    def_circle_radius: float = 10.0       # radius of the defender ring
    def_phase_deg: float = 0.0            # starting angle
    att_spawn_formation: str = "edges"    # attackers from arena edges

    # ---------- Visuals ----------
    per_agent_colors: bool = True
    trail_enable: bool = True
    trail_lifetime: float = 2.5
    trail_stride: int = 2

    # Team base colors (used if per_agent_colors=False)
    defender_color: tuple = (0.15, 0.45, 1.0, 1.0)
    attacker_color: tuple = (1.0, 0.25, 0.25, 1.0)

    # Drone visual sizing (tweak for style)
    drone_body_radius: float = 0.28
    drone_body_height: float = 0.10
    drone_arm_half: float = 0.95          # half-length of each arm
    drone_arm_thickness: float = 0.07
    drone_rotor_radius: float = 0.22
    drone_rotor_thickness: float = 0.02

    # Camera
    cam_distance: float = 70.0
    cam_yaw: float = 40.0
    cam_pitch: float = -30.0
    cam_target_z: float = 3.0
    cam_zoom_step: float = 3.0
    cam_yaw_step: float = 5.0
    cam_pitch_step: float = 3.0
    cam_pan_step: float = 3.0
    cam_min_dist: float = 5.0
    cam_max_dist: float = 200.0
    cam_min_pitch: float = -85.0
    cam_max_pitch: float = 10.0

    # Visual styling of the map
    ground_color: tuple = (0.35, 0.45, 0.35, 1.0)
    asphalt_color: tuple = (0.12, 0.12, 0.12, 1.0)
    concrete_color: tuple = (0.6, 0.6, 0.6, 1.0)
    wall_color: tuple = (0.45, 0.45, 0.48, 1.0)
    tower_color: tuple = (0.3, 0.3, 0.35, 1.0)
    hangar_color: tuple = (0.5, 0.55, 0.6, 1.0)
