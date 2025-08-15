# drone_env/environment.py
import time
import numpy as np
import pybullet as p, pybullet_data

from impact import ImpactManager
from .config import EnvConfig
from .agents import DefenderAgent, AttackerAgent
from .math_utils import unit, rand_on_arena_edge
from .base_layout import build_military_base
from .formations import circle_positions
from .drone_shapes import create_drone_visual

def hsv_to_rgb(h, s, v):
    i = int(h*6.0) % 6
    f = h*6.0 - i
    p0 = v*(1.0 - s); q = v*(1.0 - f*s); t = v*(1.0 - (1.0 - f)*s)
    if i == 0: r,g,b = v,t,p0
    elif i == 1: r,g,b = q,v,p0
    elif i == 2: r,g,b = p0,v,t
    elif i == 3: r,g,b = p0,q,v
    elif i == 4: r,g,b = t,p0,v
    else: r,g,b = v,p0,q
    return (float(r), float(g), float(b))

class DroneEnvironment:
    """Env with quadcopter visuals, circular formation, camera API, and near-wall/business rules."""
    def __init__(self, cfg: EnvConfig, render_mode: str = "human", seed: int = 123):
        assert render_mode in ("human", "none")
        self.cfg = cfg
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        self.client = None
        self.defenders = []
        self.attackers = []
        self.steps = 0
        self.step_count = 0
        self.impact = ImpactManager(client=self.client, dt=self.cfg.dt, default_ttl=0.75)
        self.done = False

        # physics & visuals
        self._col_sphere = None
        self._def_colors = {}
        self._att_colors = {}

        # camera state
        self._cam_distance = float(cfg.cam_distance)
        self._cam_yaw = float(cfg.cam_yaw)
        self._cam_pitch = float(cfg.cam_pitch)
        self._cam_target = np.array([cfg.base_pos[0], cfg.base_pos[1], cfg.cam_target_z], dtype=np.float32)
        self._follow_agent_name: str | None = None

        # trails
        self._prev_pos = {}

    # ---------- lifecycle ----------
    def connect(self):
        if self.client is not None: return
        self.client = p.connect(p.GUI) if self.render_mode == "human" else p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81, physicsClientId=self.client)
        p.setTimeStep(self.cfg.dt, physicsClientId=self.client)

    def _apply_camera(self):
        if self.render_mode == "human":
            self._cam_distance = float(np.clip(self._cam_distance, self.cfg.cam_min_dist, self.cfg.cam_max_dist))
            self._cam_pitch    = float(np.clip(self._cam_pitch, self.cfg.cam_min_pitch, self.cfg.cam_max_pitch))
            p.resetDebugVisualizerCamera(
                cameraDistance=self._cam_distance,
                cameraYaw=self._cam_yaw,
                cameraPitch=self._cam_pitch,
                cameraTargetPosition=self._cam_target.tolist()
            )

    def _setup_camera(self): self._apply_camera()

    def build(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0,0,-9.81, physicsClientId=self.client)
        p.setTimeStep(self.cfg.dt, physicsClientId=self.client)
        build_military_base(self.cfg)
        self._setup_camera()
        self._col_sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=0.20)

    # ---------- spawns ----------
    def _spawn_defenders(self):
        self.defenders.clear(); self._def_colors.clear()
        cx, cy, _ = self.cfg.base_pos
        pts = circle_positions(self.cfg.n_defenders, center_xy=(cx, cy),
                               radius=self.cfg.def_circle_radius, z=2.0, phase_deg=self.cfg.def_phase_deg)
        for i in range(self.cfg.n_defenders):
            # color
            if self.cfg.per_agent_colors:
                h = (i / max(1, self.cfg.n_defenders)) % 1.0
                rgb = hsv_to_rgb(h, 0.8, 1.0)
                body_rgba = (*rgb, 1.0)
            else:
                body_rgba = self.cfg.defender_color
            arm_rgba   = (0.15, 0.15, 0.15, 1.0)
            rotor_rgba = (0.05, 0.05, 0.05, 1.0)

            vis_id = create_drone_visual(
                body_radius=self.cfg.drone_body_radius,
                body_height=self.cfg.drone_body_height,
                arm_half=self.cfg.drone_arm_half,
                arm_thickness=self.cfg.drone_arm_thickness,
                rotor_radius=self.cfg.drone_rotor_radius,
                rotor_thickness=self.cfg.drone_rotor_thickness,
                color_body=body_rgba,
                color_arm=arm_rgba,
                color_rotor=rotor_rgba,
            )

            pos = pts[i].astype(np.float32)
            vel = np.zeros(3, dtype=np.float32)
            bid = p.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=self._col_sphere,
                baseVisualShapeIndex=vis_id,
                basePosition=pos.tolist(),
            )
            name = f"def_{i}"
            self._def_colors[name] = body_rgba
            self.defenders.append(DefenderAgent(name, pos, vel, bid))
            self._prev_pos[name] = pos.copy()

    def _spawn_attackers(self):
        self.attackers.clear(); self._att_colors.clear()
        pts = [rand_on_arena_edge(self.rng, self.cfg.arena_half, z=3.0) for _ in range(self.cfg.n_attackers)]
        pts = np.stack(pts, axis=0)

        base = np.array(self.cfg.base_pos, dtype=np.float32)
        for i in range(self.cfg.n_attackers):
            if self.cfg.per_agent_colors:
                h = (0.02 + 0.12 * i) % 1.0
                rgb = hsv_to_rgb(h, 0.85, 0.95)
                body_rgba = (*rgb, 1.0)
            else:
                body_rgba = self.cfg.attacker_color
            arm_rgba   = (0.15, 0.15, 0.15, 1.0)
            rotor_rgba = (0.05, 0.05, 0.05, 1.0)

            vis_id = create_drone_visual(
                body_radius=self.cfg.drone_body_radius,
                body_height=self.cfg.drone_body_height,
                arm_half=self.cfg.drone_arm_half,
                arm_thickness=self.cfg.drone_arm_thickness,
                rotor_radius=self.cfg.drone_rotor_radius,
                rotor_thickness=self.cfg.drone_rotor_thickness,
                color_body=body_rgba,
                color_arm=arm_rgba,
                color_rotor=rotor_rgba,
            )

            pos = pts[i].astype(np.float32)
            vel = unit(base - pos) * self.cfg.att_speed
            bid = p.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=self._col_sphere,
                baseVisualShapeIndex=vis_id,
                basePosition=pos.tolist(),
            )
            name = f"att_{i}"
            self._att_colors[name] = body_rgba
            self.attackers.append(AttackerAgent(name, pos, vel, bid))
            self._prev_pos[name] = pos.copy()

    # ---------- public API ----------
    def reset(self, seed: int | None = None):
        # ... your existing reset ...
        self.done = False
        self.steps = 0
        self.step_count = 0
        self._prev_pos = {}
        self._prev_nearest = 0.0  # <--- for distance-delta shaping
        # (re)spawn defenders/attackers here...
        state = self.state()
        # seed _prev_nearest from current state so first step delta is not huge
        self._prev_nearest = 0.0
        return state

    def step(self, actions: dict[str, np.ndarray] | None = None):
        # --- fast exit if already done ---
        if self.done:
            return self.state(), {
                "reward": 0.0,
                "collisions": [],
                "caught": [],
                "near_perimeter": False,
                "perimeter_nearest": float("inf"),
                "near_base": False,
                "nearest_dist": float("inf"),
                "base_breached": False,
                "game_over": True,
            }

        if actions is None:
            actions = {}

        # Snapshot BEFORE stepping for reward deltas
        prev_state = self.state()

        # =========================
        # 1) Integrate defenders (kinematic) from actions
        # =========================
        for d in self.defenders:
            a = actions.get(d.name, np.zeros(3, dtype=np.float32))
            # be robust to wrong shapes
            if not isinstance(a, np.ndarray):
                a = np.asarray(a, dtype=np.float32)
            if a.shape != (3,):
                aa = np.zeros(3, dtype=np.float32)
                aa[:min(3, a.size)] = a[:min(3, a.size)]
                a = aa
            d.step(a, self.cfg)  # your DefenderAgent.step does acc→vel→damp→clip→pos

        # =========================
        # 2) Integrate attackers (your scripted logic)
        # =========================
        base_xy = np.array(self.cfg.base_pos, dtype=np.float32)
        for a in self.attackers:
            a.step(self.cfg, base_xy)

        # =========================
        # 3) Collision: defender vs attacker  (ram = remove both + impact FX)
        # =========================
        collisions = []
        col_radius = float(getattr(self.cfg, "collision_radius", 0.35))  # tune to your drone size
        killed_def_idx = set()
        killed_atk_idx = set()

        for di, d in enumerate(self.defenders):
            dp = np.asarray(d.pos, dtype=np.float32)
            for ai, atk in enumerate(self.attackers):
                ap = np.asarray(atk.pos, dtype=np.float32)
                if np.linalg.norm(dp - ap) <= col_radius:
                    hit_pos = ((dp + ap) * 0.5).astype(float).tolist()
                    # impact UI (if you initialized self.impact)
                    try:
                        self.impact.hit(hit_pos, step_now=getattr(self, "step_count", 0), label="⚡ COLLISION")
                    except Exception:
                        pass
                    # schedule removal
                    killed_def_idx.add(di)
                    killed_atk_idx.add(ai)
                    collisions.append({"defender": d.name, "attacker": atk.name, "pos": hit_pos})

        # Remove collided bodies from Bullet and lists (do AFTER loops)
        if killed_def_idx or killed_atk_idx:
            # remove attackers
            new_attackers = []
            for i, atk in enumerate(self.attackers):
                if i in killed_atk_idx:
                    try:
                        p.removeBody(int(atk.body_id), physicsClientId=self.client)
                    except Exception:
                        pass
                    # drop any debug color / prev_pos entries
                    self._prev_pos.pop(atk.name, None)
                    self._att_colors.pop(atk.name, None) if hasattr(self, "_att_colors") else None
                else:
                    new_attackers.append(atk)
            self.attackers = new_attackers

            # remove defenders
            new_defenders = []
            for i, d in enumerate(self.defenders):
                if i in killed_def_idx:
                    try:
                        p.removeBody(int(d.body_id), physicsClientId=self.client)
                    except Exception:
                        pass
                    self._prev_pos.pop(d.name, None)
                    self._def_colors.pop(d.name, None)
                else:
                    new_defenders.append(d)
            self.defenders = new_defenders

        # =========================
        # 4) Update visuals in Bullet (teleport to kinematic poses)
        # =========================
        for d in self.defenders:
            p.resetBasePositionAndOrientation(
                int(d.body_id),
                np.asarray(d.pos, dtype=float).tolist(),
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.client,
            )
        for a in self.attackers:
            p.resetBasePositionAndOrientation(
                int(a.body_id),
                np.asarray(a.pos, dtype=float).tolist(),
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.client,
            )

        # =========================
        # 5) Camera follow (human)
        # =========================
        if self._follow_agent_name is not None and self.render_mode == "human":
            pos = self._find_agent_position(self._follow_agent_name)
            if pos is not None:
                self._cam_target = np.array([pos[0], pos[1], self.cfg.cam_target_z], dtype=np.float32)
                self._apply_camera()

        # =========================
        # 6) Trails (human)
        # =========================
        if self.render_mode == "human" and getattr(self.cfg, "trail_enable", False):
            stride = max(1, int(getattr(self.cfg, "trail_stride", 1)))
            if self.steps % stride == 0:
                for agent in (*self.defenders, *self.attackers):
                    prev = self._prev_pos.get(agent.name, agent.pos)
                    cur = agent.pos
                    if np.linalg.norm(cur - prev) > 1e-6:
                        color = (
                            (self._def_colors.get(agent.name)
                             or getattr(self, "_att_colors", {}).get(agent.name)
                             or (1.0, 1.0, 1.0, 1.0))
                        )
                        p.addUserDebugLine(
                            np.asarray(prev, dtype=float).tolist(),
                            np.asarray(cur, dtype=float).tolist(),
                            lineColorRGB=list(color[:3]),
                            lineWidth=2.0,
                            lifeTime=float(getattr(self.cfg, "trail_lifetime", 2.0)),
                            physicsClientId=self.client,
                        )
                    self._prev_pos[agent.name] = cur.copy()

        # =========================
        # 7) Events (near/ breach / caught)
        # =========================
        caught = self._caught_attackers()  # your existing logic
        near_perimeter, perim_nearest = self._any_near_perimeter_outer()
        near_base, nearest_dist = self._any_near_base()
        base_breached = self._core_breach()

        # If you still use "caught" to despawn attackers, keep this (moved far away)
        if caught:
            far = np.array([999.0, 999.0, 10.0], dtype=np.float32)
            for idx in caught:
                if 0 <= idx < len(self.attackers):
                    self.attackers[idx].pos = far.copy()
                    self.attackers[idx].vel[:] = 0
                    p.resetBasePositionAndOrientation(int(self.attackers[idx].body_id), far.tolist(), [0, 0, 0, 1],
                                                      physicsClientId=self.client)

        # Game over flags (keep your config switches)
        game_over = False
        if getattr(self.cfg, "end_on_perimeter_near", False) and near_perimeter: game_over = True
        if getattr(self.cfg, "end_on_near", False) and near_base:               game_over = True
        if getattr(self.cfg, "end_on_breach", False) and base_breached:         game_over = True
        # Optional: win if all attackers removed
        if len(self.attackers) == 0:                                            game_over = True

        if game_over:
            self.done = True

        # =========================
        # 8) Reward shaping (dense + terminal)
        # =========================
        # weights
        R_collision = 5.0
        R_breach = -10.0
        R_clear = 3.0
        k_dist = 0.10
        k_align = 0.05
        k_ctrl = -0.002
        k_nearP = -0.10
        k_nearB = -0.25
        step_cost = -0.001
        reward_clip = 10.0

        # nearest distance (defenders' centroid to nearest attacker)
        def _nearest_dist(state):
            dpos = [np.asarray(d["pos"], np.float32) for d in state.get("defenders", [])]
            apos = [np.asarray(a["pos"], np.float32) for a in state.get("attackers", [])]
            if not dpos or not apos:
                return 0.0
            dcen = np.stack(dpos).mean(axis=0)
            dists = [np.linalg.norm(dcen - ap) for ap in apos]
            return float(min(dists)) if dists else 0.0

        prev_dist = _nearest_dist(prev_state)
        curr_state = self.state()  # build AFTER all updates above
        curr_dist = _nearest_dist(curr_state)

        r = 0.0
        # terminals
        for _hit in collisions:
            r += R_collision
        if base_breached:
            r += R_breach
        if len(self.attackers) == 0 and game_over:
            r += R_clear
        # dense
        r += k_dist * (prev_dist - curr_dist)  # progress
        if curr_state.get("defenders") and curr_state.get("attackers"):
            d0 = curr_state["defenders"][0]
            a0 = curr_state["attackers"][0]
            v = np.asarray(d0.get("vel", [0, 0, 0]), np.float32)
            tgt = np.asarray(a0.get("pos", [0, 0, 0]), np.float32) - np.asarray(d0.get("pos", [0, 0, 0]), np.float32)
            nv = np.linalg.norm(v);
            nt = np.linalg.norm(tgt)
            align = float(np.dot(v, tgt) / ((nv * nt) + 1e-8)) if (nv > 1e-6 and nt > 1e-6) else 0.0
            r += k_align * align
        for name, a in (actions or {}).items():
            aa = np.asarray(a, np.float32)
            r += k_ctrl * float(np.dot(aa, aa))
        if near_perimeter: r += k_nearP
        if near_base:      r += k_nearB
        r += step_cost
        r = float(np.clip(r, -reward_clip, reward_clip))

        # =========================
        # 9) Bookkeeping, impact FX cleanup, timing
        # =========================
        self.steps += 1
        self.step_count += 1
        try:
            self.impact.update(self.step_count)
        except Exception:
            pass

        if self.render_mode == "human":
            import time as _time
            _time.sleep(float(self.cfg.dt))

        # =========================
        # 10) Return state + events (reward inside events so wrapper can read it)
        # =========================
        return curr_state, {
            "reward": r,
            "collisions": collisions,
            "caught": caught,
            "near_perimeter": bool(near_perimeter),
            "perimeter_nearest": float(perim_nearest),
            "near_base": bool(near_base),
            "nearest_dist": float(nearest_dist),
            "base_breached": bool(base_breached),
            "game_over": bool(self.done),
        }

    def state(self):
        return {
            "defenders": [{"name": d.name, "pos": d.pos.copy(), "vel": d.vel.copy()} for d in self.defenders],
            "attackers": [{"name": a.name, "pos": a.pos.copy(), "vel": a.vel.copy()} for a in self.attackers],
            "base": {"pos": np.array(self.cfg.base_pos, dtype=np.float32)},
            "step": self.steps
        }

    # ---------- camera API ----------
    def get_camera(self):
        return dict(distance=float(self._cam_distance), yaw=float(self._cam_yaw),
                    pitch=float(self._cam_pitch), target=self._cam_target.copy())

    def set_camera(self, distance=None, yaw=None, pitch=None, target=None):
        if distance is not None: self._cam_distance = float(distance)
        if yaw is not None:      self._cam_yaw      = float(yaw)
        if pitch is not None:    self._cam_pitch    = float(pitch)
        if target is not None:   self._cam_target   = np.asarray(target, dtype=np.float32).reshape(3)
        self._apply_camera()

    def move_camera(self, dz=0.0, dyaw=0.0, dpitch=0.0, dtarget_world=(0.0,0.0,0.0)):
        self._cam_distance += float(dz)
        self._cam_yaw      += float(dyaw)
        self._cam_pitch    += float(dpitch)
        self._cam_target   = self._cam_target + np.asarray(dtarget_world, dtype=np.float32).reshape(3)
        self._apply_camera()

    def pan_local(self, right=0.0, forward=0.0, up=0.0):
        yaw = np.deg2rad(self._cam_yaw)
        cam_right   = np.array([ np.cos(yaw),  np.sin(yaw), 0.0], dtype=np.float32)
        cam_forward = np.array([-np.sin(yaw),  np.cos(yaw), 0.0], dtype=np.float32)
        delta = right * cam_right + forward * cam_forward + np.array([0.0, 0.0, up], dtype=np.float32)
        self._cam_target = self._cam_target + delta
        self._apply_camera()

    def set_follow(self, agent_name: str | None):
        self._follow_agent_name = agent_name
        if agent_name is not None:
            pos = self._find_agent_position(agent_name)
            if pos is not None:
                self._cam_target = np.array([pos[0], pos[1], self.cfg.cam_target_z], dtype=np.float32)
        self._apply_camera()

    # ---------- helpers & events ----------
    def _find_agent_position(self, name: str):
        for d in self.defenders:
            if d.name == name: return d.pos
        for a in self.attackers:
            if a.name == name: return a.pos
        return None

    def _caught_attackers(self):
        caught = []
        for j, a in enumerate(self.attackers):
            dists = [np.linalg.norm(d.pos - a.pos) for d in self.defenders]
            if len(dists) and min(dists) <= self.cfg.catch_radius:
                caught.append(j)
                self.impact.hit(pos=a.pos, step_now=self.step_count, ttl=0.75, label="⚡ COLLISION")
                # optional extras:
                self.impact.ring(a.pos, step_now=self.step_count, radius=1.2, ttl=0.5)
                self.impact.rays(a.pos, step_now=self.step_count, length=1.3, ttl=0.4)

        return caught

    def _any_near_perimeter_outer(self):
        """Distance to outer wall faces: trigger before touching any wall."""
        hx, hy = self.cfg.base_half_x, self.cfg.base_half_y
        t = self.cfg.wall_thickness
        Bx, By = hx + t, hy + t  # outer faces
        margin = self.cfg.perimeter_near_margin
        nearest = float("inf"); near = False
        for a in self.attackers:
            x, y = float(a.pos[0]), float(a.pos[1])
            dx = max(abs(x) - Bx, 0.0)
            dy = max(abs(y) - By, 0.0)
            d_out = (dx*dx + dy*dy) ** 0.5
            nearest = min(nearest, d_out)
            if d_out <= margin: near = True
        if nearest == float("inf"): nearest = 0.0
        return near, nearest

    def _any_near_base(self):
        base = np.array(self.cfg.base_pos, dtype=np.float32)
        dists = [np.linalg.norm(a.pos - base) for a in self.attackers]
        if not dists: return False, float("inf")
        nearest = min(dists)
        threshold = self.cfg.base_radius + self.cfg.near_margin
        return (nearest <= threshold), nearest

    def _core_breach(self):
        base = np.array(self.cfg.base_pos, dtype=np.float32)
        for a in self.attackers:
            if np.linalg.norm(a.pos - base) <= self.cfg.base_radius:
                return True
        return False

    def close(self):
        if self.client is not None:
            p.disconnect(self.client); self.client = None
        if hasattr(self, "impact"):
            self.impact.close()

    def _compute_reward(self, prev_state, curr_state, events, actions_dict):
        # prev_state/curr_state: what your state() returns before/after step
        # events: your computed flags (near_base, base_breached, collisions, etc.)

        # === weights ===
        R_collision = 5.0
        R_breach = -10.0
        R_clear = 3.0

        k_dist = 0.10
        k_align = 0.05
        k_ctrl = -0.002
        k_nearP = -0.10
        k_nearB = -0.25
        k_bounds = 0.01
        step_cost = -0.01
        reward_clip = 10.0

        r = 0.0

        # --- terminal events ---
        # collisions reported as list of {"defender","attacker","pos"}
        for _hit in events.get("collisions", []):
            r += R_collision
        if events.get("base_breached", False):
            r += R_breach
        if events.get("game_over", False) and len(curr_state.get("attackers", [])) == 0:
            r += R_clear

        # --- dense shaping ---
        # 1) distance progress to nearest attacker (use defenders' centroid or lead)
        def _nearest_dist(state):
            dpos = [np.asarray(d["pos"], np.float32) for d in state.get("defenders", [])]
            apos = [np.asarray(a["pos"], np.float32) for a in state.get("attackers", [])]
            if not dpos or not apos: return 0.0
            dcen = np.stack(dpos).mean(axis=0)
            dists = [np.linalg.norm(dcen - a) for a in apos]
            return float(min(dists)) if dists else 0.0

        prev_dist = _nearest_dist(prev_state)
        curr_dist = _nearest_dist(curr_state)
        r += k_dist * (prev_dist - curr_dist)  # positive if getting closer

        # 2) velocity alignment (for def_0; extend to all if you want)
        if curr_state.get("defenders") and curr_state.get("attackers"):
            d0 = curr_state["defenders"][0]
            a0 = curr_state["attackers"][0]
            v = np.asarray(d0.get("vel", [0, 0, 0]), np.float32)
            tgt = np.asarray(a0.get("pos", [0, 0, 0]), np.float32) - np.asarray(d0.get("pos", [0, 0, 0]), np.float32)
            nv = np.linalg.norm(v);
            nt = np.linalg.norm(tgt)
            align = float(np.dot(v, tgt) / ((nv * nt) + 1e-8)) if (nv > 1e-6 and nt > 1e-6) else 0.0
            r += k_align * align  # in [-k_align, +k_align]

        # 3) control penalty (sum over defenders with an action)
        for name, a in (actions_dict or {}).items():
            aa = np.asarray(a, np.float32)
            r += k_ctrl * float(np.dot(aa, aa))  # -||a||^2

        # 4) perimeter/base proximity penalties (use your computed flags)
        if events.get("near_perimeter", False):
            r += k_nearP
        if events.get("near_base", False):
            r += k_nearB

        # 5) bounds penalty (if you have arena limits)
        arena = getattr(self.cfg, "arena_bounds", None)  # e.g., (xmin,xmax,ymin,ymax,zmin,zmax)
        if arena:
            xmin, xmax, ymin, ymax, zmin, zmax = arena
            for d in curr_state.get("defenders", []):
                x, y, z = d["pos"]
                if not (xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax):
                    r += k_bounds

        # 6) small step cost to encourage faster intercepts
        r += step_cost

        # clip for stability
        return float(np.clip(r, -reward_clip, reward_clip))
