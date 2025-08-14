# Control def_0 with keyboard; others idle. Ends game on near/breach.
# Move: W/S (±Y), A/D (∓X), R/F (±Z)
# Camera: +/- zoom, ←/→ yaw, ↑/↓ pitch, I/J/K/L pan, U/O up/down, T/1..9 follow, 0 stop, C base, G centroid, ESC quit.
import numpy as np
import pybullet as p
from drone_env.config import EnvConfig
from drone_env.environment import DroneEnvironment

def defenders_centroid(state):
    pts = [np.array(d["pos"], dtype=np.float32) for d in state["defenders"]]
    if not pts: return np.zeros(3, dtype=np.float32)
    return np.stack(pts, axis=0).mean(axis=0)

if __name__ == "__main__":
    cfg = EnvConfig()
    env = DroneEnvironment(cfg, render_mode="human", seed=7)
    state = env.reset()
    lead = state["defenders"][0]["name"]
    scale = 0.6
    print("Controls: W/S, A/D, R/F; Q/E scale; +/- zoom; arrows yaw/pitch; "
          "I/J/K/L/U/O pan; T follow def_0; 1..9 follow def_i; 0 stop; C base; G defenders; ESC quit")

    try:
        while True:
            keys = p.getKeyboardEvents()
            if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:  # ESC
                break

            # -------- drone control (def_0) --------
            ax = ay = az = 0.0
            if ord('w') in keys: ay += 1.0
            if ord('s') in keys: ay -= 1.0
            if ord('a') in keys: ax -= 1.0
            if ord('d') in keys: ax += 1.0
            if ord('r') in keys: az += 1.0
            if ord('f') in keys: az -= 1.0

            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                scale = max(0.1, scale - 0.1); print("Accel scale:", scale)
            if ord('e') in keys and keys[ord('e')] & p.KEY_WAS_TRIGGERED:
                scale = min(1.5, scale + 0.1); print("Accel scale:", scale)

            a = np.array([ax, ay, az], dtype=np.float32)
            n = np.linalg.norm(a)
            if n > 1e-6: a /= n
            a *= scale

            # -------- camera control --------
            if ord('=') in keys or ord('+') in keys: env.move_camera(dz=-cfg.cam_zoom_step)
            if ord('-') in keys or ord('_') in keys: env.move_camera(dz= cfg.cam_zoom_step)
            if p.B3G_LEFT_ARROW  in keys: env.move_camera(dyaw=-cfg.cam_yaw_step)
            if p.B3G_RIGHT_ARROW in keys: env.move_camera(dyaw= cfg.cam_yaw_step)
            if p.B3G_UP_ARROW    in keys: env.move_camera(dpitch= cfg.cam_pitch_step)
            if p.B3G_DOWN_ARROW  in keys: env.move_camera(dpitch=-cfg.cam_pitch_step)

            dpan = np.zeros(3, dtype=np.float32)
            if ord('i') in keys: dpan[1] += cfg.cam_pan_step
            if ord('k') in keys: dpan[1] -= cfg.cam_pan_step
            if ord('j') in keys: dpan[0] -= cfg.cam_pan_step
            if ord('l') in keys: dpan[0] += cfg.cam_pan_step
            if ord('u') in keys: dpan[2] += cfg.cam_pan_step
            if ord('o') in keys: dpan[2] -= cfg.cam_pan_step
            if np.linalg.norm(dpan) > 0: env.move_camera(dtarget=dpan)

            if ord('t') in keys and keys[ord('t')] & p.KEY_WAS_TRIGGERED:
                env.set_follow(lead); print(f"Follow: {lead}")
            for i, k in enumerate([ord('1'),ord('2'),ord('3'),ord('4'),ord('5'),ord('6'),ord('7'),ord('8'),ord('9')]):
                if k in keys and keys[k] & p.KEY_WAS_TRIGGERED:
                    name = f"def_{i}"; env.set_follow(name); print(f"Follow: {name}")
            if ord('0') in keys and keys[ord('0')] & p.KEY_WAS_TRIGGERED:
                env.set_follow(None); print("Follow: off")

            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                env.set_follow(None)
                env.set_camera(target=[cfg.base_pos[0], cfg.base_pos[1], cfg.cam_target_z])
                print("Camera: centered on base")
            if ord('g') in keys and keys[ord('g')] & p.KEY_WAS_TRIGGERED:
                env.set_follow(None)
                cen = defenders_centroid(state)
                env.set_camera(target=[cen[0], cen[1], cfg.cam_target_z])
                print("Camera: centered on defenders")

            # -------- step the sim --------
            state, events = env.step({ state["defenders"][0]["name"]: a })

            if events["near_base"]:
                print(f"[END] Step {state['step']}: attacker NEAR base "
                      f"(nearest_dist={events['nearest_dist']:.2f} m)")
                break

            if events["base_breached"]:
                print(f"[END] Step {state['step']}: BASE BREACHED")
                break

            if events["game_over"]:
                break
    finally:
        env.close()
