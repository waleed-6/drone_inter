import numpy as np
from drone_env.config import EnvConfig
from drone_env.environment import DroneEnvironment

if __name__ == "__main__":
    cfg = EnvConfig()
    env = DroneEnvironment(cfg, render_mode="human", seed=42)
    state = env.reset()
    print("Defenders:", [d["name"] for d in state["defenders"]],
          "Attackers:", [a["name"] for a in state["attackers"]])

    try:
        while True:
            actions = { d["name"]: np.random.uniform(-1,1,size=3).astype("float32")
                        for d in state["defenders"] }
            state, events = env.step(actions)

            if events["near_perimeter"]:
                print(f"[END] step {state['step']}: attacker NEAR PERIMETER "
                      f"(outside_dist={events['perimeter_nearest']:.2f} m)")
                break
            if events["near_base"]:
                print(f"[END] step {state['step']}: attacker NEAR CORE "
                      f"(nearest_dist={events['nearest_dist']:.2f} m)")
                break
            if events["base_breached"]:
                print(f"[END] step {state['step']}: BASE BREACHED")
                break
            if events["game_over"] or state["step"] >= cfg.max_steps:
                print("[END] game over or max_steps"); break
    finally:
        env.close()
