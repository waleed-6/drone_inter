# eval_ppo_defense.py
# Evaluate a trained PPO policy in the GUI.
# Uses SB3 VecEnv API: reset()->obs, step()->(obs, rewards, dones, infos)

import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# VecNormalize is optional; we try to load stats if present
_USE_VECNORM = True
try:
    from stable_baselines3.common.vec_env import VecNormalize
except Exception:
    _USE_VECNORM = False
    VecNormalize = None  # type: ignore

from drone_env.config import EnvConfig
from drone_env.gym_wrapper import DefenderRLWrapper

def make_env_gui(seed: int = 7):
    def _f():
        cfg = EnvConfig()
        # Restore your business rules for viewing, if desired
        cfg.max_steps = 1000
        cfg.end_on_perimeter_near = True
        cfg.end_on_near = True
        cfg.end_on_breach = True
        return DefenderRLWrapper(cfg=cfg, render_mode="human", seed=seed)
    return _f

if __name__ == "__main__":
    # Build a single-env VecEnv so SB3 model API works
    venv = DummyVecEnv([make_env_gui(7)])

    # Try to load VecNormalize stats (if you saved them during training)
    if _USE_VECNORM:
        try:
            venv = VecNormalize.load("models/vecnorm.pkl", venv)
            venv.training = False
            venv.norm_reward = False
            print("[info] loaded VecNormalize stats")
        except Exception as e:
            print("[warn] could not load VecNormalize stats:", e)
            print("[warn] continuing without observation normalization")

    # Load the trained model
    model = PPO.load("models/ppo_defense_ckpt_150000_steps", device="cpu")

    # --- VecEnv API: reset returns ONLY obs ---
    obs = venv.reset()

    while True:
        # SB3 wants batched obs; DummyVecEnv already provides shape (n_envs, obs_dim)
        action, _ = model.predict(obs, deterministic=True)

        # --- VecEnv API: step returns 4 values ---
        obs, rewards, dones, infos = venv.step(action)

        # When done, VecEnv expects you to reset that env
        if dones[0]:
            obs = venv.reset()

        # Let the GUI breathe (your env also sleeps dt internally)
        time.sleep(0.01)
