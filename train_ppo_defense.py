# train_ppo_defense.py
# PPO training for defender control (single agent controls ALL defenders).
# Run:  python3 train_ppo_defense.py

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from drone_env.config import EnvConfig
from drone_env.gym_wrapper import DefenderRLWrapper
import torch
def make_env(seed: int = 0):
    def _f():
        cfg = EnvConfig()
        # For training stability, reduce early termination & keep episodes long
        cfg.max_steps = 1200
        cfg.end_on_perimeter_near = False
        cfg.end_on_near = False
        cfg.end_on_breach = True
        # Slight randomness (optional): vary attacker speed a little
        # You can add domain randomization here if desired.
        env = DefenderRLWrapper(cfg=cfg, render_mode="none", seed=seed)
        return env
    return _f

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    seed = 42

    # Single-process vector env (easy to start; you can parallelize later)
    env = DummyVecEnv([make_env(seed)])
    # Normalize observations (and optionally rewards)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    policy_kwargs = dict(
        net_arch=[256, 256],
        activation_fn=torch.nn.Tanh
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device="cpu",  # macOS CPU is fine; MPS is optional with recent PyTorch
    )

    # save checkpoints every ~N steps
    ckpt_cb = CheckpointCallback(save_freq=10_000, save_path="models", name_prefix="ppo_defense_ckpt")

    total_timesteps = 50_000  # start here; increase if you want stronger policies
    model.learn(total_timesteps=total_timesteps, callback=[ckpt_cb])

    # Save policy and VecNormalize stats
    model.save("models/ppo_defense_final")
    env.save("models/vecnorm.pkl")

    # Clean up
    env.close()
    print("Training done. Saved to models/ppo_defense_final and models/vecnorm.pkl")
