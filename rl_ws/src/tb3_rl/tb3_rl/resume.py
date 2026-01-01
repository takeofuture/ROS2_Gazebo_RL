# tb3_rl/resume.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from .env import TB3ArenaEnv

def main():
    env = DummyVecEnv([lambda: TB3ArenaEnv()])

    ckpt = "checkpoints/ppo_tb3_arena_12000_steps.zip"  # ←再開したいzip
    model = PPO.load(ckpt, env=env)

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000,
        save_path="checkpoints",
        name_prefix="ppo_tb3_arena"
    )

    # 追加で学習（例：さらに50,000ステップ）
    model.learn(total_timesteps=50_000, callback=checkpoint_callback)

    model.save("ppo_tb3_arena_final")

if __name__ == "__main__":
    main()
