# tb3_rl/train.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from .env import TB3ArenaEnv

def main():
    env = DummyVecEnv([lambda: TB3ArenaEnv()])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
    )

    os.makedirs("checkpoints", exist_ok=True)

    # 例：5000ステップごとに保存（好きに調整）
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000,
        save_path="checkpoints",
        name_prefix="ppo_tb3_arena"
    )

    model.learn(total_timesteps=50_000, callback=checkpoint_callback)

    # 最後も念のため保存
    model.save("ppo_tb3_arena_final")

if __name__ == "__main__":
    main()
