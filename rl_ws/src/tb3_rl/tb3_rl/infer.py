# tb3_rl/infer.py
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .env import TB3ArenaEnv


def main():
    env = DummyVecEnv([lambda: TB3ArenaEnv()])

    model = PPO.load("checkpoints/ppo_tb3_arena_12000_steps.zip", env=env)

    obs = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Episode finished → reset")
            obs = env.reset()

        time.sleep(0.05)  # 動きが速すぎないように


if __name__ == "__main__":
    main()
