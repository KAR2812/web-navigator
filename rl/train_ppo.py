# rl/train_ppo.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.selenium_env import SeleniumNavEnv

def make_env():
    # change to your local training page (see rl_train_site)
    start_url = "http://localhost:8000/train_page.html"
    return SeleniumNavEnv(start_url=start_url, max_steps=80, headless=True)

def main():
    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./rl/logs/ppo_selenium/")
    # Train for a modest number of timesteps for testing:
    model.learn(total_timesteps=20000)
    os.makedirs("rl/models", exist_ok=True)
    model.save("rl/models/ppo_selenium")
    print("Saved model to rl/models/ppo_selenium")

if __name__ == "__main__":
    main()
