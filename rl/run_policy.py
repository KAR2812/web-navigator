# rl/run_policy.py
from stable_baselines3 import PPO
import numpy as np
import time

MODEL_PATH = "rl/models/ppo_selenium"

def _ensure_array(x):
    """Ensure x is a numpy array. If it's a list/tuple with a single element, unwrap it."""
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x)
    except Exception:
        if isinstance(x, (list, tuple)) and len(x) == 1:
            return np.asarray(x[0])
        raise

def main():
    # Create the raw env and load the model (model will wrap the env internally)
    # Keep this code as you had it so the model wraps the same env instance:
    import rl.envs.selenium_env as env_mod  # ensure package import works
    raw_env = env_mod.SeleniumNavEnv(start_url="http://localhost:8000/train_page.html", max_steps=120, headless=False)

    # Load the model and tell it about the env (so SB3 wraps it consistently)
    model = PPO.load(MODEL_PATH, env=raw_env)

    # Use the env that the model actually has (this may be a VecEnv wrapper)
    vec_env = getattr(model, "env", None)
    if vec_env is None:
        # fallback to the raw env
        vec_env = raw_env

    # Reset using the model's env so returned obs matches what model.predict expects
    reset_ret = vec_env.reset()
    # Support both Gymnasium (obs, info) and classic (obs)
    if isinstance(reset_ret, tuple) and len(reset_ret) == 2:
        obs, info = reset_ret
    else:
        obs = reset_ret

    obs = _ensure_array(obs)

    total = 0.0
    done = False

    while True:
        # model.predict expects the observation in the same format the model's env returns
        action, _ = model.predict(obs, deterministic=True)

        step_ret = vec_env.step(action)

        # step_ret may be:
        # - old Gym: (obs, reward, done, info)  -> len==4
        # - Gymnasium: (obs, reward, terminated, truncated, info) -> len==5
        if len(step_ret) == 4:
            next_obs, reward, done_arr, info = step_ret
            terminated = done_arr
            truncated = False
        elif len(step_ret) == 5:
            next_obs, reward, terminated, truncated, info = step_ret
        else:
            raise ValueError(f"Unexpected env.step() return length: {len(step_ret)}")

        # If vectorized, rewards/terminated may be arrays; take index 0 for single-env run
        # Support both scalar and array-like
        if isinstance(reward, (list, tuple, np.ndarray)):
            reward_val = float(np.asarray(reward).ravel()[0])
        else:
            reward_val = float(reward)

        total += reward_val

        # Render: try to render the underlying raw env if vec_env wraps envs
        try:
            if hasattr(vec_env, "envs") and len(getattr(vec_env, "envs")) > 0:
                # render the first underlying env
                vec_env.envs[0].render()
            else:
                # raw env or custom wrapper
                vec_env.render()
        except Exception:
            # non-fatal: rendering may not be supported in headless mode
            pass

        time.sleep(0.3)

        # Determine done flag (handle arrays / scalars)
        def _to_bool(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                return bool(np.asarray(x).ravel()[0])
            return bool(x)

        done_flag = _to_bool(terminated) or _to_bool(truncated)

        if done_flag:
            print("Episode done. Total reward:", total)
            # reset env for next episode (and prepare obs for potential next iteration)
            reset_ret = vec_env.reset()
            if isinstance(reset_ret, tuple) and len(reset_ret) == 2:
                obs, info = reset_ret
            else:
                obs = reset_ret
            obs = _ensure_array(obs)
            # break if you only wanted a single episode:
            break

        # prepare obs for next loop
        if isinstance(next_obs, tuple) and len(next_obs) == 2:
            obs, _ = next_obs
        else:
            obs = next_obs
        obs = _ensure_array(obs)

    # Cleanup
    try:
        # attempt to close the wrapped envs properly
        if hasattr(vec_env, "close"):
            vec_env.close()
        else:
            raw_env.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
