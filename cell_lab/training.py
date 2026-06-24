from __future__ import annotations

from pathlib import Path

from .env import CellEnv

try:
    from stable_baselines3 import PPO, SAC
except ImportError:  # pragma: no cover
    PPO = None
    SAC = None


def train_agent(algorithm: str = "ppo", total_timesteps: int = 100_000, model_dir: str = "models") -> Path:
    if PPO is None or SAC is None:
        raise RuntimeError("RL dependencies are missing. Install gymnasium, stable-baselines3, and torch first.")

    env = CellEnv(max_steps=400)
    algo = algorithm.lower()
    if algo == "ppo":
        model = PPO("MlpPolicy", env, verbose=1)
    elif algo == "sac":
        model = SAC("MlpPolicy", env, verbose=1)
    else:
        raise ValueError("algorithm must be one of: ppo, sac")

    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    model_path = Path(model_dir)
    model_path.mkdir(exist_ok=True)
    output = model_path / f"cell_lab_{algo}"
    model.save(output)
    return output.with_suffix(".zip")
