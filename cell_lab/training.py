from __future__ import annotations

import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .env import CellEnv
from .policies import heuristic_action

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError:  # pragma: no cover
    PPO = None
    SAC = None
    CallbackList = None
    CheckpointCallback = None
    EvalCallback = None
    Monitor = None


def train_agent(
    algorithm: str = "ppo",
    total_timesteps: int = 100_000,
    model_dir: str = "models",
    log_dir: str = "train_logs",
    run_name: str | None = None,
    resume_from: str | None = None,
    eval_freq: int = 10_000,
    checkpoint_freq: int = 10_000,
) -> Path:
    algo = algorithm.lower()
    if algo == "native":
        return train_native_model(total_timesteps=total_timesteps, model_dir=model_dir)

    if PPO is None or SAC is None or EvalCallback is None or CheckpointCallback is None or CallbackList is None:
        raise RuntimeError(
            "RL dependencies are missing for PPO/SAC. "
            "Use `python main.py train --algorithm native --timesteps 5000` on this machine."
        )

    run_id = run_name or f"{algo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_root = Path(model_dir)
    log_root = Path(log_dir)
    run_model_dir = model_root / run_id
    run_log_dir = log_root / run_id
    run_model_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir.mkdir(parents=True, exist_ok=True)

    env = CellEnv(max_steps=400, seed=42)
    eval_env = CellEnv(max_steps=400, seed=4242)
    if Monitor is not None:
        eval_env = Monitor(eval_env)

    if algo == "ppo":
        model_cls = PPO
    elif algo == "sac":
        model_cls = SAC
    else:
        raise ValueError("algorithm must be one of: native, ppo, sac")

    model_kwargs: dict[str, Any] = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 1,
        "tensorboard_log": str(run_log_dir / "tensorboard"),
    }

    if resume_from:
        model = model_cls.load(resume_from, env=env, tensorboard_log=str(run_log_dir / "tensorboard"))
    else:
        model = model_cls(**model_kwargs)

    best_model_dir = run_model_dir / "best_model"
    best_model_dir.mkdir(exist_ok=True)
    checkpoints_dir = run_model_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    eval_dir = run_log_dir / "eval"
    eval_dir.mkdir(exist_ok=True)

    callbacks = CallbackList(
        [
            EvalCallback(
                eval_env,
                best_model_save_path=str(best_model_dir),
                log_path=str(eval_dir),
                eval_freq=max(1, eval_freq),
                deterministic=True,
                render=False,
            ),
            CheckpointCallback(
                save_freq=max(1, checkpoint_freq),
                save_path=str(checkpoints_dir),
                name_prefix=f"{algo}_checkpoint",
                save_replay_buffer=(algo == "sac"),
                save_vecnormalize=False,
            ),
        ]
    )

    final_output = run_model_dir / f"cell_lab_{algo}_final"
    latest_output = model_root / f"cell_lab_{algo}"
    interrupted_output = run_model_dir / f"cell_lab_{algo}_interrupted"

    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=False, callback=callbacks)
        model.save(final_output)
        model.save(latest_output)
    except KeyboardInterrupt:
        model.save(interrupted_output)
        model.save(latest_output)
        raise

    _write_run_summary(
        run_log_dir / "summary.json",
        {
            "algorithm": algo,
            "run_id": run_id,
            "total_timesteps": total_timesteps,
            "final_model": str(final_output.with_suffix(".zip")),
            "latest_model": str(latest_output.with_suffix(".zip")),
            "best_model_dir": str(best_model_dir),
            "eval_dir": str(eval_dir),
            "tensorboard_dir": str(run_log_dir / "tensorboard"),
            "checkpoints_dir": str(checkpoints_dir),
            "resumed_from": resume_from,
        },
    )
    return final_output.with_suffix(".zip")


def train_native_model(total_timesteps: int = 5_000, model_dir: str = "models") -> Path:
    dataset = _collect_heuristic_dataset(sample_steps=total_timesteps)
    weights, biases = _fit_linear_clone(dataset, epochs=18, learning_rate=0.45)

    model_path = Path(model_dir)
    model_path.mkdir(exist_ok=True)
    output = model_path / "cell_lab_native_model.json"
    payload = {
        "type": "native_linear_v1",
        "observation_size": len(dataset[0][0]),
        "action_size": len(dataset[0][1]),
        "trained_from": "heuristic imitation",
        "samples": len(dataset),
        "weights": weights,
        "biases": biases,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def _collect_heuristic_dataset(sample_steps: int) -> list[tuple[list[float], list[float]]]:
    env = CellEnv(max_steps=500, seed=7)
    observation, _ = env.reset(seed=7)
    dataset: list[tuple[list[float], list[float]]] = []
    simulator = env.simulator

    for _ in range(max(200, sample_steps)):
        action = heuristic_action(simulator)
        dataset.append(([float(value) for value in observation], [float(value) for value in action]))
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            reseed = random.randint(0, 10_000)
            observation, _ = env.reset(seed=reseed)
            simulator = env.simulator
    return dataset


def _fit_linear_clone(
    dataset: Sequence[tuple[Sequence[float], Sequence[float]]],
    epochs: int = 12,
    learning_rate: float = 0.35,
) -> tuple[list[list[float]], list[float]]:
    observation_size = len(dataset[0][0])
    action_size = len(dataset[0][1])
    rng = random.Random(42)
    weights = [[rng.uniform(-0.05, 0.05) for _ in range(observation_size)] for _ in range(action_size)]
    biases = [0.0 for _ in range(action_size)]

    for _ in range(epochs):
        for observation, target_actions in dataset:
            for action_index in range(action_size):
                z_value = biases[action_index]
                for feature_index, value in enumerate(observation):
                    z_value += weights[action_index][feature_index] * float(value)
                prediction = _sigmoid(z_value)
                error = prediction - float(target_actions[action_index])
                gradient = error * prediction * (1.0 - prediction)
                for feature_index, value in enumerate(observation):
                    weights[action_index][feature_index] -= learning_rate * gradient * float(value)
                biases[action_index] -= learning_rate * gradient

    return weights, biases


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _write_run_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
