from __future__ import annotations

import argparse

from .core import CellSimulator
from .plotting import run_and_plot
from .policies import ModelPolicy, heuristic_action
from .training import train_agent
from .ui import launch_ui


def _simulate(mode: str, steps: int, model_path: str | None) -> None:
    simulator = CellSimulator(max_steps=steps)
    observation = simulator.reset()
    policy = ModelPolicy(model_path) if mode == "model" and model_path else None

    while simulator.steps < simulator.max_steps and simulator.state.alive:
        if mode == "manual":
            action = [0.5] * 8
        elif mode == "model" and policy is not None:
            action = policy.predict(observation)
        else:
            action = heuristic_action(simulator)
        result = simulator.step(action)
        observation = result.observation

    print(simulator.summary())
    print("Ready to divide:", simulator.ready_to_divide())


def main() -> None:
    parser = argparse.ArgumentParser(description="Cell Lab")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("ui", help="Launch the desktop UI")

    simulate_parser = subparsers.add_parser("simulate", help="Run a terminal simulation")
    simulate_parser.add_argument("--mode", choices=["manual", "heuristic", "model"], default="heuristic")
    simulate_parser.add_argument("--steps", type=int, default=300)
    simulate_parser.add_argument("--model-path")

    plot_parser = subparsers.add_parser("plot", help="Run a simulation and save a plot")
    plot_parser.add_argument("--mode", choices=["manual", "heuristic", "model"], default="heuristic")
    plot_parser.add_argument("--steps", type=int, default=300)
    plot_parser.add_argument("--model-path")

    train_parser = subparsers.add_parser("train", help="Train an RL agent")
    train_parser.add_argument("--algorithm", choices=["ppo", "sac"], default="ppo")
    train_parser.add_argument("--timesteps", type=int, default=100_000)

    args = parser.parse_args()

    if args.command == "ui":
        launch_ui()
    elif args.command == "simulate":
        _simulate(args.mode, args.steps, args.model_path)
    elif args.command == "plot":
        path = run_and_plot(mode=args.mode, steps=args.steps, model_path=args.model_path)
        print(f"Plot saved to {path}")
    elif args.command == "train":
        path = train_agent(algorithm=args.algorithm, total_timesteps=args.timesteps)
        print(f"Model saved to {path}")
    else:
        parser.print_help()
