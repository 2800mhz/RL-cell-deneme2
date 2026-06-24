from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from .core import CellSimulator

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None


class CellEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps: int = 500, seed: int | None = None) -> None:
        self.simulator = CellSimulator(max_steps=max_steps, seed=seed)
        self.max_steps = max_steps
        self.render_mode = "human"
        if spaces is not None:
            try:
                import numpy as np

                self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(CellSimulator.observation_size,), dtype=np.float32)
                self.action_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
            except ImportError:  # pragma: no cover
                self.observation_space = None
                self.action_space = None

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[Tuple[float, ...], Dict[str, Any]]:
        if seed is not None:
            self.simulator = CellSimulator(max_steps=self.max_steps, seed=seed)
        observation = self.simulator.reset()
        return observation, self.simulator.get_metrics()

    def step(self, action: Sequence[float]) -> Tuple[Tuple[float, ...], float, bool, bool, Dict[str, Any]]:
        result = self.simulator.step(action)
        truncated = self.simulator.steps >= self.max_steps and self.simulator.state.alive
        return result.observation, result.reward, result.done, truncated, result.info

    def render(self) -> None:
        print(self.simulator.summary())
