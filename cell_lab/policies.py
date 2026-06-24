from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .core import CellSimulator

try:
    from stable_baselines3 import PPO, SAC
except ImportError:  # pragma: no cover
    PPO = None
    SAC = None


def heuristic_action(simulator: CellSimulator) -> list[float]:
    s = simulator.state
    action = [0.0] * 8

    damage = s.dna_damage + s.membrane_damage
    stress = s.ros + s.waste

    action[0] = 0.98 if s.glucose < 70 or s.amino_acids < 55 or s.atp < 40 or s.membrane_integrity < 68 else 0.55
    action[1] = 0.92 if s.oxygen < 65 or s.atp < 48 else 0.45

    if s.atp < 35:
        if s.oxygen > 28:
            action[2] = 0.34
            action[3] = 0.92
        else:
            action[2] = 0.88
            action[3] = 0.18
    elif s.atp < 55:
        if s.oxygen > 22:
            action[2] = 0.28
            action[3] = 0.72
        else:
            action[2] = 0.62
            action[3] = 0.14
    else:
        action[2] = 0.18
        action[3] = 0.24

    synthesis_ready = s.ribosome_capacity > 45 and s.er_stability > 45 and s.golgi_capacity > 45
    action[4] = 0.58 if s.atp > 52 and damage < 18 and stress < 58 and s.amino_acids > 35 and synthesis_ready else 0.10
    action[5] = 0.92 if damage > 14 or s.nucleus_stability < 55 else 0.35 if damage > 7 else 0.08
    action[6] = 0.92 if s.ros > 14 or s.waste > 26 or s.lysosome_activity < 45 else 0.40 if stress > 28 else 0.12

    if (
        s.size > 1.55
        and s.atp > 62
        and damage < 10
        and s.waste < 18
        and s.ros < 15
        and s.nucleus_stability > 65
        and s.membrane_integrity > 60
    ):
        action[7] = 0.65
    else:
        action[7] = 0.02

    return [max(0.0, min(1.0, value)) for value in action]


class ModelPolicy:
    def __init__(self, model_path: str | Path) -> None:
        if PPO is None and SAC is None:
            raise RuntimeError("stable-baselines3 is not installed. Install optional RL dependencies first.")

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(path)

        last_error: Optional[Exception] = None
        for loader in (PPO, SAC):
            if loader is None:
                continue
            try:
                self.model = loader.load(path)
                return
            except Exception as exc:  # pragma: no cover
                last_error = exc
        raise RuntimeError(f"Could not load model from {path}") from last_error

    def predict(self, observation: Sequence[float]) -> list[float]:
        action, _ = self.model.predict(observation, deterministic=True)
        return [float(value) for value in action]
