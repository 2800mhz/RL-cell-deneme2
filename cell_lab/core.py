from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, List, Sequence, Tuple


@dataclass
class CellState:
    atp: float = 70.0
    glucose: float = 60.0
    oxygen: float = 60.0
    amino_acids: float = 55.0
    waste: float = 8.0
    ros: float = 6.0
    dna_damage: float = 4.0
    membrane_damage: float = 3.0
    size: float = 1.0
    age: int = 0
    division_progress: float = 0.0
    mitochondria_efficiency: float = 72.0
    ribosome_capacity: float = 68.0
    nucleus_stability: float = 78.0
    membrane_integrity: float = 82.0
    lysosome_activity: float = 64.0
    er_stability: float = 70.0
    golgi_capacity: float = 66.0
    alive: bool = True


@dataclass
class StepResult:
    observation: Tuple[float, ...]
    reward: float
    done: bool
    info: Dict[str, float]


class CellSimulator:
    observation_size = 19
    action_names = [
        "nutrient_uptake",
        "oxygen_uptake",
        "glycolysis",
        "respiration",
        "protein_synthesis",
        "repair",
        "detox",
        "division_focus",
    ]

    def __init__(self, max_steps: int = 500, seed: int | None = None) -> None:
        self.max_steps = max_steps
        self.rng = Random(seed)
        self.history: List[Dict[str, float]] = []
        self.state = CellState()
        self.steps = 0

    def reset(self) -> Tuple[float, ...]:
        self.state = CellState()
        self.steps = 0
        self.history = []
        self._record()
        return self.get_observation()

    def get_observation(self) -> Tuple[float, ...]:
        s = self.state
        return (
            s.atp / 100.0,
            s.glucose / 100.0,
            s.oxygen / 100.0,
            s.amino_acids / 100.0,
            s.waste / 100.0,
            s.ros / 100.0,
            s.dna_damage / 100.0,
            s.membrane_damage / 100.0,
            min(s.size / 2.5, 1.0),
            min(s.division_progress / 100.0, 1.0),
            min(s.age / max(self.max_steps, 1), 1.0),
            s.mitochondria_efficiency / 100.0,
            s.ribosome_capacity / 100.0,
            s.nucleus_stability / 100.0,
            s.membrane_integrity / 100.0,
            s.lysosome_activity / 100.0,
            s.er_stability / 100.0,
            s.golgi_capacity / 100.0,
            1.0 if s.alive else 0.0,
        )

    def step(self, action: Sequence[float]) -> StepResult:
        if len(action) != len(self.action_names):
            raise ValueError(f"Expected {len(self.action_names)} actions, got {len(action)}")
        action_arr = [self._clip(float(value), 0.0, 1.0) for value in action]

        s = self.state
        if not s.alive:
            return StepResult(self.get_observation(), -100.0, True, self.get_metrics())

        self.steps += 1
        s.age += 1

        uptake_nutrients, uptake_oxygen, glycolysis, respiration, synthesis, repair, detox, division = action_arr
        membrane_factor = 0.65 + 0.35 * (s.membrane_integrity / 100.0)
        mitochondria_factor = 0.55 + 0.45 * (s.mitochondria_efficiency / 100.0)
        ribosome_factor = 0.45 + 0.55 * (s.ribosome_capacity / 100.0)
        nucleus_factor = 0.55 + 0.45 * (s.nucleus_stability / 100.0)
        lysosome_factor = 0.45 + 0.55 * (s.lysosome_activity / 100.0)
        er_factor = 0.50 + 0.50 * (s.er_stability / 100.0)
        golgi_factor = 0.50 + 0.50 * (s.golgi_capacity / 100.0)

        # Resource intake
        s.glucose = min(100.0, s.glucose + 8.0 * uptake_nutrients * membrane_factor)
        s.oxygen = min(100.0, s.oxygen + 5.0 * uptake_oxygen * membrane_factor)
        s.amino_acids = min(100.0, s.amino_acids + 5.0 * uptake_nutrients * membrane_factor)

        # Baseline metabolism
        maintenance = 2.2 + 0.65 * s.size + 0.04 * (s.dna_damage + s.membrane_damage) + (100.0 - s.er_stability) * 0.008
        s.atp -= maintenance
        s.glucose -= 0.8
        s.oxygen -= 0.6
        s.waste += 0.8

        # Energy pathways
        glycolysis_flux = min(s.glucose, 7.0 * glycolysis)
        s.glucose -= glycolysis_flux
        s.atp += glycolysis_flux * 1.8
        s.waste += glycolysis_flux * 0.4

        respiration_flux = min(s.glucose, s.oxygen, 6.0 * respiration)
        s.glucose -= respiration_flux
        s.oxygen -= respiration_flux
        s.atp += respiration_flux * 3.6 * mitochondria_factor
        s.ros += respiration_flux * (0.28 + (1.0 - mitochondria_factor) * 0.45)
        s.waste += respiration_flux * 0.25

        # Building, repair, detox
        synthesis_flux = min(s.atp / 6.5, s.amino_acids, 4.0 * synthesis * ribosome_factor * er_factor)
        s.atp -= synthesis_flux * 3.8
        s.amino_acids -= synthesis_flux
        s.size += synthesis_flux * 0.012 * golgi_factor
        s.division_progress += synthesis_flux * 1.3 * nucleus_factor * golgi_factor
        s.er_stability -= synthesis_flux * 0.38
        s.golgi_capacity -= synthesis_flux * 0.16

        repair_flux = min(s.atp / 5.0, 3.0 * repair * nucleus_factor)
        s.atp -= repair_flux * 3.0
        s.dna_damage = max(0.0, s.dna_damage - repair_flux * (1.4 + nucleus_factor))
        s.membrane_damage = max(0.0, s.membrane_damage - repair_flux * (0.9 + membrane_factor))
        s.nucleus_stability += repair_flux * 0.9
        s.membrane_integrity += repair_flux * 0.6

        detox_flux = min(s.atp / 4.0, 3.0 * detox * lysosome_factor)
        s.atp -= detox_flux * 2.5
        s.ros = max(0.0, s.ros - detox_flux * (1.4 + 0.9 * mitochondria_factor))
        s.waste = max(0.0, s.waste - detox_flux * (1.0 + lysosome_factor))
        s.lysosome_activity += detox_flux * 0.5

        # Division focus is useful only when the cell is healthy.
        if s.size > 1.2 and s.atp > 35 and s.dna_damage < 20 and s.nucleus_stability > 55:
            s.division_progress += 1.8 * division * nucleus_factor
        else:
            s.division_progress -= 0.6 * division

        # Passive drift and stress
        deficiency = max(0.0, 18.0 - s.atp) * 0.14
        starvation = max(0.0, 12.0 - s.glucose) * 0.10
        hypoxia = max(0.0, 10.0 - s.oxygen) * 0.16
        toxicity = max(0.0, s.waste - 45.0) * 0.08 + max(0.0, s.ros - 30.0) * 0.16
        protein_stress = max(0.0, 60.0 - s.er_stability) * 0.05

        s.dna_damage += deficiency + 0.5 * toxicity + protein_stress + self.rng.uniform(0.0, 0.15)
        s.membrane_damage += starvation + hypoxia + 0.35 * toxicity + max(0.0, 55.0 - s.membrane_integrity) * 0.03
        s.ros += 0.25 + 0.3 * hypoxia
        s.waste += 0.15 + 0.1 * starvation + max(0.0, 55.0 - s.golgi_capacity) * 0.03

        # Organel dynamics
        s.mitochondria_efficiency += respiration * 0.25 - max(0.0, s.ros - 24.0) * 0.06 - hypoxia * 0.2
        s.ribosome_capacity += synthesis * 0.18 - protein_stress * 0.9 - deficiency * 0.15
        s.nucleus_stability += repair * 0.35 - s.dna_damage * 0.035
        s.membrane_integrity += uptake_nutrients * 0.10 - s.membrane_damage * 0.045 - toxicity * 0.08
        s.lysosome_activity += detox * 0.22 - max(0.0, s.waste - 40.0) * 0.04
        s.er_stability += repair * 0.12 - synthesis * 0.28 - max(0.0, s.ros - 25.0) * 0.05
        s.golgi_capacity += synthesis * 0.08 - max(0.0, s.waste - 35.0) * 0.03

        s.atp = self._clip(s.atp, 0.0, 120.0)
        s.glucose = self._clip(s.glucose, 0.0, 100.0)
        s.oxygen = self._clip(s.oxygen, 0.0, 100.0)
        s.amino_acids = self._clip(s.amino_acids, 0.0, 100.0)
        s.waste = self._clip(s.waste, 0.0, 100.0)
        s.ros = self._clip(s.ros, 0.0, 100.0)
        s.dna_damage = self._clip(s.dna_damage, 0.0, 100.0)
        s.membrane_damage = self._clip(s.membrane_damage, 0.0, 100.0)
        s.size = self._clip(s.size, 0.7, 2.5)
        s.division_progress = self._clip(s.division_progress, 0.0, 100.0)
        s.mitochondria_efficiency = self._clip(s.mitochondria_efficiency, 0.0, 100.0)
        s.ribosome_capacity = self._clip(s.ribosome_capacity, 0.0, 100.0)
        s.nucleus_stability = self._clip(s.nucleus_stability, 0.0, 100.0)
        s.membrane_integrity = self._clip(s.membrane_integrity, 0.0, 100.0)
        s.lysosome_activity = self._clip(s.lysosome_activity, 0.0, 100.0)
        s.er_stability = self._clip(s.er_stability, 0.0, 100.0)
        s.golgi_capacity = self._clip(s.golgi_capacity, 0.0, 100.0)

        done = self._check_done()
        reward = self._calculate_reward()
        info = self.get_metrics()
        self._record()
        return StepResult(self.get_observation(), reward, done, info)

    def _calculate_reward(self) -> float:
        s = self.state
        if not s.alive:
            return -120.0

        energy_term = -abs(s.atp - 55.0) / 25.0 + 2.0
        growth_term = (s.size - 1.0) * 6.0
        health_term = 2.0 - (s.dna_damage + s.membrane_damage) / 35.0
        cleanup_term = 1.2 - (s.waste + s.ros) / 60.0
        division_term = s.division_progress / 60.0
        organelle_term = (
            s.mitochondria_efficiency
            + s.ribosome_capacity
            + s.nucleus_stability
            + s.membrane_integrity
            + s.lysosome_activity
            + s.er_stability
            + s.golgi_capacity
        ) / 280.0
        survival_bonus = 1.0
        return float(energy_term + growth_term + health_term + cleanup_term + division_term + organelle_term + survival_bonus)

    def _check_done(self) -> bool:
        s = self.state
        if s.atp <= 0.0 or s.dna_damage >= 100.0 or s.membrane_damage >= 100.0:
            s.alive = False
            return True
        if s.waste >= 100.0 or s.ros >= 100.0:
            s.alive = False
            return True
        if self.steps >= self.max_steps:
            return True
        return False

    def ready_to_divide(self) -> bool:
        s = self.state
        return (
            s.alive
            and s.size >= 1.8
            and s.division_progress >= 90.0
            and s.dna_damage < 12.0
            and s.nucleus_stability > 65.0
            and s.membrane_integrity > 60.0
        )

    def get_metrics(self) -> Dict[str, float]:
        s = self.state
        return {
            "atp": s.atp,
            "glucose": s.glucose,
            "oxygen": s.oxygen,
            "amino_acids": s.amino_acids,
            "waste": s.waste,
            "ros": s.ros,
            "dna_damage": s.dna_damage,
            "membrane_damage": s.membrane_damage,
            "size": s.size,
            "age": float(s.age),
            "division_progress": s.division_progress,
            "mitochondria_efficiency": s.mitochondria_efficiency,
            "ribosome_capacity": s.ribosome_capacity,
            "nucleus_stability": s.nucleus_stability,
            "membrane_integrity": s.membrane_integrity,
            "lysosome_activity": s.lysosome_activity,
            "er_stability": s.er_stability,
            "golgi_capacity": s.golgi_capacity,
            "alive": 1.0 if s.alive else 0.0,
            "ready_to_divide": 1.0 if self.ready_to_divide() else 0.0,
        }

    def summary(self) -> str:
        s = self.state
        status = "alive" if s.alive else "dead"
        return (
            f"Cell is {status} | step={self.steps}/{self.max_steps} | "
            f"ATP={s.atp:.1f} | glucose={s.glucose:.1f} | oxygen={s.oxygen:.1f} | "
            f"damage={s.dna_damage + s.membrane_damage:.1f} | size={s.size:.2f} | "
            f"division={s.division_progress:.1f} | mito={s.mitochondria_efficiency:.1f} | "
            f"nucleus={s.nucleus_stability:.1f}"
        )

    def _record(self) -> None:
        metrics = self.get_metrics()
        metrics["step"] = float(self.steps)
        self.history.append(metrics)

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
