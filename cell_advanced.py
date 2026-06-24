"""Legacy compatibility exports for the rebuilt project."""

from cell_lab.core import CellSimulator as AdvancedCell
from cell_lab.core import CellState, StepResult

__all__ = ["AdvancedCell", "CellState", "StepResult"]
