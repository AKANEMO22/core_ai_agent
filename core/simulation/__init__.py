"""Simulation harness and quality scoring for release gates."""

from core.simulation.harness import CandidatePersona, SimulationResult, SimulationRunner
from core.simulation.quality import QualityReport, cohen_kappa, mean_absolute_error, score

__all__ = [
    "CandidatePersona",
    "QualityReport",
    "SimulationResult",
    "SimulationRunner",
    "cohen_kappa",
    "mean_absolute_error",
    "score",
]
