"""Công cụ chạy mô phỏng và đo chất lượng đầu ra.

``harness`` chạy persona giả lập qua app thật để kiểm tra flow offline;
``live`` chạy smoke test có LLM thật bằng ``RUN_LIVE_LLM=1 uv run python -m core.simulation.live``;
``quality`` so sánh decision đã lưu với golden set bằng
``uv run python -m core.simulation.quality --golden <jsonl> --decisions-dir <workspaces>``.
"""

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
