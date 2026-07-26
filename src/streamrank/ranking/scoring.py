from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from streamrank.ranking.calibration import Calibrator, IdentityCalibrator

POSITIVE_OBJECTIVES = {"is_click", "long_view", "is_like", "freshness"}
NEGATIVE_OBJECTIVES = {"is_hate"}


@dataclass
class PolicyScorer:
    """A single, explicit score-descending convention."""

    weights: Mapping[str, float]
    calibrators: Mapping[str, Calibrator] | None = None

    def __post_init__(self) -> None:
        for objective in POSITIVE_OBJECTIVES:
            if objective in self.weights and self.weights[objective] < 0:
                raise ValueError(f"positive objective {objective} must have a non-negative weight")
        for objective in NEGATIVE_OBJECTIVES:
            if objective in self.weights and self.weights[objective] > 0:
                raise ValueError(f"negative objective {objective} must have a non-positive weight")

    def score(self, probabilities: Mapping[str, float], freshness: float = 0.0) -> float:
        calibrated: dict[str, float] = {}
        calibrators = self.calibrators or {}
        for task in ("is_click", "long_view", "is_like", "is_hate"):
            calibrator = calibrators.get(task, IdentityCalibrator())
            calibrated[task] = calibrator.predict(float(probabilities.get(task, 0.0)))
        normalized_freshness = min(1.0, max(0.0, float(freshness)))
        values = {**calibrated, "freshness": normalized_freshness}
        return sum(float(self.weights.get(name, 0.0)) * value for name, value in values.items())
