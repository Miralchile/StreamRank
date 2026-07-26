from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class Calibrator(Protocol):
    def predict(self, probability: float) -> float: ...


@dataclass
class IdentityCalibrator:
    def predict(self, probability: float) -> float:
        return min(1.0, max(0.0, float(probability)))


@dataclass
class PlattCalibrator:
    """Calibrates logits on an unsampled, temporally later calibration set."""

    slope: float = 1.0
    intercept: float = 0.0
    epsilon: float = 1e-6

    @staticmethod
    def _logit(probability: float, epsilon: float) -> float:
        clipped = min(1.0 - epsilon, max(epsilon, probability))
        return math.log(clipped / (1.0 - clipped))

    def fit(
        self,
        probabilities: Iterable[float],
        labels: Iterable[int],
        learning_rate: float = 0.05,
        iterations: int = 1500,
        l2: float = 1e-4,
    ) -> "PlattCalibrator":
        xs = [self._logit(float(value), self.epsilon) for value in probabilities]
        ys = [int(label) for label in labels]
        if len(xs) != len(ys) or not xs:
            raise ValueError("probabilities and labels must be non-empty and equal length")
        if len(set(ys)) < 2:
            raise ValueError("calibration requires both positive and negative labels")
        slope, intercept = self.slope, self.intercept
        for _ in range(iterations):
            predictions = [_sigmoid(slope * x + intercept) for x in xs]
            grad_slope = sum(
                (pred - y) * x for pred, y, x in zip(predictions, ys, xs, strict=True)
            ) / len(xs)
            grad_intercept = sum(pred - y for pred, y in zip(predictions, ys, strict=True)) / len(
                xs
            )
            slope -= learning_rate * (grad_slope + l2 * slope)
            # Calibration must not reverse the model's ordering.
            slope = max(0.0, slope)
            intercept -= learning_rate * grad_intercept
        self.slope, self.intercept = slope, intercept
        return self

    def predict(self, probability: float) -> float:
        return _sigmoid(self.slope * self._logit(probability, self.epsilon) + self.intercept)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PlattCalibrator":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
