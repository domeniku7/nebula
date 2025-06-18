from __future__ import annotations

import torch
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall


class FlaggingEvaluator:
    """Utility to track precision, recall and F1 score for binary flagging."""

    def __init__(self) -> None:
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.f1 = BinaryF1Score()

    def update(self, predicted: bool, actual: bool) -> None:
        """Update metrics with a new prediction."""
        pred_tensor = torch.tensor([int(predicted)])
        actual_tensor = torch.tensor([int(actual)])
        self.precision.update(pred_tensor, actual_tensor)
        self.recall.update(pred_tensor, actual_tensor)
        self.f1.update(pred_tensor, actual_tensor)

    def compute(self, reset: bool = True) -> dict[str, float]:
        """Compute metrics and optionally reset internal state."""
        metrics = {
            "precision": float(self.precision.compute()),
            "recall": float(self.recall.compute()),
            "f1": float(self.f1.compute()),
        }
        if reset:
            self.precision.reset()
            self.recall.reset()
            self.f1.reset()
        return metrics