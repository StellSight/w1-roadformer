"""语义分割的评估指标。"""

from __future__ import annotations

import torch


def _valid(target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    if ignore_index >= 0:
        return target != ignore_index
    return torch.ones_like(target, dtype=torch.bool)


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1.0,
    ignore_index: int = -100,
) -> dict[str, float]:
    results: dict[str, float] = {}
    scores: list[float] = []
    valid = _valid(target, ignore_index)
    for cls_idx in range(num_classes):
        pred_cls = ((pred == cls_idx) & valid).float()
        target_cls = ((target == cls_idx) & valid).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        score = ((2.0 * intersection + smooth) / (union + smooth)).item()
        results[f"class_{cls_idx}"] = score
        scores.append(score)
    results["mean"] = sum(scores) / len(scores)
    return results


def iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100,
) -> dict[str, float]:
    results: dict[str, float] = {}
    scores: list[float] = []
    valid = _valid(target, ignore_index)
    for cls_idx in range(num_classes):
        pred_cls = (pred == cls_idx) & valid
        target_cls = (target == cls_idx) & valid
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        score = (intersection / union.clamp_min(1)).item()
        results[f"class_{cls_idx}"] = score
        scores.append(score)
    results["mean"] = sum(scores) / len(scores)
    return results


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = -100) -> float:
    valid = _valid(target, ignore_index)
    correct = (pred == target)[valid].sum().float()
    total = valid.sum().float().clamp_min(1)
    return (correct / total).item()


class MetricsTracker:
    """累积批次指标。update 接收的是类别索引，不是 logits。"""

    def __init__(self, num_classes: int, ignore_index: int = -100) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self._intersection = [0.0] * self.num_classes
        self._union = [0.0] * self.num_classes
        self._correct_pixels = 0.0
        self._total_pixels = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.detach()
        target = target.detach()
        valid = _valid(target, self.ignore_index)
        self._correct_pixels += (pred == target)[valid].sum().item()
        self._total_pixels += valid.sum().item()
        for cls_idx in range(self.num_classes):
            pred_cls = (pred == cls_idx) & valid
            target_cls = (target == cls_idx) & valid
            self._intersection[cls_idx] += (pred_cls & target_cls).sum().item()
            self._union[cls_idx] += (pred_cls | target_cls).sum().item()

    def compute(self) -> dict[str, float]:
        results: dict[str, float] = {}
        scores: list[float] = []
        for cls_idx in range(self.num_classes):
            score = self._intersection[cls_idx] / max(self._union[cls_idx], 1.0)
            results[f"iou_class_{cls_idx}"] = score
            scores.append(score)
        results["iou_mean"] = sum(scores) / len(scores) if scores else 0.0
        results["pixel_accuracy"] = self._correct_pixels / max(self._total_pixels, 1.0)
        return results
