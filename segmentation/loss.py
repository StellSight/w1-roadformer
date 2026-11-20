"""语义分割的损失函数。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """多类别软 Dice。Loss = 1 - Dice。"""

    def __init__(self, smooth: float = 1.0, ignore_index: int = -100) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        probs = torch.softmax(logits.float(), dim=1)
        target = target.long()
        valid = target != self.ignore_index if self.ignore_index >= 0 else torch.ones_like(target, dtype=torch.bool)
        target_safe = target.clamp(0, num_classes - 1)
        one_hot = F.one_hot(target_safe, num_classes=num_classes).permute(0, 3, 1, 2).float()
        mask = valid.unsqueeze(1).float()
        probs = probs * mask
        one_hot = one_hot * mask
        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()


class FocalLoss(nn.Module):
    """FL = -α * (1 - p_t)^γ * log(p_t)。"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = -100) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits.float(), dim=1)
        probs = log_probs.exp()
        target = target.long()
        valid = target != self.ignore_index if self.ignore_index >= 0 else torch.ones_like(target, dtype=torch.bool)
        target_safe = target.clamp_min(0)
        target_probs = probs.gather(1, target_safe.unsqueeze(1)).squeeze(1)
        focal_weight = self.alpha * (1.0 - target_probs) ** self.gamma
        ce_loss = F.nll_loss(log_probs, target_safe, reduction="none")
        loss = focal_weight * ce_loss
        loss = loss * valid.float()
        return loss.sum() / valid.float().sum().clamp_min(1)


class CombinedLoss(nn.Module):
    """总损失 = λ_bce * Focal + λ_dice * Dice。

    参数名与样例一致。道路像素少，默认用类别权重把损失压到道路上。
    """

    def __init__(
        self,
        num_classes: int = 2,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: list[float] | None = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32), persistent=False)
        else:
            self.class_weights = None
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.focal = FocalLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            ce = F.cross_entropy(logits.float(), target.long(), weight=weight, ignore_index=self.ce.ignore_index)
        else:
            ce = self.focal(logits, target)
        return self.bce_weight * ce + self.dice_weight * self.dice(logits, target)
