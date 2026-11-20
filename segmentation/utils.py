"""检查点、预测可视化和训练曲线。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure

from segmentation.config import Config


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_iou: float,
    config: Config,
    filepath: Path,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    """保存可恢复训练、也可被推理入口加载的检查点。"""

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_iou": best_iou,
        "config": config,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: str = "cpu",
) -> dict:
    """把 model_state_dict 载入模型，并按需恢复优化器。"""

    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def visualize_prediction(
    image: torch.Tensor,
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor | None = None,
    class_names: list[str] | None = None,
    save_path: Path | None = None,
) -> Figure:
    """并排绘制输入、预测和可选的真值。"""

    array = image.detach().cpu().numpy().transpose(1, 2, 0)
    if array.min() < 0:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        array = array * std + mean
    array = np.clip(array, 0, 1)

    columns = 3 if gt_mask is not None else 2
    figure, axes = plt.subplots(1, columns, figsize=(5 * columns, 5))
    axes[0].imshow(array)
    axes[0].set_title("输入图像")
    axes[0].axis("off")
    shown = axes[1].imshow(pred_mask.detach().cpu().numpy(), cmap="viridis", vmin=0)
    axes[1].set_title("预测结果")
    axes[1].axis("off")
    if class_names is not None:
        colorbar = figure.colorbar(shown, ax=axes[1], ticks=range(len(class_names)))
        colorbar.ax.set_yticklabels(class_names)
    if gt_mask is not None:
        axes[2].imshow(gt_mask.detach().cpu().numpy(), cmap="viridis", vmin=0)
        axes[2].set_title("真实标签")
        axes[2].axis("off")
    figure.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
    return figure


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_ious: list[float] | None = None,
    save_path: Path | None = None,
) -> Figure:
    """绘制训练/验证损失和验证 mIoU。"""

    panels = 2 if val_ious is not None else 1
    figure, axes = plt.subplots(1, panels, figsize=(6 * panels, 4))
    if panels == 1:
        axes = [axes]
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label="train")
    axes[0].plot(epochs, val_losses, label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    if val_ious is not None and panels > 1:
        axes[1].plot(epochs, val_ious, label="val mIoU")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("mIoU")
        axes[1].set_title("validation mIoU")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    figure.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
    return figure
