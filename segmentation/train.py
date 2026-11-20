"""语义分割训练脚本。

用法:
    python -m segmentation.train --sample_datasets /dir/to/dataset --out_weights /dir/to/best.pth

    python -m segmentation.train --demo --epochs 1 --image-size 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from segmentation.config import Config
from segmentation.dataset import create_dataloaders, generate_demo_dataset
from segmentation.loss import CombinedLoss
from segmentation.metrics import MetricsTracker
from segmentation.network import count_parameters, create_roadformer, load_encoder_weights
from segmentation.utils import plot_training_curves, save_checkpoint


def load_full_weights(model: torch.nn.Module, path: str | Path) -> tuple[float, int]:
    """加载完整模型权重，返回其中记录的最佳 mIoU 和轮次。"""

    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(path), map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint.get("model"))
    if state is None:
        raise KeyError(f"{path} 里没有 model_state_dict，不能在此基础上继续训练。")
    model.load_state_dict(state)
    return float(checkpoint.get("best_iou", -1.0)), int(checkpoint.get("epoch", -1))


class Trainer:
    """管理完整的训练生命周期。"""

    def __init__(self, config: Config, train_loader: DataLoader, val_loader: DataLoader) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(config.device_str)
        self.model = create_roadformer(
            in_channels=config.in_channels,
            num_classes=config.num_classes,
            decoder_dim=config.decoder_dim,
            dropout=config.dropout,
        ).to(self.device)
        if config.encoder_weights:
            matched, total = load_encoder_weights(self.model.encoder, config.encoder_weights)
            print(f"已加载编码器权重 {matched}/{total}：{config.encoder_weights}")
        self.baseline_iou = -1.0
        if config.init_weights:
            self.baseline_iou, prior_epoch = load_full_weights(self.model, config.init_weights)
            print(
                f"已加载完整权重：{config.init_weights}（第 {prior_epoch + 1} 轮，mIoU {self.baseline_iou:.4f}）"
            )
            print("只有验证 mIoU 超过这个数，才会保存新的最佳权重。")

        total, trainable = count_parameters(self.model)
        print(f"W1-RoadFormer 参数量 — 总计: {total:,}，可训练: {trainable:,}")

        self.criterion = CombinedLoss(
            num_classes=config.num_classes,
            bce_weight=0.5,
            dice_weight=0.5,
            class_weights=[0.25, 0.75],
        ).to(self.device)

        encoder_ids = {id(parameter) for parameter in self.model.encoder.parameters()}
        encoder_params = [parameter for parameter in self.model.parameters() if id(parameter) in encoder_ids]
        decoder_params = [parameter for parameter in self.model.parameters() if id(parameter) not in encoder_ids]
        pretrained_encoder = bool(config.encoder_weights or config.init_weights)
        encoder_lr = config.learning_rate * (0.1 if pretrained_encoder else 1.0)
        self.optimizer = AdamW(
            [
                {"params": encoder_params, "lr": encoder_lr},
                {"params": decoder_params, "lr": config.learning_rate},
            ],
            weight_decay=config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience,
        )
        self.metrics = MetricsTracker(num_classes=config.num_classes)
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_ious: list[float] = []
        self.best_iou = self.baseline_iou
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        # MiT-B2 的中间特征会超过 float16 的上限，半精度会把损失变成 nan。这里始终用 float32。
        self.use_amp = False
        self.scaler = None

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        progress = tqdm(self.train_loader, desc="训练中", leave=False)
        for batch in progress:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def validate_epoch(self) -> tuple[float, dict[str, float]]:
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        progress = tqdm(self.val_loader, desc="验证中", leave=False)
        for batch in progress:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            logits = self.model(images)
            loss = self.criterion(logits, masks)
            total_loss += loss.item()
            self.metrics.update(logits.argmax(dim=1), masks)
            progress.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / max(len(self.val_loader), 1), self.metrics.compute()

    def fit(self) -> None:
        print(f"\n训练设备: {self.device}")
        print(f"轮次数: {self.config.num_epochs} | 批次大小: {self.config.batch_size}")
        print(f"训练样本: {len(self.train_loader.dataset)} | 验证样本: {len(self.val_loader.dataset)}\n")
        epoch = 0
        for epoch in range(self.config.num_epochs):
            print(f"--- 第 {epoch + 1}/{self.config.num_epochs} 轮 ---")
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.validate_epoch()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_ious.append(val_metrics["iou_mean"])
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(
                f"  验证 mIoU: {val_metrics['iou_mean']:.4f}  |  "
                f"道路 IoU: {val_metrics.get('iou_class_1', 0.0):.4f}  |  "
                f"准确率: {val_metrics['pixel_accuracy']:.4f}"
            )
            self.scheduler.step(val_metrics["iou_mean"])
            if val_metrics["iou_mean"] > self.best_iou:
                self.best_iou = val_metrics["iou_mean"]
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.best_iou,
                    self.config,
                    self.config.checkpoint,
                    scheduler=self.scheduler,
                )
                print(f"  已保存新的最佳模型（mIoU: {self.best_iou:.4f}）")
            else:
                self.epochs_without_improvement += 1
                print(f"  已连续 {self.epochs_without_improvement} 轮无提升")
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"\n第 {epoch + 1} 轮触发早停")
                break

        last_path = self.config.log_dir / "last.pth"
        save_checkpoint(
            self.model,
            self.optimizer,
            epoch,
            self.best_iou,
            self.config,
            last_path,
            scheduler=self.scheduler,
        )
        plot_training_curves(
            self.train_losses,
            self.val_losses,
            self.val_ious,
            save_path=self.config.log_dir / "training_curves.png",
        )
        if self.best_epoch < 0:
            print(f"\n训练完成。没有超过基准 mIoU {self.best_iou:.4f}，不覆盖新的最佳权重。")
        else:
            print(f"\n训练完成。最佳 mIoU: {self.best_iou:.4f}（本轮次第 {self.best_epoch + 1} 轮）")
            print(f"检查点已保存至: {self.config.checkpoint}")
            print("打包提交前，把该文件复制为项目根目录的 pretrained.pth，再执行 .\\pack.ps1")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="训练 W1-RoadFormer")
    parser.add_argument("--sample_datasets", type=Path, help="数据集根目录，训练影像在 train/images，掩膜在 train/masks")
    parser.add_argument("--out_weights", type=Path, help="最佳权重保存路径")
    parser.add_argument("--epochs", type=int, default=80, help="最大训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="训练批次大小")
    parser.add_argument("--lr", type=float, default=6e-5, help="学习率")
    parser.add_argument("--image-size", type=int, default=512, help="输入图像边长")
    parser.add_argument("--num-classes", type=int, default=2, help="分割类别数")
    parser.add_argument("--decoder-dim", type=int, default=768, help="解码头通道数")
    parser.add_argument("--encoder-weights", type=str, default="", help="MiT-B2 编码器预训练权重")
    parser.add_argument("--init-weights", type=str, default="", help="完整模型权重，在此基础上继续训练")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader 工作进程数")
    parser.add_argument("--device", type=str, default="auto", choices=["cuda", "cpu", "auto"], help="计算设备")
    parser.add_argument("--demo", action="store_true", help="生成合成数据并做一次短训练，确认入口能跑通")
    args = parser.parse_args(argv)

    if args.demo:
        if "--epochs" not in sys.argv:
            args.epochs = 1
        if "--image-size" not in sys.argv:
            args.image_size = 128
        if "--batch-size" not in sys.argv:
            args.batch_size = 2
        if "--num-workers" not in sys.argv:
            args.num_workers = 0
        demo_root = Path("logs/demo/dataset")
        generate_demo_dataset(demo_root, size=args.image_size)
        args.sample_datasets = demo_root
        args.out_weights = args.out_weights or Path("logs/demo/best.pth")
        print(f"已生成演示数据：{demo_root}")

    if args.init_weights:
        if "--lr" not in sys.argv:
            args.lr = 1e-5
        if "--epochs" not in sys.argv:
            args.epochs = 20

    if args.sample_datasets is None or args.out_weights is None:
        parser.error("必须提供 --sample_datasets 和 --out_weights")

    config = Config(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        num_classes=args.num_classes,
        decoder_dim=args.decoder_dim,
        checkpoint=args.out_weights,
        device=args.device,
        num_workers=args.num_workers,
        encoder_weights=args.encoder_weights,
        init_weights=args.init_weights,
        imagenet_norm=bool(args.encoder_weights or args.init_weights),
        log_dir=Path(args.out_weights).parent,
    )
    if args.init_weights:
        try:
            prior = torch.load(args.init_weights, map_location="cpu", weights_only=False)
        except TypeError:
            prior = torch.load(args.init_weights, map_location="cpu")
        prior_config = prior.get("config")
        if prior_config is not None:
            config.norm_low = getattr(prior_config, "norm_low", config.norm_low)
            config.norm_high = getattr(prior_config, "norm_high", config.norm_high)
            config.imagenet_norm = bool(getattr(prior_config, "imagenet_norm", True))
        config.lr_scheduler_patience = 8
    image_dir = Path(args.sample_datasets) / "train" / "images"
    mask_dir = Path(args.sample_datasets) / "train" / "masks"
    train_loader, val_loader = create_dataloaders(
        image_dir=image_dir,
        mask_dir=mask_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers,
        config=config,
    )
    Trainer(config, train_loader, val_loader).fit()


if __name__ == "__main__":
    main()
