"""训练与推理的主配置。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class Config:
    """训练与推理共用的配置。

    属性:
        in_channels: 输入图像通道数，武汉一号 RGB 为 3。
        num_classes: 分割类别数，含背景。
        decoder_dim: SegFormer 解码头通道数，B2 为 768。
        dropout: 解码头 dropout。
        learning_rate: 解码器学习率。编码器加载预训练权重时使用其 0.1 倍。
        weight_decay: AdamW 权重衰减。
        batch_size: 训练批次大小。
        num_epochs: 最大训练轮数。
        image_size: 输入图像边长，官方切片为 512，且必须能被 32 整除。
        device: "cuda"、"cpu" 或 "auto"。
        num_workers: DataLoader 工作进程数。
        checkpoint: 最佳权重要写入的路径，对应命令行 --out_weights。
        log_dir: 曲线和最后一轮权重的目录。
        encoder_weights: MiT-B2 编码器预训练权重，可为空。
        init_weights: 完整模型权重。给出时在该权重上继续训练，而不是从头开始。
        imagenet_norm: 百分位归一化之后是否再做 ImageNet 标准化。
        norm_low: 三个波段的低分位，由训练集统计后写入。
        norm_high: 三个波段的高分位。
        early_stopping_patience: 验证 mIoU 连续无提升时停止的轮数。
        lr_scheduler_patience: 降低学习率前允许无提升的轮数。
        lr_scheduler_factor: 学习率衰减因子。
    """

    in_channels: int = 3
    num_classes: int = 2
    decoder_dim: int = 768
    dropout: float = 0.1

    learning_rate: float = 6e-5
    weight_decay: float = 0.01

    batch_size: int = 4
    num_epochs: int = 80
    image_size: int = 512

    device: Literal["cuda", "cpu", "auto"] = "auto"
    num_workers: int = 4

    checkpoint: Path = field(default_factory=lambda: Path("best.pth"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    encoder_weights: str = ""
    init_weights: str = ""
    imagenet_norm: bool = False
    norm_low: tuple[float, float, float] | None = None
    norm_high: tuple[float, float, float] | None = None

    early_stopping_patience: int = 10
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5

    def __post_init__(self) -> None:
        self.checkpoint = Path(self.checkpoint)
        self.log_dir = Path(self.log_dir)
        if self.image_size % 32 != 0:
            raise ValueError("image_size 必须能被 32 整除。")

    @property
    def device_str(self) -> str:
        """解析并返回计算设备字符串。"""
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
