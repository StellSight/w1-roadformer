"""W1-RoadFormer.

编码器是 SegFormer 的 MiT-B2：重叠切片、高效自注意力、Mix-FFN。
模块名与 NVlabs SegFormer 的 mit_b2 对齐，便于加载公开的编码器预训练权重。
解码器是多尺度线性投影后上采样融合，输出与输入同尺寸的两类 logits。
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int, stride: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, height, width


class DWConv(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = x.shape
        x = x.transpose(1, 2).reshape(batch, channels, height, width)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int, attn_drop: float, proj_drop: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"通道数 {dim} 不能被头数 {num_heads} 整除")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, length, channels = x.shape
        query = self.q(x).reshape(batch, length, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            reduced = x.permute(0, 2, 1).reshape(batch, channels, height, width)
            reduced = self.sr(reduced).reshape(batch, channels, -1).permute(0, 2, 1)
            reduced = self.norm(reduced)
            key_value = self.kv(reduced)
        else:
            key_value = self.kv(x)
        key_value = key_value.reshape(batch, -1, 2, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        key, value = key_value[0], key_value[1]
        # 注意力分数用 float32 计算。1660 上半精度的 q@k 会溢出成 inf，softmax 之后变成 nan。
        value_dtype = value.dtype
        attention = (query.float() @ key.float().transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1)).to(value_dtype)
        out = (attention @ value).transpose(1, 2).reshape(batch, length, channels)
        return self.proj_drop(self.proj(out))


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, sr_ratio: int, drop_path: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = EfficientAttention(dim, num_heads, sr_ratio, attn_drop=0.0, proj_drop=0.0)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio), drop=0.0)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x


class MiTB2Encoder(nn.Module):
    """SegFormer MiT-B2。四阶段通道数为 64 / 128 / 320 / 512。"""

    def __init__(self, drop_path_rate: float = 0.1) -> None:
        super().__init__()
        embed_dims = (64, 128, 320, 512)
        num_heads = (1, 2, 5, 8)
        depths = (3, 4, 6, 3)
        sr_ratios = (8, 4, 2, 1)
        mlp_ratios = (4.0, 4.0, 4.0, 4.0)
        patch_sizes = (7, 3, 3, 3)
        strides = (4, 2, 2, 2)
        drop_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()

        in_channels = 3
        cursor = 0
        for stage, (embed_dim, depth) in enumerate(zip(embed_dims, depths)):
            setattr(
                self,
                f"patch_embed{stage + 1}",
                OverlapPatchEmbed(in_channels, embed_dim, patch_sizes[stage], strides[stage]),
            )
            blocks = [
                EncoderBlock(
                    embed_dim,
                    num_heads[stage],
                    mlp_ratios[stage],
                    sr_ratios[stage],
                    drop_rates[cursor + block_index],
                )
                for block_index in range(depth)
            ]
            setattr(self, f"block{stage + 1}", nn.ModuleList(blocks))
            setattr(self, f"norm{stage + 1}", nn.LayerNorm(embed_dim, eps=1e-6))
            in_channels = embed_dim
            cursor += depth
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        batch = x.shape[0]
        for stage in range(1, 5):
            embed = getattr(self, f"patch_embed{stage}")
            blocks = getattr(self, f"block{stage}")
            norm = getattr(self, f"norm{stage}")
            x, height, width = embed(x)
            for block in blocks:
                x = block(x, height, width)
            x = norm(x)
            x = x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
            features.append(x)
        return features


class LinearProjection(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x.transpose(1, 2).reshape(batch, -1, height, width)


class MLPDecoder(nn.Module):
    def __init__(self, in_channels: tuple[int, ...], embed_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.projections = nn.ModuleList(LinearProjection(channels, embed_dim) for channels in in_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            item = projection(feature)
            if item.shape[-2:] != target_size:
                item = F.interpolate(item, size=target_size, mode="bilinear", align_corners=False)
            projected.append(item)
        fused = self.pred(self.dropout(self.fuse(torch.cat(projected, dim=1))))
        return F.interpolate(fused, size=output_size, mode="bilinear", align_corners=False)


class W1RoadFormer(nn.Module):
    """武汉一号道路分割模型。输出形状为 (N, num_classes, H, W)。"""

    def __init__(self, num_classes: int = 2, decoder_dim: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = MiTB2Encoder()
        self.decoder = MLPDecoder((64, 128, 320, 512), decoder_dim, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.decoder(features, output_size=x.shape[-2:])


def create_roadformer(
    in_channels: int = 3,
    num_classes: int = 2,
    decoder_dim: int = 768,
    dropout: float = 0.1,
) -> W1RoadFormer:
    """创建 W1-RoadFormer。输入通道目前只支持 3。"""

    if in_channels != 3:
        raise ValueError("W1-RoadFormer 目前只接受 3 通道 RGB。")
    return W1RoadFormer(num_classes=num_classes, decoder_dim=decoder_dim, dropout=dropout)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """返回 (总参数量, 可训练参数量)。"""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def _convert_mmseg_mit(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """把 mmsegmentation 的 MixVisionTransformer 键名转成当前编码器的键名。"""

    converted: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        parts = key.split(".")
        if len(parts) >= 4 and parts[0] == "layers" and parts[2] == "0" and parts[3] in {"projection", "norm"}:
            stage = int(parts[1]) + 1
            leaf = "proj" if parts[3] == "projection" else "norm"
            converted[f"patch_embed{stage}.{leaf}.{'.'.join(parts[4:])}"] = value
            continue
        if len(parts) == 4 and parts[0] == "layers" and parts[2] == "2" and parts[3] in {"weight", "bias"}:
            converted[f"norm{int(parts[1]) + 1}.{parts[3]}"] = value
            continue
        if not (len(parts) >= 5 and parts[0] == "layers" and parts[2] == "1"):
            continue
        prefix = f"block{int(parts[1]) + 1}.{parts[3]}"
        tail = parts[4:]
        if tail[:2] == ["attn", "attn"] and tail[2] in {"in_proj_weight", "in_proj_bias"}:
            query, key_weight, value_weight = value.chunk(3, dim=0)
            converted[f"{prefix}.attn.q.{tail[2].removeprefix('in_proj_')}"] = query
            converted[f"{prefix}.attn.kv.{tail[2].removeprefix('in_proj_')}"] = torch.cat([key_weight, value_weight], dim=0)
            continue
        if tail[:3] == ["attn", "attn", "out_proj"]:
            converted[f"{prefix}.attn.proj.{tail[3]}"] = value
            continue
        if tail[0] == "attn" and tail[1] in {"sr", "norm"}:
            converted[f"{prefix}.attn.{tail[1]}.{tail[2]}"] = value
            continue
        if tail[0] in {"norm1", "norm2"}:
            converted[f"{prefix}.{tail[0]}.{tail[1]}"] = value
            continue
        if tail[:3] == ["ffn", "layers", "0"]:
            converted[f"{prefix}.mlp.fc1.{tail[3]}"] = value.reshape(value.shape[0], -1) if value.ndim == 4 else value
            continue
        if tail[:3] == ["ffn", "layers", "1"]:
            converted[f"{prefix}.mlp.dwconv.dwconv.{tail[3]}"] = value
            continue
        if tail[:3] == ["ffn", "layers", "4"]:
            converted[f"{prefix}.mlp.fc2.{tail[3]}"] = value.reshape(value.shape[0], -1) if value.ndim == 4 else value
    return converted


def load_encoder_weights(encoder: MiTB2Encoder, path: str | Path) -> tuple[int, int]:
    """加载 MiT-B2 编码器权重。一个键都对不上时直接报错，避免误以为用了预训练。"""

    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(path), map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        name = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "backbone.", "encoder.", "model."):
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    changed = True
        cleaned[name] = value
    if any(name.startswith("layers.") for name in cleaned):
        cleaned = _convert_mmseg_mit(cleaned)

    current = encoder.state_dict()
    matched = {
        key: value
        for key, value in cleaned.items()
        if key in current and current[key].shape == value.shape
    }
    if not matched:
        sample = ", ".join(list(cleaned)[:6])
        raise RuntimeError(
            "编码器权重没有一个键对上。请使用 SegFormer MiT-B2 预训练文件，"
            f"键名应类似 patch_embed1.proj.weight。当前文件里的键例如：{sample}"
        )
    encoder.load_state_dict(matched, strict=False)
    return len(matched), len(current)
