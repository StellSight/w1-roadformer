"""赛道三推理入口。

平台执行的命令等效于：

    python -m segmentation.inference --pretrained ./pretrained.pth --input_dataset /dir/to/dataset/ --output_dir /dir/to/output_dir

测试影像位于 input_dataset/test/images/。输出为同名单波段 uint8 GeoTIFF，像素值是类别号：0 背景，1 道路。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from contextlib import nullcontext

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from segmentation.config import Config
from segmentation.network import W1RoadFormer, create_roadformer
from segmentation.dataset import apply_imagenet, scale_to_unit

IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W1-RoadFormer 测试集推理：同名 tif 分类掩膜")
    parser.add_argument("--pretrained", type=Path, required=True, help="模型权重路径（.pth / .pt）")
    parser.add_argument("--input_dataset", type=Path, required=True, help="数据集根目录，影像在 test/images 下")
    parser.add_argument("--output_dir", type=Path, required=True, help="预测结果输出文件夹")
    parser.add_argument("--device", default="auto", choices=["cuda", "cpu", "auto"], help="计算设备")
    parser.add_argument("--batch_size", type=int, default=4, help="推理批大小，平台不传此参数时使用默认值")
    return parser.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 cuda，但当前环境没有可用 GPU。")
    return torch.device(name)


def config_get(config: object, key: str, default):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    """读取权重。Windows 上保存的 Path 在 Linux 容器里要改成 PosixPath。"""

    import pathlib
    import sys

    if sys.platform != "win32":
        pathlib.WindowsPath = pathlib.PosixPath
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[W1RoadFormer, Config, dict | None]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint.get("config") or Config()
    if isinstance(config, dict):
        known = {item.name for item in Config.__dataclass_fields__.values()}
        config = Config(**{key: value for key, value in config.items() if key in known})
    state = checkpoint.get("model_state_dict", checkpoint.get("model"))
    if state is None:
        raise KeyError("权重文件中没有 model_state_dict 或 model。")
    model = create_roadformer(
        in_channels=int(config_get(config, "in_channels", 3)),
        num_classes=int(config_get(config, "num_classes", 2)),
        decoder_dim=int(config_get(config, "decoder_dim", 768)),
        dropout=float(config_get(config, "dropout", 0.1)),
    )
    model.load_state_dict(state)
    model.to(device).eval()
    if config_get(config, "norm_low", None) is not None and config_get(config, "norm_high", None) is not None:
        norm_stats = {
            "p_low": list(config_get(config, "norm_low", [])),
            "p_high": list(config_get(config, "norm_high", [])),
        }
    else:
        norm_stats = checkpoint.get("norm_stats")
    return model, config, norm_stats


def image_stats(image: np.ndarray) -> dict[str, list[float]]:
    flat = image.reshape(-1, image.shape[-1]).astype(np.float64)
    low = np.percentile(flat, 2, axis=0)
    high = np.maximum(np.percentile(flat, 98, axis=0), low + 1.0)
    return {"p_low": low.astype(float).tolist(), "p_high": high.astype(float).tolist()}


def prepare_tensor(
    chw: np.ndarray,
    norm_stats: dict | None,
    image_size: int,
    imagenet_norm: bool,
) -> tuple[torch.Tensor, dict]:
    if chw.shape[0] == 1:
        chw = np.repeat(chw, 3, axis=0)
    elif chw.shape[0] > 3:
        chw = chw[:3]
    image = np.transpose(chw, (1, 2, 0))
    stats = norm_stats or image_stats(image)
    image = scale_to_unit(image, stats)
    if imagenet_norm:
        image = apply_imagenet(image)
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
    if tensor.shape[-2:] != (image_size, image_size):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor, stats


def class_map(logit: torch.Tensor, height: int, width: int) -> np.ndarray:
    if tuple(logit.shape[-2:]) != (height, width):
        logit = F.interpolate(logit.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False).squeeze(0)
    return logit.argmax(dim=0).to("cpu").numpy().astype(np.uint8)


def save_mask(profile: dict, prediction: np.ndarray, out_path: Path) -> None:
    written = profile.copy()
    written.update(driver="GTiff", count=1, dtype=rasterio.uint8, compress="lzw", nodata=None)
    with rasterio.open(out_path, "w", **written) as dataset:
        dataset.write(prediction, 1)


def list_test_images(dataset_dir: Path) -> list[Path]:
    image_dir = dataset_dir / "test" / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"未找到 {image_dir}，输入应为包含 test/images 的数据集目录")
    paths = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"{image_dir} 中未找到支持的影像文件")
    return paths


def run_inference(dataset_dir: Path, checkpoint_path: Path, output_dir: Path, device_name: str, batch_size: int) -> None:
    device = resolve_device(device_name)
    model, config, norm_stats = load_model(checkpoint_path, device)
    image_size = int(config_get(config, "image_size", 512))
    imagenet_norm = bool(config_get(config, "imagenet_norm", False))
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_test_images(dataset_dir)
    # MiT-B2 在 float16 下会溢出成 nan。best.pth 在验证图上已复现，推理保持 float32。
    use_fp16 = False
    print(f"数据集推理: 共 {len(image_paths)} 张切片, 设备 {device}, 结果输出至 {output_dir}", flush=True)

    pending_tensors: list[torch.Tensor] = []
    pending_meta: list[tuple[dict, int, int, Path]] = []

    def flush() -> None:
        if not pending_tensors:
            return
        batch = torch.stack(pending_tensors, dim=0).to(device, non_blocking=True)
        amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_fp16 else nullcontext()
        with torch.inference_mode(), amp_context:
            logits = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        for logit, (profile, height, width, out_path) in zip(logits, pending_meta):
            save_mask(profile, class_map(logit, height, width), out_path)
            print(f"  -> {out_path.name}", flush=True)
        pending_tensors.clear()
        pending_meta.clear()

    for index, image_path in enumerate(image_paths, 1):
        print(f"[{index}/{len(image_paths)}] {image_path.name}", flush=True)
        with rasterio.open(image_path) as source:
            profile = source.profile.copy()
            chw = source.read()
            height, width = source.height, source.width
        tensor, _ = prepare_tensor(chw, norm_stats, image_size, imagenet_norm)
        pending_tensors.append(tensor)
        pending_meta.append((profile, height, width, output_dir / f"{image_path.stem}.tif"))
        if len(pending_tensors) >= batch_size:
            flush()
    flush()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_inference(args.input_dataset, args.pretrained, args.output_dir, args.device, args.batch_size)


if __name__ == "__main__":
    main()
