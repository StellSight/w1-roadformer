"""语义分割的数据集类与 DataLoader 工厂函数。

官方训练集目录::

    {sample_datasets}/train/images
    {sample_datasets}/train/masks

影像与掩膜按文件名（不含扩展名）对应。武汉一号切片是 UInt16 RGB，不用 8 位方式读取。
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from segmentation.config import Config

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def read_array(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        import rasterio

        with rasterio.open(path) as source:
            array = source.read()
        if array.ndim == 3 and array.shape[0] <= 4:
            array = np.transpose(array, (1, 2, 0))
        return array
    return np.asarray(Image.open(path))


def to_hwc(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[0] <= 4 and image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image[..., :3]


def read_mask(path: Path) -> np.ndarray:
    array = np.asarray(read_array(path))
    if array.ndim == 3:
        array = array[..., 0]
    mask = array.astype(np.int64)
    if mask.size and int(mask.max()) > 1:
        mask = (mask > 0).astype(np.int64)
    return mask


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape[0] == size and image.shape[1] == size:
        return image
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).float()
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return tensor.squeeze(0).permute(1, 2, 0).numpy()


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    if mask.shape[0] == size and mask.shape[1] == size:
        return mask
    tensor = torch.from_numpy(mask[None, None].astype(np.float32))
    tensor = F.interpolate(tensor, size=(size, size), mode="nearest")
    return tensor.squeeze().numpy().astype(np.int64)


def scale_to_unit(image: np.ndarray, norm_stats: dict[str, list[float]]) -> np.ndarray:
    low = np.asarray(norm_stats["p_low"], dtype=np.float32)
    high = np.asarray(norm_stats["p_high"], dtype=np.float32)
    out = np.clip(image.astype(np.float32), low, high)
    return (out - low) / np.maximum(high - low, 1e-6)


def apply_imagenet(image: np.ndarray) -> np.ndarray:
    return ((image - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)


def compute_norm_stats(image_paths: list[Path], seed: int = 42) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    chosen = list(image_paths)
    if len(chosen) > 200:
        picked = rng.choice(len(chosen), size=200, replace=False)
        chosen = [chosen[index] for index in picked]
    samples = []
    for path in chosen:
        flat = to_hwc(read_array(path)).reshape(-1, 3)
        take = min(4096, flat.shape[0])
        index = rng.choice(flat.shape[0], size=take, replace=False)
        samples.append(flat[index])
    data = np.concatenate(samples, axis=0).astype(np.float64)
    low = np.percentile(data, 2, axis=0)
    high = np.maximum(np.percentile(data, 98, axis=0), low + 1.0)
    return {"p_low": low.astype(float).tolist(), "p_high": high.astype(float).tolist()}


def get_transforms(image_size: int = 512, augment: bool = False) -> dict[str, int | bool]:
    """返回数据集使用的尺寸和是否增强。几何增强在数据集内与掩膜同步进行。"""

    return {"image_size": image_size, "augment": augment}


class SegmentationDataset(Dataset):
    """从平行目录读取影像-掩膜对。__getitem__ 返回 image、mask、path。"""

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        image_size: int = 512,
        augment: bool = False,
        norm_stats: dict[str, list[float]] | None = None,
        imagenet_norm: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.augment = augment
        self.norm_stats = norm_stats
        self.imagenet_norm = imagenet_norm
        masks = {path.stem: path for path in list_images(self.mask_dir)}
        self.pairs: list[tuple[Path, Path]] = []
        missing = []
        for image_path in list_images(self.image_dir):
            mask_path = masks.get(image_path.stem)
            if mask_path is None:
                missing.append(image_path.name)
            else:
                self.pairs.append((image_path, mask_path))
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(f"这些影像没有同名掩膜：{preview}")
        if not self.pairs:
            raise FileNotFoundError(f"在 {self.image_dir} 中未找到图像文件")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path = self.pairs[index]
        image = resize_image(to_hwc(read_array(image_path)), self.image_size)
        mask = resize_mask(read_mask(mask_path), self.image_size)
        stats = self.norm_stats or compute_norm_stats([image_path])
        image = scale_to_unit(image, stats)
        if self.augment:
            image, mask = self._augment(image, mask)
        if self.imagenet_norm:
            image = apply_imagenet(image)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).long(),
            "path": str(image_path),
        }

    @staticmethod
    def _augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """第一次提交用的增强：水平/垂直翻转、90 度旋转、亮度 ±10%。"""

        if random.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        if random.random() < 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
        turns = random.randint(0, 3)
        if turns:
            image = np.rot90(image, turns)
            mask = np.rot90(mask, turns)
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        if random.random() < 0.5:
            image = np.clip(image * np.float32(random.uniform(0.9, 1.1)), 0.0, 1.0)
        return image, mask


def create_dataloaders(
    image_dir: Path,
    mask_dir: Path,
    batch_size: int = 4,
    image_size: int = 512,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
    config: Config | None = None,
) -> tuple[DataLoader, DataLoader]:
    """按文件名配对后划分训练/验证集，并把归一化分位数写回 config。"""

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    probe = SegmentationDataset(image_dir, mask_dir, image_size=image_size, augment=False)
    if config is not None and config.norm_low is not None and config.norm_high is not None:
        norm_stats = {"p_low": list(config.norm_low), "p_high": list(config.norm_high)}
    else:
        norm_stats = compute_norm_stats([image for image, _ in probe.pairs], seed=seed)
        if config is not None:
            config.norm_low = tuple(norm_stats["p_low"])  # type: ignore[assignment]
            config.norm_high = tuple(norm_stats["p_high"])  # type: ignore[assignment]
    imagenet_norm = bool(config.imagenet_norm) if config is not None else False

    count = len(probe)
    val_size = max(1, int(round(count * val_split))) if count > 1 else 0
    val_size = min(val_size, count - 1) if count > 1 else 0
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(count, generator=generator).tolist()
    train_index = order[val_size:]
    val_index = order[:val_size]

    train_base = SegmentationDataset(
        image_dir, mask_dir, image_size, augment=True, norm_stats=norm_stats, imagenet_norm=imagenet_norm
    )
    val_base = SegmentationDataset(
        image_dir, mask_dir, image_size, augment=False, norm_stats=norm_stats, imagenet_norm=imagenet_norm
    )
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        Subset(train_base, train_index),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        Subset(val_base, val_index) if val_index else val_base,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader


def generate_demo_dataset(root: Path, count: int = 4, size: int = 128, seed: int = 0) -> Path:
    """生成带 train/images 与 train/masks 的合成数据，只用于确认训练入口。"""

    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(seed)
    image_dir = root / "train" / "images"
    mask_dir = root / "train" / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 3,
        "dtype": "uint16",
        "crs": "EPSG:4326",
        "transform": from_origin(114.8, 30.6, 0.0001, 0.0001),
    }
    for index in range(count):
        grid_y, grid_x = np.mgrid[0:size, 0:size]
        road = (np.abs(grid_y - grid_x) <= 2).astype(np.uint8)
        image = rng.integers(6000, 9000, size=(3, size, size), dtype=np.uint16)
        image[:, road.astype(bool)] = np.array([21000, 20500, 20000], dtype=np.uint16)[:, None]
        with rasterio.open(image_dir / f"demo_{index:03d}.tif", "w", **profile) as dataset:
            dataset.write(image)
        mask_profile = profile.copy()
        mask_profile.update(count=1, dtype="uint8")
        with rasterio.open(mask_dir / f"demo_{index:03d}.tif", "w", **mask_profile) as dataset:
            dataset.write(road, 1)
    return root
