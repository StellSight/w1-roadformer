# W1-RoadFormer

武汉一号 0.5 米 RGB 影像的道路语义分割模型。编码器为 SegFormer MiT-B2，解码器把四个尺度的特征投影到同一通道后融合，输出与输入同尺寸的背景/道路两类 logits。

这个目录对应赛道三样例要求的工程结构。推理入口是 `segmentation/inference.py`。

## 环境

Python 3.10 及以上。先装与 CUDA 匹配的 PyTorch，再装其余依赖。运行时镜像不能联网，依赖要在构建镜像时装好。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pillow tifffile
```

CPU 机器只做流程验证时，把上面的索引换成 `https://download.pytorch.org/whl/cpu`。

## 数据

训练影像和掩膜按文件名（不含扩展名）配对。掩膜里大于 0 的像素视为道路。

```text
data/
  train/
    images/   # uint16 RGB，tif 或 png，官方切片为 512×512
    masks/
  val/
    images/
    masks/
  test/
    images/
```

有独立验证目录时用验证目录。没有时，程序会按文件名排序留出末尾 10% 做验证，并在日志里说明。正式训练请按场景划分，不要把相邻切片分到训练集和验证集两边。

UInt16 归一化使用训练集的 2% 和 98% 分位数，裁剪后映射到 [0, 1]。加载 MiT-B2 预训练权重时，会在此之后再做 ImageNet 均值方差归一化。验证和推理使用训练时写进检查点的同一组参数。

## 训练

平台传入数据集根目录，影像在 `train/images`，掩膜在 `train/masks`。最佳权重写到 `--out_weights`。

```bash
python -m segmentation.train --sample_datasets /dir/to/dataset --out_weights /dir/to/best.pth --epochs 80 --batch-size 4
```

有 MiT-B2 编码器预训练权重时加上 `--encoder-weights weights/mit_b2.pth`。

确认入口能跑通，不使用赛题影像：

```bash
python -m segmentation.train --demo --epochs 1 --image-size 128 --batch-size 2 --num-workers 0
```

## 推理

平台传入的是数据集根目录，影像在 `test/images` 下。输出为同名 `.tif`，单波段 uint8，0 是背景，1 是道路。

```bash
python -m segmentation.inference --pretrained pretrained.pth --input_dataset /dir/to/dataset --output_dir /dir/to/output_dir
```

```bash
python -m segmentation.inference -h
```

## 提交镜像

先把 `--out_weights` 得到的权重复制成项目根目录的 `pretrained.pth`。镜像构建时会把它复制进去，平台运行时不再另传权重。容器离线运行，依赖在构建阶段安装，镜像需小于 20GB。

```powershell
copy D:\path\to\best.pth pretrained.pth
.\pack.ps1
```

`pack.ps1` 会执行：

```bash
docker build -t w1-roadformer:latest .
docker save w1-roadformer:latest -o w1-roadformer.tar
```

上传 `w1-roadformer.tar`。本地先用官方那条命令确认能跑通：

```bash
docker run --rm --gpus all -v D:/data:/data w1-roadformer:latest --input_dataset /data/dataset --output_dir /data/output
```

其中 `/data/dataset/test/images` 下放测试影像。

当前仓库里的 `pretrained.pth` 只是未训练的初始权重，用来保证镜像能启动。用它打榜没有精度。正式提交前必须换成在 OGE 上训练得到的 `best.pt`。

## 和初赛方案的关系

初赛方案名是 W1-TopoRoadNet。W1-RoadFormer 是它现在的可训练实现：保留 SegFormer-B2、Focal + Dice、UInt16 分位归一化，以及翻转和 90° 旋转。方向卷积、边界辅助头和断裂修复留到基线指标稳定后再加。
