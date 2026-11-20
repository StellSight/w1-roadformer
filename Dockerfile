FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
WORKDIR /app

# 设置国内高速源
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# 安装gdal工具
RUN apt update -y && apt install -y gdal-bin

# 安装必要第三方库
RUN pip install --no-cache-dir \
    rasterio \
    numpy>=1.24 \
    pillow>=9.0 \
    matplotlib>=3.7 \
    tqdm>=4.65 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

#复制代码、权重和设置环境变量
COPY segmentation/ ./segmentation/
COPY pretrained.pth ./pretrained.pth
ENV PYTHONPATH=/app

# 设置镜像entrypoint（必须设置，否则无法成功执行）
ENTRYPOINT ["python","-m","segmentation.inference","--pretrained","./pretrained.pth"]
