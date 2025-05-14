FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Cài đặt các thư viện hệ thống
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libxrender-dev \
    libopenblas-dev \
    ninja-build \
    git \
    git-lfs \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập Git LFS
RUN git lfs install

# Thiết lập python
RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN python -m pip install --upgrade pip

# Cài đặt PyTorch và torchvision phù hợp CUDA 11.8
RUN pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Copy code vào container
WORKDIR /app
COPY . .

# Cài đặt các requirements còn lại
RUN pip install -r requirements.txt --no-cache-dir

# Clone checkpoints từ HuggingFace (dùng Git LFS)
RUN mkdir -p checkpoints && \
    cd checkpoints && \
    git clone https://huggingface.co/digital-avatar/ditto-talkinghead . && \
    rm -rf .git

EXPOSE 8000
CMD ["python", "-u", "/app/rp_handler.py"]
