# Base image với CUDA 11.8 và cuDNN 8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Cài đặt dependencies hệ thống
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    git-lfs \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập Git LFS
RUN git lfs install

# Thiết lập môi trường Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN python -m pip install --upgrade pip

# Copy source code và cài đặt dependencies
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt --no-cache-dir

# Tải pretrained models từ Hugging Face
RUN mkdir -p checkpoints && \
    cd checkpoints && \
    git clone https://huggingface.co/digital-avatar/ditto-talkinghead && \
    mv ditto-talkinghead/* . && \
    rm -rf ditto-talkinghead .git

EXPOSE 8000
CMD ["python", "-u", "/app/rp_handler.py"]
