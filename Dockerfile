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
    wget \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập môi trường Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN python -m pip install --upgrade pip

# Copy source code và cài đặt dependencies
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt --no-cache-dir

# Tải pretrained models
RUN wget https://github.com/antgroup/ditto-talkinghead/releases/download/v0.4/ditto_trt_Ampere_Plus.zip -P ./checkpoints/ \
    && unzip ./checkpoints/ditto_trt_Ampere_Plus.zip -d ./checkpoints/ \
    && rm ./checkpoints/ditto_trt_Ampere_Plus.zip

EXPOSE 8000
CMD ["python", "-u", "/app/rp_handler.py"]
