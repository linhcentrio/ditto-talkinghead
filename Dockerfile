# Sử dụng ảnh nền NVIDIA TensorRT với Python 3
FROM nvcr.io/nvidia/tensorrt:23.12-py3

# Thiết lập thư mục làm việc ban đầu cho ứng dụng bên trong container
WORKDIR /app

# --- Cài đặt các gói hệ thống cần cho build và runtime ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ffmpeg libsndfile1 build-essential python3-dev git-lfs && \
    rm -rf /var/lib/apt/lists/*

# --- Cài đặt các gói Python ---
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir \
    cuda-python \
    librosa \
    tqdm \
    imageio \
    opencv-python-headless \
    scikit-image \
    cython \
    imageio-ffmpeg \
    colored \
    numpy==2.0.1 \
    typing_extensions --upgrade \
    runpod \
    filetype \
    Pillow \
    requests

# --- Clone mã nguồn dự án ---
# Clone mã nguồn từ GitHub vào thư mục làm việc /app
RUN git clone https://github.com/antgroup/ditto-talkinghead .

# --- Biên dịch Cython Extensions ---
RUN cd core/utils/blend && \
    cython blend.pyx && \
    gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
      -I$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])") \
      -I$(python3 -c "import numpy; print(numpy.get_include())") \
      blend.c -o blend_impl.so

# --- Thiết lập Python Path ---
ENV PYTHONPATH=/app

# --- Copy Script Handler ---
COPY rp_handler.py /app/rp_handler.py

# --- Clone Checkpoints (Mô hình) ---
RUN git clone https://huggingface.co/digital-avatar/ditto-talkinghead /app/checkpoints

# --- Cấu hình lệnh chạy khi container bắt đầu ---
CMD ["python", "rp_handler.py"]

# --- Optional: Tạo thư mục tạm cho Input/Output ---
# RUN mkdir -p /tmp/input /tmp/output && chmod -R 777 /tmp/input /tmp/output
