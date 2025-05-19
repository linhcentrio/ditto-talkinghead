import runpod
import os
import subprocess
import base64
import cv2  # Vẫn cần cv2 vì filetype có thể dùng nó để kiểm tra định dạng video
import numpy as np # Vẫn cần numpy cho các thư viện khác dùng đến
import filetype
from PIL import Image # Vẫn cần PIL cho các thư viện khác dùng đến
import io # Vẫn cần io cho các thư viện khác dùng đến

# --- Helper Functions ---

def read_file_as_base64(file_path):
    """Đọc nội dung file và mã hóa Base64"""
    if not os.path.exists(file_path):
        # Trả về chuỗi rỗng hoặc xử lý lỗi nếu file không tồn tại
        # print(f"Warning: File not found for base64 encoding: {file_path}")
        return "" # Hoặc raise FileNotFoundError(f"Output file not found: {file_path}")
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def save_base64_to_file(base64_data, file_path):
    """Giải mã Base64 và lưu vào file"""
    try:
        file_bytes = base64.b64decode(base64_data)
        with open(file_path, "wb") as f:
            f.write(file_bytes)
    except Exception as e:
        print(f"Error saving base64 to file {file_path}: {e}")
        raise # Re-raise the exception after logging


def is_video(file_bytes):
    """Xác định xem dữ liệu có phải là video hay không dựa trên magic numbers"""
    # Sử dụng filetype để kiểm tra định dạng dựa trên bytes
    # filetype rất hiệu quả cho việc này
    kind = filetype.guess(file_bytes)
    if kind is None:
        return False
    # filetype trả về mime type, kiểm tra xem nó có bắt đầu bằng 'video/' không
    return kind.mime.startswith('video/')

# Hàm extract_best_frame đã bị loại bỏ vì không còn cần thiết cho logic handler mới

# --- RunPod Handler ---

def handler(event):
    """
    RunPod Serverless handler.
    Receives input as base64 audio and source (image or video),
    saves them to temporary files, runs inference.py, and returns
    the output video as base64.
    """
    try:
        # Lấy đầu vào từ request payload
        job_input = event.get("input")
        if not job_input:
            return {"error": "No input data provided in the request."}

        # --- Kiểm tra và Lưu Audio ---
        audio_base64 = job_input.get("audio_base64")
        if not audio_base64:
            return {"error": "Missing audio_base64 in input."}

        # Tạo thư mục tạm thời để lưu các file input/output
        # RunPod thường cung cấp /tmp là writable, nhưng tạo thư mục con là an toàn
        input_dir = "/tmp/input"
        output_dir = "/tmp/output"
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Lưu audio vào file tạm thời
        audio_path = os.path.join(input_dir, "audio.wav") # Tên file audio tạm thời
        try:
            save_base64_to_file(audio_base64, audio_path)
        except Exception as e:
             return {"error": f"Failed to save audio file: {e}"}

        # --- Kiểm tra và Lưu Source (Ảnh hoặc Video) ---
        image_base64 = job_input.get("image_base64")
        video_base64 = job_input.get("video_base64")

        source_file_path = None # Biến sẽ lưu đường dẫn của file nguồn (ảnh hoặc video) tạm thời
        source_info = {} # Thông tin về nguồn (để trả về trong response)

        if image_base64 and video_base64:
             return {"error": "Provide either image_base64 or video_base64, not both."}
        elif image_base64:
            # Xử lý input là hình ảnh base64
            source_file_path = os.path.join(input_dir, "source_image.png") # Tên file nguồn ảnh tạm thời
            try:
                save_base64_to_file(image_base64, source_file_path)
            except Exception as e:
                 return {"error": f"Failed to save image file: {e}"}
            source_info["type"] = "image"

        elif video_base64:
            # Xử lý input là video base64
            source_file_path = os.path.join(input_dir, "source_video.mp4") # Tên file nguồn video tạm thời
            try:
                video_bytes = base64.b64decode(video_base64)
                # Kiểm tra định dạng trước khi lưu
                if not is_video(video_bytes):
                    return {"error": "Invalid video_base64 data. Not recognized as a video file."}

                # Lưu toàn bộ nội dung video vào file tạm thời
                with open(source_file_path, "wb") as f:
                    f.write(video_bytes)

            except Exception as e:
                 return {"error": f"Failed to save video file: {e}"}

            source_info["type"] = "video"
            # Không còn cần source_info["frame_used"] nữa vì xử lý toàn bộ video trong inference.py

        else:
            # Nếu không có cả image_base64 và video_base64
            return {"error": "Missing source data. Provide either image_base64 or video_base64."}

        # --- Chuẩn bị đường dẫn Output ---
        output_path = os.path.join(output_dir, "result.mp4") # Đường dẫn file output tạm thời

        # --- Chạy mô hình Ditto sử dụng subprocess ---
        # Đường dẫn tới các tệp models/cfg pkls được giả định là cố định bên trong container
        # sau khi build (ví dụ: clone vào /app/checkpoints)
        checkpoints_root = "./checkpoints/ditto_trt_Ampere_Plus" # Path relative to WORKDIR /app
        cfg_pkl_path = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" # Path relative to WORKDIR /app

        # Kiểm tra nhanh xem các thư mục checkpoints có tồn tại không
        if not os.path.exists(checkpoints_root) or not os.path.exists(cfg_pkl_path):
             return {"error": f"Model checkpoints not found. Expected at {checkpoints_root} and {cfg_pkl_path}. Ensure they are cloned correctly in the Dockerfile."}


        cmd = [
            "python", "inference.py",
            "--data_root", checkpoints_root,
            "--cfg_pkl", cfg_pkl_path,
            "--audio_path", audio_path,      # Đường dẫn audio tạm thời
            "--source_path", source_file_path, # <--- Truyền đường dẫn source (ảnh/video) tạm thời
            "--output_path", output_path      # Đường dẫn output tạm thời
        ]

        print(f"Running inference command: {' '.join(cmd)}") # In command để debug

        # Thực thi lệnh inference
        # capture_output=True để lấy stdout/stderr cho debug
        # text=True để stdout/stderr là string thay vì bytes
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Inference subprocess stdout:", result.stdout)
            if result.stderr:
                 print("Inference subprocess stderr:", result.stderr)

        except FileNotFoundError:
             # Lỗi này xảy ra nếu lệnh 'python' hoặc 'inference.py' không tìm thấy
             return {"error": "Inference script not found. Ensure Python and inference.py are in the container's PATH and working directory."}
        except subprocess.CalledProcessError as e:
            # Lỗi này xảy ra nếu lệnh inference.py chạy nhưng trả về mã lỗi khác 0
            print(f"Inference subprocess failed with return code {e.returncode}")
            print("Subprocess stdout:", e.stdout)
            print("Subprocess stderr:", e.stderr)
            return {"error": f"Inference process failed. Error details: {e.stderr or e.stdout or 'No output'}"}
        except Exception as e:
             # Các lỗi khác khi chạy subprocess
             print(f"An error occurred while running inference subprocess: {e}")
             return {"error": f"An error occurred during inference execution: {str(e)}"}

        # --- Đọc file Output và Trả về kết quả ---
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
             # Kiểm tra xem file output có được tạo và có nội dung không
             return {"error": f"Output file was not created or is empty by inference.py at {output_path}. Check subprocess logs for details."}

        try:
            video_base64 = read_file_as_base64(output_path)
            if not video_base64:
                 # Trường hợp read_file_as_base64 trả về rỗng nếu file không tồn tại (đã check ở trên)
                 # Hoặc nếu file rỗng sau khi đọc (ít khả năng xảy ra nếu getsize > 0)
                 return {"error": f"Failed to read output file or file is empty after creation: {output_path}"}
        except Exception as e:
             return {"error": f"Failed to encode output file to base64: {e}"}


        # Trả về kết quả
        result = {"output": video_base64}
        if source_info: # Thêm thông tin loại nguồn vào response
            result["source_info"] = source_info

        # RunPod Serverless sẽ tự động dọn dẹp /tmp sau khi handler hoàn thành

        return result

    except Exception as e:
        # Xử lý các lỗi không được bắt riêng
        print(f"An unexpected error occurred in handler: {e}")
        # traceback.print_exc() # Uncomment để in full traceback khi debug
        return {"error": f"An unexpected error occurred: {str(e)}"}

# Khởi tạo serverless function RunPod
runpod.serverless.start({"handler": handler})
