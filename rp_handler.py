import runpod
import os
import subprocess
import base64
import cv2
import numpy as np
import filetype
from PIL import Image
import io
import requests

# --- Helper Functions ---

def read_file_as_base64(file_path):
    """Đọc nội dung file và mã hóa Base64"""
    if not os.path.exists(file_path):
        # Trả về chuỗi rỗng nếu file không tồn tại (ví dụ: file output lỗi)
        # In cảnh báo hoặc lỗi ở nơi gọi hàm này nếu cần xử lý cụ thể hơn
        return ""
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding file {file_path} to base64: {e}")
        raise # Re-raise the exception


def save_base64_to_file(base64_data, file_path):
    """Giải mã Base64 và lưu vào file"""
    try:
        file_bytes = base64.b64decode(base64_data)
        with open(file_path, "wb") as f:
            f.write(file_bytes)
    except Exception as e:
        print(f"Error saving base64 to file {file_path}: {e}")
        raise # Re-raise the exception


def is_video(file_bytes):
    """Xác định xem dữ liệu có phải là video hay không dựa trên magic numbers"""
    # Sử dụng filetype để kiểm tra định dạng
    kind = filetype.guess(file_bytes)
    if kind is None:
        return False
    return kind.mime.startswith('video/')

def download_file(url, destination_path):
    """Tải file từ URL về đường dẫn đích"""
    try:
        # Tạo thư mục đích nếu chưa tồn tại
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Successfully downloaded {url} to {destination_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from {url}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during download from {url}: {e}")
        return False


# --- RunPod Handler ---

def handler(event):
    """
    RunPod Serverless handler.
    Nhận input audio và source (ảnh/video) qua base64 hoặc URLs,
    tải file nếu cần, chạy inference.py, và trả về video kết quả dưới dạng base64.
    """
    try:
        job_input = event.get("input")
        if not job_input:
            return {"error": "No input data provided in the request."}

        # Tạo thư mục tạm thời cho input/output
        input_dir = "/tmp/input"
        output_dir = "/tmp/output"
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # --- Xử lý Input Audio (URL hoặc Base64) ---
        audio_path = os.path.join(input_dir, "audio.wav") # Đường dẫn file audio tạm thời
        audio_base64 = job_input.get("audio_base64")
        audio_url = job_input.get("audio_url")

        if audio_url:
            # Ưu tiên URL nếu cả hai đều có
            if not download_file(audio_url, audio_path):
                 return {"error": f"Failed to download audio from URL: {audio_url}"}
            # Có thể thêm bước kiểm tra định dạng audio sau khi tải nếu cần
        elif audio_base64:
            try:
                save_base64_to_file(audio_base64, audio_path)
            except Exception as e:
                 return {"error": f"Failed to save audio file from base64: {e}"}
        else:
            return {"error": "Missing audio input. Provide audio_base64 or audio_url."}


        # --- Xử lý Input Source (URL hoặc Base64 - Ảnh hoặc Video) ---
        source_file_path = None # Biến sẽ lưu đường dẫn của file nguồn tạm thời
        source_info = {} # Thông tin về nguồn (để trả về trong response)

        image_base64 = job_input.get("image_base64")
        image_url = job_input.get("image_url")
        video_base64 = job_input.get("video_base64")
        video_url = job_input.get("video_url")

        # Kiểm tra xem người dùng có cung cấp input nguồn nào không
        if not (image_base64 or image_url or video_base64 or video_url):
             return {"error": "Missing source data. Provide image_base64/url or video_base64/url."}

        # Xử lý theo thứ tự ưu tiên: image_url -> image_base64 -> video_url -> video_base64
        if image_url:
            source_file_path = os.path.join(input_dir, "source_image.png") # Tên file nguồn ảnh tạm thời
            if not download_file(image_url, source_file_path):
                 return {"error": f"Failed to download image from URL: {image_url}"}
            source_info["type"] = "image"

        elif image_base64:
            source_file_path = os.path.join(input_dir, "source_image.png")
            try:
                save_base64_to_file(image_base64, source_file_path)
            except Exception as e:
                 return {"error": f"Failed to save image file from base64: {e}"}
            source_info["type"] = "image"

        elif video_url:
            source_file_path = os.path.join(input_dir, "source_video.mp4") # Tên file nguồn video tạm thời
            if not download_file(video_url, source_file_path):
                 return {"error": f"Failed to download video from URL: {video_url}"}
            # Kiểm tra định dạng video sau khi tải
            try:
                with open(source_file_path, 'rb') as f:
                     if not is_video(f.read()):
                          # Xóa file đã tải về nếu không phải video
                          os.remove(source_file_path)
                          return {"error": f"Downloaded file from {video_url} is not a valid video."}
            except Exception as e:
                 print(f"Error verifying downloaded video: {e}")
                 return {"error": f"Error verifying downloaded video from {video_url}: {e}"}

            source_info["type"] = "video"

        elif video_base64:
            source_file_path = os.path.join(input_dir, "source_video.mp4")
            try:
                video_bytes = base64.b64decode(video_base64)
                if not is_video(video_bytes):
                    return {"error": "Invalid video_base64 data. Not recognized as a video file."}
                with open(source_file_path, "wb") as f:
                    f.write(video_bytes)
            except Exception as e:
                 return {"error": f"Failed to save video file from base64: {e}"}

            source_info["type"] = "video"

        else:
            # Lẽ ra đã bị bắt ở kiểm tra ban đầu, nhưng để đề phòng
            return {"error": "Internal error: No valid source input found."}

        # Đảm bảo source_file_path đã được thiết lập
        if source_file_path is None or not os.path.exists(source_file_path):
             # Kiểm tra lại lần nữa để đảm bảo file nguồn tạm thời đã được tạo
             return {"error": f"Internal error: Failed to create temporary source file at {source_file_path}."}


        # --- Chuẩn bị đường dẫn Output ---
        output_path = os.path.join(output_dir, "result.mp4") # Đường dẫn file output tạm thời
        # os.makedirs(os.path.dirname(output_path), exist_ok=True) # output_dir đã được tạo ở trên


        # --- Chạy mô hình Ditto sử dụng subprocess ---
        checkpoints_root = "./checkpoints/ditto_trt_Ampere_Plus" # Path relative to WORKDIR /app
        cfg_pkl_path = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" # Path relative to WORKDIR /app

        # Kiểm tra nhanh xem các thư mục checkpoints có tồn tại không
        if not os.path.exists(checkpoints_root) or not os.path.exists(cfg_pkl_path):
             return {"error": f"Model checkpoints not found. Expected at {checkpoints_root} and {cfg_pkl_path}. Ensure they are cloned/copied correctly in the Dockerfile."}


        cmd = [
            "python", "inference.py",
            "--data_root", checkpoints_root,
            "--cfg_pkl", cfg_pkl_path,
            "--audio_path", audio_path,      # Truyền đường dẫn audio tạm thời (đã tải/lưu)
            "--source_path", source_file_path, # <--- Truyền đường dẫn source (ảnh/video) tạm thời (đã tải/lưu)
            "--output_path", output_path      # Truyền đường dẫn output tạm thời
        ]

        print(f"Running inference command: {' '.join(cmd)}") # Log command để debug

        # Thực thi lệnh inference
        try:
            # check=True: Raise exception nếu lệnh thất bại
            # capture_output=True, text=True: Bắt stdout/stderr dưới dạng string
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Inference subprocess stdout:", result.stdout)
            if result.stderr:
                 # Log stderr ngay cả khi lệnh thành công, có thể có cảnh báo
                 print("Inference subprocess stderr:", result.stderr)

        except FileNotFoundError:
             # Lỗi này xảy ra nếu lệnh 'python' hoặc 'inference.py' không tìm thấy
             return {"error": "Inference script or Python executable not found in container. Ensure Python and inference.py are in PATH and working directory."}
        except subprocess.CalledProcessError as e:
            # Lỗi này xảy ra nếu lệnh inference.py chạy nhưng trả về mã lỗi khác 0
            print(f"Inference subprocess failed with return code {e.returncode}")
            print("Subprocess stdout:", e.stdout)
            print("Subprocess stderr:", e.stderr)
            return {"error": f"Inference process failed. Check logs for details. Stderr: {e.stderr}"}
        except Exception as e:
             # Các lỗi khác khi chạy subprocess
             print(f"An error occurred while running inference subprocess: {e}")
             return {"error": f"An unexpected error occurred during inference execution: {str(e)}"}

        # --- Đọc file Output và Trả về kết quả ---
        # Kiểm tra xem file output có được tạo và có nội dung không
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
             # Thử đọc stderr/stdout từ subprocess nếu có để cung cấp thêm thông tin lỗi
             error_output = result.stderr or result.stdout if 'result' in locals() else "No subprocess output captured."
             return {"error": f"Output file was not created or is empty by inference.py at {output_path}. Check subprocess logs for details. Output: {error_output[:500]}..."} # Giới hạn độ dài log lỗi


        try:
            # Đọc nội dung file output và mã hóa base64
            video_base64 = read_file_as_base64(output_path)
            if not video_base64:
                 # read_file_as_base64 có thể trả về rỗng nếu file rỗng hoặc lỗi đọc
                 return {"error": f"Failed to read output file or file is empty after creation: {output_path}"}
        except Exception as e:
             return {"error": f"Failed to encode output file to base64: {e}"}


        # Trả về kết quả thành công
        result = {"output": video_base64}
        if source_info: # Thêm thông tin loại nguồn vào response
            result["source_info"] = source_info

        # RunPod Serverless sẽ tự động dọn dẹp thư mục /tmp sau khi handler hoàn thành

        return result

    except Exception as e:
        # Xử lý các lỗi không được bắt riêng ở trên
        print(f"An unexpected error occurred in handler: {e}")
        # traceback.print_exc() # Uncomment để in full traceback khi debug
        return {"error": f"An unexpected error occurred in handler: {str(e)}"}

# Uncomment cho RunPod deployment
runpod.serverless.start({"handler": handler})
