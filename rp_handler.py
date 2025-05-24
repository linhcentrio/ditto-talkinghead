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
from minio_config import upload_file_to_minio, cleanup_local_file

# --- Helper Functions (giữ nguyên) ---
def read_file_as_base64(file_path):
    """Đọc nội dung file và mã hóa Base64"""
    if not os.path.exists(file_path):
        return ""
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding file {file_path} to base64: {e}")
        raise

def save_base64_to_file(base64_data, file_path):
    """Giải mã Base64 và lưu vào file"""
    try:
        file_bytes = base64.b64decode(base64_data)
        with open(file_path, "wb") as f:
            f.write(file_bytes)
    except Exception as e:
        print(f"Error saving base64 to file {file_path}: {e}")
        raise

def is_video(file_bytes):
    """Xác định xem dữ liệu có phải là video hay không"""
    kind = filetype.guess(file_bytes)
    if kind is None:
        return False
    return kind.mime.startswith('video/')

def download_file(url, destination_path):
    """Tải file từ URL về đường dẫn đích"""
    try:
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
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

# --- RunPod Handler với MinIO Integration ---
def handler(event):
    """
    RunPod Serverless handler với MinIO storage integration.
    Thay đổi: Upload output video lên MinIO và trả về direct URL thay vì base64.
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

        # --- Xử lý Input Audio (giữ nguyên logic) ---
        audio_path = os.path.join(input_dir, "audio.wav")
        audio_base64 = job_input.get("audio_base64")
        audio_url = job_input.get("audio_url")

        if audio_url:
            if not download_file(audio_url, audio_path):
                return {"error": f"Failed to download audio from URL: {audio_url}"}
        elif audio_base64:
            try:
                save_base64_to_file(audio_base64, audio_path)
            except Exception as e:
                return {"error": f"Failed to save audio file from base64: {e}"}
        else:
            return {"error": "Missing audio input. Provide audio_base64 or audio_url."}

        # --- Xử lý Input Source (giữ nguyên logic) ---
        source_file_path = None
        source_info = {}
        
        image_base64 = job_input.get("image_base64")
        image_url = job_input.get("image_url")
        video_base64 = job_input.get("video_base64")
        video_url = job_input.get("video_url")

        if not (image_base64 or image_url or video_base64 or video_url):
            return {"error": "Missing source data. Provide image_base64/url or video_base64/url."}

        # Logic xử lý source (giữ nguyên)
        if image_url:
            source_file_path = os.path.join(input_dir, "source_image.png")
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
            source_file_path = os.path.join(input_dir, "source_video.mp4")
            if not download_file(video_url, source_file_path):
                return {"error": f"Failed to download video from URL: {video_url}"}
            try:
                with open(source_file_path, 'rb') as f:
                    if not is_video(f.read()):
                        os.remove(source_file_path)
                        return {"error": f"Downloaded file from {video_url} is not a valid video."}
            except Exception as e:
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

        if source_file_path is None or not os.path.exists(source_file_path):
            return {"error": f"Internal error: Failed to create temporary source file at {source_file_path}."}

        # --- Chuẩn bị đường dẫn Output ---
        output_path = os.path.join(output_dir, "result.mp4")

        # --- Chạy mô hình Ditto (giữ nguyên logic) ---
        checkpoints_root = "./checkpoints/ditto_trt_Ampere_Plus"
        cfg_pkl_path = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"

        if not os.path.exists(checkpoints_root) or not os.path.exists(cfg_pkl_path):
            return {"error": f"Model checkpoints not found. Expected at {checkpoints_root} and {cfg_pkl_path}."}

        cmd = [
            "python", "inference.py",
            "--data_root", checkpoints_root,
            "--cfg_pkl", cfg_pkl_path,
            "--audio_path", audio_path,
            "--source_path", source_file_path,
            "--output_path", output_path
        ]

        print(f"Running inference command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Inference subprocess stdout:", result.stdout)
            if result.stderr:
                print("Inference subprocess stderr:", result.stderr)
        except FileNotFoundError:
            return {"error": "Inference script or Python executable not found in container."}
        except subprocess.CalledProcessError as e:
            print(f"Inference subprocess failed with return code {e.returncode}")
            print("Subprocess stdout:", e.stdout)
            print("Subprocess stderr:", e.stderr)
            return {"error": f"Inference process failed. Stderr: {e.stderr}"}
        except Exception as e:
            print(f"An error occurred while running inference subprocess: {e}")
            return {"error": f"An unexpected error occurred during inference execution: {str(e)}"}

        # --- Kiểm tra Output File ---
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            error_output = result.stderr or result.stdout if 'result' in locals() else "No subprocess output captured."
            return {"error": f"Output file was not created or is empty at {output_path}. Output: {error_output[:500]}..."}

        # --- *** THAY ĐỔI CHÍNH: Upload lên MinIO thay vì trả về Base64 *** ---
        try:
            # Upload file lên MinIO
            direct_url = upload_file_to_minio(output_path)
            
            # Cleanup local file sau khi upload thành công
            cleanup_local_file(output_path)
            
            print(f"Successfully uploaded output video to MinIO: {direct_url}")
            
            # Trả về response mới với URL thay vì base64
            result_response = {
                "output_url": direct_url,
                "status": "success",
                "message": "Video processed and uploaded successfully"
            }
            
            if source_info:
                result_response["source_info"] = source_info
                
            return result_response
            
        except Exception as e:
            print(f"Error uploading to MinIO: {e}")
            # Fallback: nếu MinIO upload failed, vẫn trả về base64 để không mất kết quả
            try:
                video_base64 = read_file_as_base64(output_path)
                if not video_base64:
                    return {"error": f"Failed to upload to MinIO and backup base64 encoding failed: {output_path}"}
                
                return {
                    "output": video_base64,
                    "source_info": source_info,
                    "warning": "MinIO upload failed, returned base64 as fallback"
                }
            except Exception as backup_e:
                return {"error": f"MinIO upload failed and base64 fallback failed: MinIO: {e}, Base64: {backup_e}"}

    except Exception as e:
        print(f"An unexpected error occurred in handler: {e}")
        return {"error": f"An unexpected error occurred in handler: {str(e)}"}

# Uncomment cho RunPod deployment
runpod.serverless.start({"handler": handler})
