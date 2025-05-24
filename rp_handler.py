import runpod
import os
import subprocess
import base64
import time
import asyncio
import concurrent.futures
from typing import Dict, Any, Optional, Tuple
import logging

# Import third-party libraries
import cv2
import numpy as np
import filetype
from PIL import Image
import io
import requests

# Import MinIO configuration
from minio_config import (
    upload_file_to_minio, 
    cleanup_local_file, 
    health_check,
    get_minio_stats,
    MinIOConfig
)

# ==== Cấu hình Logging ====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==== Constants ====
INPUT_DIR = "/tmp/input"
OUTPUT_DIR = "/tmp/output"
CHECKPOINTS_ROOT = "./checkpoints/ditto_trt_Ampere_Plus"
CFG_PKL_PATH = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"

# Performance settings
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB limit
DOWNLOAD_TIMEOUT = 60  # seconds
SUBPROCESS_TIMEOUT = 300  # 5 minutes

class TalkingHeadProcessor:
    """Encapsulated processor cho talking head generation"""
    
    def __init__(self):
        self.setup_directories()
        self.validate_environment()
    
    @staticmethod
    def setup_directories():
        """Tạo thư mục cần thiết"""
        for directory in [INPUT_DIR, OUTPUT_DIR]:
            os.makedirs(directory, exist_ok=True)
            os.chmod(directory, 0o777)  # Ensure write permissions
    
    @staticmethod
    def validate_environment():
        """Validate environment và model checkpoints"""
        if not os.path.exists(CHECKPOINTS_ROOT):
            raise FileNotFoundError(f"Model checkpoints not found: {CHECKPOINTS_ROOT}")
        
        if not os.path.exists(CFG_PKL_PATH):
            raise FileNotFoundError(f"Config file not found: {CFG_PKL_PATH}")
        
        logger.info("Environment validation passed")

class FileHandler:
    """Handler cho file operations với error handling nâng cao"""
    
    @staticmethod
    def read_file_as_base64(file_path: str) -> str:
        """Đọc file và encode base64 với error handling"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return ""
        
        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logger.warning(f"File is empty: {file_path}")
                return ""
            
            if file_size > MAX_FILE_SIZE:
                logger.error(f"File too large ({file_size} bytes): {file_path}")
                raise ValueError(f"File size exceeds limit: {file_size} > {MAX_FILE_SIZE}")
            
            with open(file_path, "rb") as f:
                content = f.read()
                encoded = base64.b64encode(content).decode("utf-8")
                logger.info(f"Successfully encoded {file_path} to base64 ({len(encoded)} chars)")
                return encoded
                
        except Exception as e:
            logger.error(f"Error encoding file {file_path} to base64: {e}")
            raise

    @staticmethod
    def save_base64_to_file(base64_data: str, file_path: str):
        """Decode base64 và lưu file với validation"""
        try:
            # Validate base64 format
            if not base64_data or len(base64_data) < 4:
                raise ValueError("Invalid base64 data")
            
            # Decode với padding correction
            padding = 4 - len(base64_data) % 4
            if padding != 4:
                base64_data += '=' * padding
            
            file_bytes = base64.b64decode(base64_data)
            
            # Validate decoded data
            if len(file_bytes) == 0:
                raise ValueError("Decoded data is empty")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, "wb") as f:
                f.write(file_bytes)
            
            logger.info(f"Saved base64 data to {file_path} ({len(file_bytes)} bytes)")
            
        except Exception as e:
            logger.error(f"Error saving base64 to file {file_path}: {e}")
            raise

    @staticmethod
    def is_video(file_bytes: bytes) -> bool:
        """Validate video file format"""
        try:
            if len(file_bytes) < 1024:  # Minimum size check
                return False
            
            kind = filetype.guess(file_bytes)
            if kind is None:
                return False
            
            video_mimes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv']
            return kind.mime in video_mimes
            
        except Exception as e:
            logger.error(f"Error validating video format: {e}")
            return False

    @staticmethod
    def download_file_with_retry(url: str, destination_path: str, max_retries: int = 3) -> bool:
        """Download file với retry logic và progress tracking"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                
                # Download với timeout và streaming
                with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
                    response.raise_for_status()
                    
                    # Check content length
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > MAX_FILE_SIZE:
                        raise ValueError(f"File too large: {content_length} bytes")
                    
                    with open(destination_path, 'wb') as f:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                
                                # Check size limit during download
                                if downloaded > MAX_FILE_SIZE:
                                    raise ValueError(f"Download size exceeded limit: {downloaded}")
                
                # Validate downloaded file
                file_size = os.path.getsize(destination_path)
                if file_size == 0:
                    raise ValueError("Downloaded file is empty")
                
                logger.info(f"Successfully downloaded {url} to {destination_path} ({file_size} bytes)")
                return True
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                
                # Cleanup partial download
                if os.path.exists(destination_path):
                    try:
                        os.remove(destination_path)
                    except:
                        pass
                
                if attempt == max_retries - 1:
                    logger.error(f"All download attempts failed for {url}")
                    return False
                
                # Wait before retry
                time.sleep(2 ** attempt)
        
        return False

class InferenceRunner:
    """Handler cho việc chạy inference với performance monitoring"""
    
    @staticmethod
    def run_inference(audio_path: str, source_path: str, output_path: str) -> Tuple[bool, str]:
        """Chạy inference với subprocess và monitoring"""
        cmd = [
            "python", "inference.py",
            "--data_root", CHECKPOINTS_ROOT,
            "--cfg_pkl", CFG_PKL_PATH,
            "--audio_path", audio_path,
            "--source_path", source_path,
            "--output_path", output_path
        ]
        
        logger.info(f"Running inference: {' '.join(cmd)}")
        start_time = time.time()
        
        try:
            # Run với timeout
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=SUBPROCESS_TIMEOUT,
                cwd="/app"
            )
            
            duration = time.time() - start_time
            logger.info(f"Inference completed in {duration:.2f}s")
            
            if result.stdout:
                logger.info(f"Inference stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"Inference stderr: {result.stderr}")
            
            # Validate output
            if not os.path.exists(output_path):
                return False, "Output file not created"
            
            output_size = os.path.getsize(output_path)
            if output_size == 0:
                return False, "Output file is empty"
            
            logger.info(f"Inference successful: {output_path} ({output_size} bytes)")
            return True, f"Success in {duration:.2f}s"
            
        except subprocess.TimeoutExpired:
            error_msg = f"Inference timeout after {SUBPROCESS_TIMEOUT}s"
            logger.error(error_msg)
            return False, error_msg
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Inference failed with code {e.returncode}: {e.stderr}"
            logger.error(error_msg)
            return False, error_msg
            
        except Exception as e:
            error_msg = f"Unexpected inference error: {e}"
            logger.error(error_msg)
            return False, error_msg

# ==== Main Handler Function ====
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless handler với MinIO integration và error handling nâng cao.
    
    Workflow: Input Processing → Inference → MinIO Upload → Response
    """
    request_start_time = time.time()
    processor = TalkingHeadProcessor()
    file_handler = FileHandler()
    inference_runner = InferenceRunner()
    
    try:
        # ==== Input Validation ====
        job_input = event.get("input")
        if not job_input:
            return {"error": "No input data provided in the request"}
        
        # ==== MinIO Health Check ====
        minio_healthy, minio_status = health_check()
        if not minio_healthy:
            logger.warning(f"MinIO health check failed: {minio_status}")
            # Vẫn tiếp tục, sẽ fallback sang base64 nếu cần
        
        # ==== Process Audio Input ====
        audio_path = os.path.join(INPUT_DIR, f"audio_{int(time.time())}.wav")
        
        audio_base64 = job_input.get("audio_base64")
        audio_url = job_input.get("audio_url")
        
        if audio_url:
            logger.info(f"Processing audio from URL: {audio_url}")
            if not file_handler.download_file_with_retry(audio_url, audio_path):
                return {"error": f"Failed to download audio from URL: {audio_url}"}
        elif audio_base64:
            logger.info("Processing audio from base64")
            try:
                file_handler.save_base64_to_file(audio_base64, audio_path)
            except Exception as e:
                return {"error": f"Failed to process audio base64: {e}"}
        else:
            return {"error": "Missing audio input. Provide audio_base64 or audio_url"}
        
        # ==== Process Source Input ====
        source_path = None
        source_info = {}
        timestamp = int(time.time())
        
        # Xử lý theo priority: image_url → image_base64 → video_url → video_base64
        image_base64 = job_input.get("image_base64")
        image_url = job_input.get("image_url")
        video_base64 = job_input.get("video_base64")
        video_url = job_input.get("video_url")
        
        if not any([image_base64, image_url, video_base64, video_url]):
            return {"error": "Missing source data. Provide image_base64/url or video_base64/url"}
        
        # Process image input
        if image_url:
            source_path = os.path.join(INPUT_DIR, f"source_image_{timestamp}.png")
            logger.info(f"Processing image from URL: {image_url}")
            if not file_handler.download_file_with_retry(image_url, source_path):
                return {"error": f"Failed to download image from URL: {image_url}"}
            source_info = {"type": "image", "source": "url"}
            
        elif image_base64:
            source_path = os.path.join(INPUT_DIR, f"source_image_{timestamp}.png")
            logger.info("Processing image from base64")
            try:
                file_handler.save_base64_to_file(image_base64, source_path)
                source_info = {"type": "image", "source": "base64"}
            except Exception as e:
                return {"error": f"Failed to process image base64: {e}"}
        
        # Process video input
        elif video_url:
            source_path = os.path.join(INPUT_DIR, f"source_video_{timestamp}.mp4")
            logger.info(f"Processing video from URL: {video_url}")
            if not file_handler.download_file_with_retry(video_url, source_path):
                return {"error": f"Failed to download video from URL: {video_url}"}
            
            # Validate video format
            try:
                with open(source_path, 'rb') as f:
                    if not file_handler.is_video(f.read()):
                        cleanup_local_file(source_path, force=True)
                        return {"error": f"Downloaded file from {video_url} is not a valid video"}
                source_info = {"type": "video", "source": "url"}
            except Exception as e:
                return {"error": f"Error validating video from {video_url}: {e}"}
                
        elif video_base64:
            source_path = os.path.join(INPUT_DIR, f"source_video_{timestamp}.mp4")
            logger.info("Processing video from base64")
            try:
                video_bytes = base64.b64decode(video_base64)
                if not file_handler.is_video(video_bytes):
                    return {"error": "Invalid video_base64 data. Not recognized as video"}
                
                with open(source_path, "wb") as f:
                    f.write(video_bytes)
                source_info = {"type": "video", "source": "base64"}
            except Exception as e:
                return {"error": f"Failed to process video base64: {e}"}
        
        # Final validation
        if not source_path or not os.path.exists(source_path):
            return {"error": "Failed to create source file"}
        
        # ==== Run Inference ====
        output_path = os.path.join(OUTPUT_DIR, f"result_{timestamp}.mp4")
        
        logger.info("Starting inference process")
        inference_success, inference_message = inference_runner.run_inference(
            audio_path, source_path, output_path
        )
        
        if not inference_success:
            # Cleanup input files
            cleanup_local_file(audio_path, force=True)
            cleanup_local_file(source_path, force=True)
            return {"error": f"Inference failed: {inference_message}"}
        
        # ==== Upload to MinIO ====
        try:
            logger.info("Uploading result to MinIO")
            direct_url = upload_file_to_minio(output_path)
            
            # Cleanup files after successful upload
            cleanup_local_file(output_path)
            cleanup_local_file(audio_path, force=True)
            cleanup_local_file(source_path, force=True)
            
            # Success response
            total_time = time.time() - request_start_time
            logger.info(f"Request completed successfully in {total_time:.2f}s")
            
            return {
                "output_url": direct_url,
                "status": "success",
                "message": f"Video processed and uploaded successfully",
                "processing_time": f"{total_time:.2f}s",
                "source_info": source_info,
                "minio_stats": get_minio_stats()
            }
            
        except Exception as minio_error:
            logger.error(f"MinIO upload failed: {minio_error}")
            
            # Fallback: Trả về base64 nếu MinIO failed
            try:
                logger.info("Falling back to base64 response")
                video_base64 = file_handler.read_file_as_base64(output_path)
                
                if not video_base64:
                    return {"error": f"MinIO upload failed and base64 fallback failed"}
                
                # Cleanup after base64 encode
                cleanup_local_file(output_path, force=True)
                cleanup_local_file(audio_path, force=True)
                cleanup_local_file(source_path, force=True)
                
                total_time = time.time() - request_start_time
                
                return {
                    "output": video_base64,
                    "status": "partial_success",
                    "message": "Video processed but MinIO upload failed, returned base64",
                    "warning": f"MinIO error: {minio_error}",
                    "processing_time": f"{total_time:.2f}s",
                    "source_info": source_info
                }
                
            except Exception as backup_error:
                # Cleanup on complete failure
                cleanup_local_file(output_path, force=True)
                cleanup_local_file(audio_path, force=True)
                cleanup_local_file(source_path, force=True)
                
                return {
                    "error": f"Both MinIO upload and base64 fallback failed. "
                           f"MinIO: {minio_error}, Fallback: {backup_error}"
                }

    except Exception as e:
        logger.error(f"Unexpected error in handler: {e}")
        
        # Emergency cleanup
        try:
            cleanup_local_file(audio_path if 'audio_path' in locals() else "", force=True)
            cleanup_local_file(source_path if 'source_path' in locals() else "", force=True)
            cleanup_local_file(output_path if 'output_path' in locals() else "", force=True)
        except:
            pass
        
        return {"error": f"Unexpected server error: {str(e)}"}

# ==== Health Check Endpoint ====
def health_handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """Health check endpoint cho RunPod"""
    try:
        # Environment check
        processor = TalkingHeadProcessor()
        
        # MinIO check
        minio_healthy, minio_status = health_check()
        
        # System stats
        import psutil
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {
                "environment": "ok",
                "minio": minio_status,
                "minio_healthy": minio_healthy
            },
            "system": {
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "cpu_count": psutil.cpu_count()
            },
            "minio_stats": get_minio_stats()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

# ==== RunPod Integration ====
if __name__ == "__main__":
    # Initialize và validate environment
    try:
        logger.info("Initializing TalkingHead service...")
        processor = TalkingHeadProcessor()
        
        # Validate MinIO connection
        minio_healthy, minio_status = health_check()
        logger.info(f"MinIO status: {minio_status}")
        
        logger.info("Service initialization completed")
        
        # Start RunPod serverless
        runpod.serverless.start({
            "handler": handler,
            "health": health_handler
        })
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise

# Alternative: Uncomment cho RunPod deployment
# runpod.serverless.start({"handler": handler})
