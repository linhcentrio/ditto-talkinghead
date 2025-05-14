import runpod
import subprocess
import os
import boto3
from uuid import uuid4
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# S3 configuration (Thay bằng Cloudflare R2 nếu cần)
S3_ENDPOINT = os.getenv('S3_ENDPOINT', 'https://s3.amazonaws.com')
S3_BUCKET = os.getenv('S3_BUCKET', 'ditto-output')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')

s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

def download_file(url: str, save_path: str):
    """Download file từ URL"""
    try:
        subprocess.run(
            f"wget -q '{url}' -O {save_path}",
            shell=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        raise

def upload_to_s3(file_path: str) -> str:
    """Upload file lên S3-compatible storage"""
    object_name = f"outputs/{uuid4()}.mp4"
    try:
        s3_client.upload_file(
            file_path,
            S3_BUCKET,
            object_name,
            ExtraArgs={'ContentType': 'video/mp4'}
        )
        return f"{S3_ENDPOINT}/{S3_BUCKET}/{object_name}"
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise

def handler(job):
    """Main handler function cho RunPod"""
    try:
        input = job["input"]
        logger.info(f"Processing job: {job['id']}")
        
        # Tải input files
        download_file(input['audio_url'], '/tmp/audio.wav')
        download_file(input['image_url'], '/tmp/source.png')
        
        # Chạy inference
        cmd = [
            "python", "inference.py",
            "--data_root", "./checkpoints/ditto_trt_Ampere_Plus",
            "--cfg_pkl", "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl",
            "--audio_path", "/tmp/audio.wav",
            "--source_path", "/tmp/source.png",
            "--output_path", "/tmp/output.mp4"
        ]
        
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Upload và trả về URL
        output_url = upload_to_s3("/tmp/output.mp4")
        return {"video_url": output_url}
    
    except Exception as e:
        logger.error(f"Error processing job: {e}")
        return {"error": str(e)}

# Khởi động RunPod serverless
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
