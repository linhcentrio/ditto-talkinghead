# minio_config.py
"""
MinIO Configuration và Helper Functions
Tách riêng config để dễ quản lý và bảo mật
"""

import os
from minio import Minio
from urllib.parse import quote
import uuid
from datetime import datetime

# ==== Cấu hình MinIO ====
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "108.181.198.160:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "aiclip-dfl") 
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "aiclipdfl")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

def get_minio_client():
    """Tạo và trả về MinIO client"""
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        return client
    except Exception as e:
        print(f"Error creating MinIO client: {e}")
        raise

def generate_unique_filename(original_filename="output.mp4"):
    """Tạo tên file unique để tránh conflict"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    file_extension = os.path.splitext(original_filename)[1]
    return f"talking_head_{timestamp}_{unique_id}{file_extension}"

def upload_file_to_minio(file_path, custom_filename=None):
    """
    Upload file lên MinIO và trả về direct URL
    
    Args:
        file_path (str): Đường dẫn file cần upload
        custom_filename (str): Tên file custom (optional)
    
    Returns:
        str: Direct download URL
    """
    try:
        client = get_minio_client()
        
        # Tạo bucket nếu chưa tồn tại
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)
            print(f"Created bucket: {MINIO_BUCKET}")
        
        # Tạo tên file unique
        if custom_filename:
            file_name = custom_filename
        else:
            file_name = generate_unique_filename(os.path.basename(file_path))
        
        # Upload file
        client.fput_object(
            MINIO_BUCKET, 
            file_name, 
            file_path,
            content_type="video/mp4"
        )
        
        # Tạo direct URL
        file_name_encoded = quote(file_name)
        protocol = "https" if MINIO_SECURE else "http"
        direct_url = f"{protocol}://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{file_name_encoded}"
        
        print(f"Successfully uploaded {file_path} to MinIO as {file_name}")
        return direct_url
        
    except Exception as e:
        print(f"Error uploading file to MinIO: {e}")
        raise

def cleanup_local_file(file_path):
    """Xóa file local sau khi upload thành công"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up local file: {file_path}")
    except Exception as e:
        print(f"Warning: Could not cleanup local file {file_path}: {e}")
