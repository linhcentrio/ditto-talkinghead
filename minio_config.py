# minio_config.py
"""
MinIO Configuration và Helper Functions với Error Handling nâng cao
Tối ưu cho production deployment với retry logic và performance optimization
"""

import os
import time
import uuid
import logging
from datetime import datetime
from typing import Optional, Tuple
from urllib.parse import quote

from minio import Minio
from minio.error import S3Error, MinioException

# ==== Cấu hình Logging ====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== Cấu hình MinIO ====
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "108.181.198.160:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "aiclip-dfl") 
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "aiclipdfl")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# ==== Cấu hình Performance & Retry ====
MAX_RETRIES = int(os.getenv("MINIO_MAX_RETRIES", "3"))
RETRY_DELAY_BASE = float(os.getenv("MINIO_RETRY_DELAY", "1.0"))
CONNECTION_TIMEOUT = int(os.getenv("MINIO_CONNECTION_TIMEOUT", "30"))
READ_TIMEOUT = int(os.getenv("MINIO_READ_TIMEOUT", "60"))

class MinIOConfig:
    """MinIO Configuration và Connection Management"""
    
    _client_instance = None
    _connection_validated = False
    
    @classmethod
    def get_client(cls) -> Minio:
        """Singleton pattern cho MinIO client với connection pooling"""
        if cls._client_instance is None:
            try:
                cls._client_instance = Minio(
                    MINIO_ENDPOINT,
                    access_key=MINIO_ACCESS_KEY,
                    secret_key=MINIO_SECRET_KEY,
                    secure=MINIO_SECURE,
                    # Performance optimizations
                    http_client=None  # Sử dụng default urllib3 với connection pooling
                )
                logger.info(f"MinIO client created for endpoint: {MINIO_ENDPOINT}")
            except Exception as e:
                logger.error(f"Failed to create MinIO client: {e}")
                raise MinioException(f"MinIO client creation failed: {e}")
        
        return cls._client_instance
    
    @classmethod
    def validate_connection(cls) -> bool:
        """Validate MinIO connection và bucket existence"""
        if cls._connection_validated:
            return True
            
        try:
            client = cls.get_client()
            
            # Test connection với list_buckets
            client.list_buckets()
            
            # Tạo bucket nếu chưa tồn tại
            if not client.bucket_exists(MINIO_BUCKET):
                client.make_bucket(MINIO_BUCKET)
                logger.info(f"Created bucket: {MINIO_BUCKET}")
            
            cls._connection_validated = True
            logger.info("MinIO connection validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"MinIO connection validation failed: {e}")
            return False

def generate_unique_filename(original_filename: str = "output.mp4") -> str:
    """Tạo tên file unique với timestamp và UUID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    file_extension = os.path.splitext(original_filename)[1] or ".mp4"
    return f"talking_head_{timestamp}_{unique_id}{file_extension}"

def calculate_file_size(file_path: str) -> int:
    """Tính size file cho optimization"""
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0

def retry_with_exponential_backoff(func, max_retries: int = MAX_RETRIES, 
                                 base_delay: float = RETRY_DELAY_BASE):
    """Decorator cho retry logic với exponential backoff"""
    def wrapper(*args, **kwargs):
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except (S3Error, MinioException, ConnectionError) as e:
                last_exception = e
                
                if attempt == max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                    break
                
                # Exponential backoff với jitter
                delay = base_delay * (2 ** attempt) + (time.time() % 1) * 0.1
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                             f"Retrying in {delay:.2f}s...")
                time.sleep(delay)
        
        raise last_exception or Exception(f"Function {func.__name__} failed after retries")
    
    return wrapper

@retry_with_exponential_backoff
def upload_file_to_minio(file_path: str, custom_filename: Optional[str] = None) -> str:
    """
    Upload file lên MinIO với retry logic và performance optimization
    
    Args:
        file_path (str): Đường dẫn file cần upload
        custom_filename (str, optional): Tên file custom
    
    Returns:
        str: Direct download URL
        
    Raises:
        MinioException: Khi upload failed sau retries
        FileNotFoundError: Khi file không tồn tại
    """
    # Validate file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_size = calculate_file_size(file_path)
    if file_size == 0:
        raise ValueError(f"File is empty: {file_path}")
    
    # Validate connection trước khi upload
    if not MinIOConfig.validate_connection():
        raise MinioException("MinIO connection validation failed")
    
    client = MinIOConfig.get_client()
    
    # Tạo tên file unique
    file_name = custom_filename or generate_unique_filename(os.path.basename(file_path))
    
    logger.info(f"Uploading {file_path} ({file_size} bytes) as {file_name}")
    
    try:
        # Upload với metadata
        metadata = {
            "X-Amz-Meta-Upload-Time": datetime.now().isoformat(),
            "X-Amz-Meta-Original-Name": os.path.basename(file_path),
            "X-Amz-Meta-File-Size": str(file_size)
        }
        
        # Upload file với optimized settings
        client.fput_object(
            MINIO_BUCKET, 
            file_name, 
            file_path,
            content_type="video/mp4",
            metadata=metadata,
            # Performance optimization cho large files
            part_size=10*1024*1024 if file_size > 50*1024*1024 else 5*1024*1024
        )
        
        # Tạo direct URL
        file_name_encoded = quote(file_name, safe='')
        protocol = "https" if MINIO_SECURE else "http"
        direct_url = f"{protocol}://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{file_name_encoded}"
        
        logger.info(f"Successfully uploaded {file_path} to MinIO as {file_name}")
        logger.info(f"Direct URL: {direct_url}")
        
        return direct_url
        
    except Exception as e:
        logger.error(f"Error uploading {file_path} to MinIO: {e}")
        raise MinioException(f"Upload failed: {e}")

def cleanup_local_file(file_path: str, force: bool = False) -> bool:
    """
    Xóa file local sau khi upload thành công
    
    Args:
        file_path (str): Đường dẫn file cần xóa
        force (bool): Force delete ngay cả khi có lỗi
    
    Returns:
        bool: True nếu xóa thành công
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up local file: {file_path}")
            return True
        return False
    except Exception as e:
        logger.warning(f"Could not cleanup local file {file_path}: {e}")
        if force:
            try:
                os.system(f"rm -f '{file_path}'")  # Force remove
                return True
            except:
                pass
        return False

def get_minio_stats() -> dict:
    """Lấy thông tin stats từ MinIO"""
    try:
        if not MinIOConfig.validate_connection():
            return {"status": "disconnected"}
            
        client = MinIOConfig.get_client()
        buckets = client.list_buckets()
        
        return {
            "status": "connected",
            "endpoint": MINIO_ENDPOINT,
            "bucket": MINIO_BUCKET,
            "total_buckets": len(buckets),
            "secure": MINIO_SECURE
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Health check function
def health_check() -> Tuple[bool, str]:
    """Health check cho MinIO connection"""
    try:
        stats = get_minio_stats()
        if stats["status"] == "connected":
            return True, "MinIO connection healthy"
        else:
            return False, f"MinIO connection failed: {stats.get('error', 'Unknown error')}"
    except Exception as e:
        return False, f"MinIO health check failed: {e}"
