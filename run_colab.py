#!/usr/bin/env python3
"""
🎭 Ditto Talking Head - One-Click Setup for Google Colab (Secure Version)
Tự động cài đặt và khởi chạy toàn bộ ứng dụng trong một lần chạy
"""

import os
import sys
import subprocess
import time
import threading
import json
import requests
from pathlib import Path
from pyngrok import ngrok

# =================== CONSTANTS ===================
REPO_URL = "https://github.com/linhcentrio/ditto-talkinghead.git"
REPO_BRANCH = "colab"
HUGGINGFACE_CONFIG_URL = "https://huggingface.co/digital-avatar/ditto-talkinghead/resolve/main/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
GDRIVE_TRT_MODELS = "https://drive.google.com/drive/folders/1-1qnqy0D9ICgRh8iNY_22j9ieNRC0-zf?usp=sharing"

class DittoSetup:
    def __init__(self):
        self.start_time = time.time()
        self.gpu_capability = 6  # Default
        self.data_root = "./checkpoints/ditto_trt"
        
    def log(self, message, prefix="🎭"):
        """In log với timestamp"""
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:6.1f}s] {prefix} {message}")
        
    def run_command(self, cmd, capture=True, timeout=300, shell=True):
        """Chạy lệnh với xử lý lỗi và timeout"""
        try:
            result = subprocess.run(
                cmd,
                shell=shell,
                capture_output=capture,
                text=True,
                timeout=timeout
            )
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, f"Timeout after {timeout}s"
        except Exception as e:
            return False, str(e)
    
    def check_system(self):
        """Kiểm tra hệ thống và GPU"""
        self.log("Kiểm tra hệ thống...")
        
        # Kiểm tra GPU
        success, output = self.run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
        if success:
            self.log(f"GPU: {output.strip()}")
        else:
            self.log("Không phát hiện GPU", "⚠️")
            
        # Kiểm tra PyTorch và CUDA
        try:
            import torch
            self.log(f"PyTorch: {torch.__version__}")
            
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.gpu_capability = torch.cuda.get_device_capability()[0]
                
                self.log(f"CUDA: {torch.version.cuda} | GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                self.log(f"GPU Compute Capability: {self.gpu_capability}")
            else:
                self.log("CUDA không khả dụng", "⚠️")
                
        except ImportError:
            self.log("PyTorch chưa được cài đặt", "❌")
            return False
            
        return True
    
    def install_dependencies(self):
        """Cài đặt tất cả dependencies"""
        self.log("Cài đặt thư viện cần thiết...")
        
        # Cập nhật pip
        self.run_command("pip install --upgrade pip setuptools wheel", timeout=180)
        
        # AI Core libraries
        ai_libs = [
            "tensorrt==8.6.1",
            "librosa", "tqdm", "filetype", "imageio", "opencv-python-headless",
            "scikit-image", "cython", "cuda-python", "imageio-ffmpeg", "colored",
            "polygraphy", "numpy==2.0.1"
        ]
        
        # UI and processing libraries
        ui_libs = [
            "streamlit", "fastapi", "uvicorn", "python-multipart", "requests",
            "pysrt", "python-dotenv", "moviepy==2.1.2",
            "openai", "edge-tts", "gradio", "transparent-background", "insightface"
        ]
        
        # Ngrok
        ngrok_libs = ["pyngrok"]
        
        all_libs = ai_libs + ui_libs + ngrok_libs
        
        for lib in all_libs:
            self.log(f"Cài đặt {lib}...")
            success, output = self.run_command(f"pip install {lib}", timeout=120)
            if not success:
                self.log(f"Lỗi cài đặt {lib}: {output}", "⚠️")
        
        # Cài đặt FFmpeg
        self.log("Cài đặt FFmpeg...")
        self.run_command("apt-get update -qq && apt-get install -y ffmpeg", timeout=180)
        
        # Cài đặt libcudnn8 (optional)
        try:
            self.run_command("apt install -y libcudnn8", timeout=60)
        except:
            self.log("Không thể cài đặt libcudnn8", "⚠️")
            
        self.log("Hoàn thành cài đặt thư viện")
        return True
    
    def setup_repository(self):
        """Clone repository và setup"""
        self.log("Thiết lập repository...")
        
        # Remove existing directory
        if os.path.exists("ditto-talkinghead"):
            self.run_command("rm -rf ditto-talkinghead")
            
        # Clone repository
        success, output = self.run_command(
            f"git clone --single-branch --branch {REPO_BRANCH} {REPO_URL}",
            timeout=120
        )
        
        if not success:
            self.log(f"Lỗi clone repository: {output}", "❌")
            return False
            
        # Change to project directory
        os.chdir("ditto-talkinghead")
        self.log("Repository đã được clone thành công")
        
        # Pull latest changes
        self.run_command("git pull")
        
        return True
    
    def download_models(self):
        """Tải models và config"""
        self.log("Tải models và config...")
        
        # Tạo thư mục checkpoints
        os.makedirs("checkpoints/ditto_cfg", exist_ok=True)
        
        # Tải config file
        self.log("Tải config file...")
        success, output = self.run_command(
            f"wget -q {HUGGINGFACE_CONFIG_URL} -O checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
        )
        
        if not success or not os.path.exists("checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"):
            self.log("Lỗi tải config file", "❌")
            return False
            
        self.log("Config file đã được tải")
        
        # Tải TRT models dựa trên GPU capability
        if self.gpu_capability < 8:
            self.log("Tải Non-Ampere TRT models...")
            # Cài đặt gdown nếu chưa có
            self.run_command("pip install --upgrade --no-cache-dir gdown")
            
            # Tải models từ Google Drive
            success, output = self.run_command(
                f"gdown {GDRIVE_TRT_MODELS} -O ./checkpoints/ditto_trt --folder",
                timeout=600
            )
            
            if success:
                self.data_root = "./checkpoints/ditto_trt"
                self.log("TRT models đã được tải")
            else:
                self.log(f"Lỗi tải TRT models: {output}", "⚠️")
                # Tạo thư mục dummy
                os.makedirs("./checkpoints/ditto_trt", exist_ok=True)
        else:
            self.log("Sử dụng Ampere+ models")
            self.data_root = "./checkpoints/ditto_trt_Ampere_Plus"
            os.makedirs(self.data_root, exist_ok=True)
            
        return True
    
    def test_ai_core(self):
        """Test AI Core SDK"""
        self.log("Kiểm tra AI Core...")
        
        try:
            # Thêm path để import
            sys.path.insert(0, os.getcwd())
            
            # Kiểm tra file inference.py
            if not os.path.exists('inference.py'):
                self.log("Không tìm thấy inference.py", "⚠️")
                return False
                
            # Import và test SDK
            from inference import StreamSDK
            
            cfg_pkl = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
            
            if not os.path.exists(cfg_pkl):
                self.log("Không tìm thấy config file", "❌")
                return False
                
            SDK = StreamSDK(cfg_pkl, self.data_root)
            self.log("AI Core SDK khởi tạo thành công")
            return True
            
        except Exception as e:
            self.log(f"Lỗi test AI Core: {str(e)}", "⚠️")
            return False
    
    def setup_api_keys(self):
        """Thiết lập API keys từ environment variables"""
        self.log("Thiết lập API keys...")
        
        # Lấy keys từ environment variables
        ngrok_token = os.environ.get('NGROK_TOKEN', '').strip()
        openai_key = os.environ.get('OPENAI_API_KEY', '').strip()
        pexels_key = os.environ.get('PEXELS_API_KEY', '').strip()
        
        # Kiểm tra Ngrok token (bắt buộc)
        if not ngrok_token:
            self.log("Ngrok token không được tìm thấy trong environment!", "❌")
            self.log("Vui lòng chạy cell thiết lập API keys trước", "💡")
            return False
            
        # Thiết lập Ngrok
        try:
            ngrok.set_auth_token(ngrok_token)
            self.log("Ngrok token đã được cấu hình")
        except Exception as e:
            self.log(f"Lỗi cấu hình Ngrok: {str(e)}", "❌")
            return False
            
        # Thiết lập OpenAI (tùy chọn)
        if openai_key:
            os.environ['OPENAI_API_KEY'] = openai_key
            self.log("OpenAI API key đã được cấu hình")
        else:
            self.log("OpenAI API key không có (sẽ dùng Edge TTS)", "ℹ️")
            
        # Thiết lập Pexels (tùy chọn)
        if pexels_key:
            os.environ['PEXELS_API_KEY'] = pexels_key
            self.log("Pexels API key đã được cấu hình")
        else:
            self.log("Pexels API key không có (tùy chọn)", "ℹ️")
            
        return True
    
    def start_streamlit_server(self):
        """Khởi động Streamlit server"""
        self.log("Khởi động Streamlit server...")
        
        # Thiết lập environment variables
        os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
        os.environ['STREAMLIT_SERVER_PORT'] = '8501'
        
        # Kiểm tra file run_streamlit.py
        if not os.path.exists("run_streamlit.py"):
            self.log("Không tìm thấy run_streamlit.py", "❌")
            return False
            
        # Khởi chạy Streamlit trong thread riêng
        def run_streamlit():
            streamlit_cmd = [
                sys.executable, "-m", "streamlit", "run", "run_streamlit.py",
                "--server.port=8501",
                "--server.address=0.0.0.0", 
                "--server.headless=true",
                "--browser.gatherUsageStats=false"
            ]
            
            try:
                subprocess.run(streamlit_cmd, check=True)
            except Exception as e:
                self.log(f"Lỗi Streamlit: {str(e)}", "❌")
                
        streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
        streamlit_thread.start()
        
        # Đợi server khởi động
        self.log("Đợi server khởi động...")
        time.sleep(15)
        
        # Kiểm tra server
        for i in range(5):
            try:
                response = requests.get("http://localhost:8501", timeout=5)
                if response.status_code == 200:
                    self.log("Streamlit server đã khởi động thành công")
                    return True
            except:
                self.log(f"Thử lần {i+1}/5: Server chưa sẵn sàng...")
                time.sleep(5)
                
        self.log("Streamlit server không thể khởi động", "❌")
        return False
    
    def create_ngrok_tunnel(self):
        """Tạo Ngrok tunnel"""
        self.log("Tạo Ngrok tunnel...")
        
        try:
            # Tạo tunnel
            public_url = ngrok.connect(8501, "http")
            
            self.log("=" * 60)
            self.log("🎉 NGROK TUNNEL ĐÃ TẠO THÀNH CÔNG!", "✅")
            self.log("=" * 60)
            self.log(f"🔗 Public URL: {public_url}")
            self.log(f"📱 Truy cập ứng dụng tại: {public_url}")
            self.log("💡 URL này là tạm thời và sẽ thay đổi khi restart")
            self.log("⏹️ Để dừng, nhấn Ctrl+C")
            self.log("=" * 60)
            
            # Giữ script chạy
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.log("Đang tắt ứng dụng...")
                ngrok.disconnect(public_url)
                ngrok.kill()
                
            return True
            
        except Exception as e:
            self.log(f"Lỗi tạo Ngrok tunnel: {str(e)}", "❌")
            return False
    
    def run_full_setup(self):
        """Chạy toàn bộ quá trình setup"""
        self.log("🎭 BẮT ĐẦU DITTO TALKING HEAD SETUP")
        self.log("=" * 60)
        
        steps = [
            ("Kiểm tra hệ thống", self.check_system),
            ("Cài đặt dependencies", self.install_dependencies),
            ("Thiết lập repository", self.setup_repository),
            ("Tải models", self.download_models),
            ("Test AI Core", self.test_ai_core),
            ("Thiết lập API keys", self.setup_api_keys),
            ("Khởi động Streamlit", self.start_streamlit_server),
            ("Tạo Ngrok tunnel", self.create_ngrok_tunnel),
        ]
        
        for step_name, step_func in steps:
            self.log(f"📋 {step_name}...")
            
            try:
                if not step_func():
                    self.log(f"❌ Lỗi tại bước: {step_name}")
                    return False
                    
                self.log(f"✅ Hoàn thành: {step_name}")
                
            except Exception as e:
                self.log(f"❌ Exception tại {step_name}: {str(e)}")
                return False
                
        elapsed = time.time() - self.start_time
        self.log(f"🎉 SETUP HOÀN TẤT! Tổng thời gian: {elapsed:.1f}s")
        return True

def main():
    """Hàm main"""
    print("🎭 Ditto Talking Head - One-Click Setup")
    print("=" * 60)
    
    # Kiểm tra API keys đã được thiết lập chưa
    ngrok_token = os.environ.get('NGROK_TOKEN', '').strip()
    
    if not ngrok_token:
        print("❌ API Keys chưa được thiết lập!")
        print("💡 Vui lòng chạy cell 'Thiết lập API Keys' trước tiên")
        print("🔗 Cell đó sẽ hướng dẫn bạn cách lấy và nhập các API keys cần thiết")
        sys.exit(1)
    
    print("✅ API Keys đã được thiết lập, bắt đầu setup...")
    
    # Khởi tạo và chạy setup
    setup = DittoSetup()
    
    success = setup.run_full_setup()
    
    if not success:
        print("\n❌ Setup thất bại! Vui lòng kiểm tra logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
