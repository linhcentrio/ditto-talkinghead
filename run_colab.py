#!/usr/bin/env python3
"""
🎭 AI video creator - Complete One-Click Setup for Google Colab
Tự động cài đặt và khởi chạy toàn bộ ứng dụng từ đầu đến cuối
"""

import os
import sys
import subprocess
import time
import threading
import json
import requests
import shutil
from pathlib import Path

# =================== SIMPLIFIED LOGGER ===================
class ProgressLogger:
    def __init__(self, total_steps=8):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()

    def log_step(self, step_name, status="progress"):
        if status == "progress":
            self.current_step += 1
            percent = (self.current_step / self.total_steps) * 100
            elapsed = time.time() - self.start_time
            print(f"[{percent:.0f}%] {step_name}...")
        elif status == "success":
            print(f"✅ {step_name}")
        elif status == "error":
            print(f"❌ {step_name}")
        elif status == "info":
            print(f"ℹ️ {step_name}")

# =================== INSTALL DEPENDENCIES FIRST ===================
def install_critical_packages_silent():
    """Cài đặt pyngrok trước khi import"""
    try:
        import pyngrok
    except ImportError:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'pyngrok'],
            capture_output=True, text=True, timeout=120
        )
    return True

# Cài đặt pyngrok trước
install_critical_packages_silent()

# Import sau khi đã cài đặt
try:
    from pyngrok import ngrok
    import torch  # Sử dụng torch có sẵn trong Colab
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# =================== CONSTANTS ===================
REPO_URL = "https://github.com/linhcentrio/ditto-talkinghead.git"
REPO_BRANCH = "colab"
HUGGINGFACE_CONFIG_URL = "https://huggingface.co/digital-avatar/ditto-talkinghead/resolve/main/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
GDRIVE_TRT_MODELS = "1-1qnqy0D9ICgRh8iNY_22j9ieNRC0-zf"

class DittoSimpleSetup:
    def __init__(self):
        self.start_time = time.time()
        self.gpu_capability = 6
        self.data_root = "./checkpoints/ditto_trt"
        self.streamlit_process = None
        self.ngrok_tunnel = None
        self.logger = ProgressLogger()
        
    def run_command_silent(self, cmd, timeout=300):
        """Chạy lệnh im lặng, chỉ trả về success/failure"""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, 
                text=True, timeout=timeout
            )
            return result.returncode == 0
        except:
            return False
    
    def check_system(self):
        """Kiểm tra hệ thống và GPU"""
        try:
            # Kiểm tra GPU capability
            if torch.cuda.is_available():
                self.gpu_capability = torch.cuda.get_device_capability()[0]
            return True
        except:
            return True  # Continue anyway
    
    def install_all_dependencies(self):
        """Cài đặt tất cả dependencies theo danh sách cụ thể"""
        
        # === CÀI ĐẶT THỨ VIỆN AI CORE ===
        print("   → Cài đặt AI Core libraries...")
        
        # Upgrade pip, setuptools, wheel
        self.run_command_silent("pip install --upgrade pip setuptools wheel > /dev/null 2>&1", timeout=180)
        
        # AI Core libraries
        ai_core_libs = [
            "tensorrt==8.6.1", "librosa", "tqdm", "filetype", "imageio", 
            "opencv-python-headless", "scikit-image", "cython", "cuda-python", 
            "imageio-ffmpeg", "colored", "polygraphy", "numpy==2.0.1"
        ]
        
        ai_core_cmd = "pip install " + " ".join(ai_core_libs) + " > /dev/null 2>&1"
        self.run_command_silent(ai_core_cmd, timeout=300)
        
        # === CÀI ĐẶT THỨ VIỆN STREAMLIT UI & PROCESSING ===
        print("   → Cài đặt Streamlit UI & Processing...")
        
        # Streamlit UI libraries
        ui_libs = [
            "streamlit", "fastapi", "uvicorn", "python-multipart", "requests"
        ]
        ui_cmd = "pip install " + " ".join(ui_libs) + " > /dev/null 2>&1"
        self.run_command_silent(ui_cmd, timeout=180)
        
        # Processing libraries
        processing_libs = [
            "pysrt", "python-dotenv", "moviepy==2.1.2"
        ]
        processing_cmd = "pip install " + " ".join(processing_libs) + " > /dev/null 2>&1"
        self.run_command_silent(processing_cmd, timeout=180)
        
        # AI/TTS libraries
        ai_tts_libs = [
            "openai", "edge-tts"
        ]
        ai_tts_cmd = "pip install " + " ".join(ai_tts_libs) + " > /dev/null 2>&1"
        self.run_command_silent(ai_tts_cmd, timeout=120)
        
        # Additional processing libraries
        additional_libs = [
            "gradio", "transparent-background", "insightface"
        ]
        additional_cmd = "pip install " + " ".join(additional_libs) + " > /dev/null 2>&1"
        self.run_command_silent(additional_cmd, timeout=180)
        
        # === CÀI ĐẶT NGROK ===
        print("   → Cài đặt Ngrok...")
        self.run_command_silent("pip install pyngrok > /dev/null 2>&1", timeout=60)
        
        # === CÀI ĐẶT FFMPEG ===
        print("   → Cài đặt FFmpeg...")
        self.run_command_silent("apt-get update -qq > /dev/null 2>&1", timeout=120)
        self.run_command_silent("apt-get install -y ffmpeg > /dev/null 2>&1", timeout=120)
        
        # Verify FFmpeg installation
        self.run_command_silent("ffmpeg -version > /dev/null 2>&1")
        
        # === FIX POTENTIAL LIBRARY CONFLICTS ===
        print("   → Fix library conflicts...")
        try:
            self.run_command_silent("apt install -y libcudnn8 > /dev/null 2>&1", timeout=60)
        except:
            pass  # Continue if libcudnn8 installation fails
        
        # Cài đặt gdown cho việc tải models
        self.run_command_silent("pip install gdown > /dev/null 2>&1", timeout=60)
                
        return True
    
    def setup_repository(self):
        """Clone repository và setup môi trường"""
        
        # Remove existing directory
        if os.path.exists("ditto-talkinghead"):
            shutil.rmtree("ditto-talkinghead")
            
        # Clone repository
        success = self.run_command_silent(
            f"git clone --single-branch --branch {REPO_BRANCH} {REPO_URL} > /dev/null 2>&1"
        )
        
        if not success:
            return False
            
        # Change to project directory
        os.chdir("ditto-talkinghead")
        
        # Pull latest changes
        self.run_command_silent("git pull > /dev/null 2>&1")
        
        return True
    
    def download_models(self):
        """Tải models và config"""
        
        # Tạo thư mục checkpoints
        os.makedirs("checkpoints/ditto_cfg", exist_ok=True)
        
        # Tải config file
        success = self.run_command_silent(
            f"wget -q {HUGGINGFACE_CONFIG_URL} -O checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
        )
        
        if not success:
            # Thử với curl
            success = self.run_command_silent(
                f"curl -L {HUGGINGFACE_CONFIG_URL} -o checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl > /dev/null 2>&1"
            )
            
        # Tải TRT models
        if self.gpu_capability < 8:
            self.data_root = "./checkpoints/ditto_trt"
            os.makedirs(self.data_root, exist_ok=True)
            
            # Ưu tiên tải từ Hugging Face trước
            print("   → Thử tải models từ Hugging Face...")
            hf_success = self.download_from_huggingface()
            
            if not hf_success:
                print("   → Hugging Face thất bại, chuyển sang Google Drive...")
                # Fallback về Google Drive
                self.run_command_silent(
                    f"gdown --folder https://drive.google.com/drive/folders/{GDRIVE_TRT_MODELS} -O {self.data_root} > /dev/null 2>&1",
                    timeout=600
                )
        else:
            self.data_root = "./checkpoints/ditto_trt_Ampere_Plus"
            os.makedirs(self.data_root, exist_ok=True)
            
        return True
    
    def download_from_huggingface(self):
        """Tải models từ Hugging Face với huggingface_hub"""
        try:
            # Cài đặt huggingface_hub nếu chưa có
            install_result = self.run_command_silent(
                "pip install huggingface_hub > /dev/null 2>&1", 
                timeout=120
            )
            
            if not install_result:
                return False
            
            # Import huggingface_hub
            from huggingface_hub import hf_hub_download
            
            # Danh sách các model files cần tải
            model_files = [
                "appearance_extractor_fp16.engine",
                "blaze_face_fp16.engine", 
                "decoder_fp16.engine",
                "face_mesh_fp16.engine",
                "hubert_fp32.engine",
                "insightface_det_fp16.engine",
                "landmark106_fp16.engine",
                "landmark203_fp16.engine",
                "lmdm_v0.4_hubert_fp32.engine",
                "motion_extractor_fp32.engine",
                "stitch_network_fp16.engine",
                "warp_network_fp16.engine"
            ]
            
            # Tải từng file
            for model_file in model_files:
                try:
                    downloaded_path = hf_hub_download(
                        repo_id="manh-linh/ditto_trt_custom",
                        filename=model_file,
                        cache_dir=self.data_root,
                        local_dir=self.data_root,
                        local_dir_use_symlinks=False
                    )
                    print(f"   ✓ Đã tải: {model_file}")
                except Exception as e:
                    print(f"   ✗ Lỗi tải {model_file}: {str(e)}")
                    return False
            
            print("   ✅ Tải hoàn tất từ Hugging Face!")
            return True
            
        except ImportError:
            print("   ✗ Không thể import huggingface_hub")
            return False
        except Exception as e:
            print(f"   ✗ Lỗi tải từ Hugging Face: {str(e)}")
            return False
    
    def test_ai_core(self):
        """Test AI Core SDK"""
        try:
            sys.path.insert(0, os.getcwd())
            
            if not os.path.exists('inference.py'):
                return False
                
            from inference import StreamSDK
            
            cfg_pkl = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
            if not os.path.exists(cfg_pkl):
                return False
                
            SDK = StreamSDK(cfg_pkl, self.data_root)
            return True
            
        except:
            return True  # Continue anyway
    
    def setup_api_keys(self):
        """Thiết lập API keys từ environment variables"""
        
        # Lấy keys từ environment variables
        ngrok_token = os.environ.get('NGROK_TOKEN', '').strip()
        openai_key = os.environ.get('OPENAI_API_KEY', '').strip()
        pexels_key = os.environ.get('PEXELS_API_KEY', '').strip()
        
        # Kiểm tra Ngrok token (bắt buộc)
        if not ngrok_token:
            return False
            
        # Thiết lập Ngrok
        try:
            ngrok.set_auth_token(ngrok_token)
        except:
            return False
            
        # Thiết lập optional keys và thông báo
        if openai_key:
            os.environ['OPENAI_API_KEY'] = openai_key
            print("   ✅ OpenAI API Key được thiết lập từ Notebook")
        else:
            print("   ℹ️ OpenAI API Key không có - có thể nhập trong tab Cài đặt")
            
        if pexels_key:
            os.environ['PEXELS_API_KEY'] = pexels_key
            print("   ✅ Pexels API Key được thiết lập từ Notebook")
            
        return True
    
    def create_streamlit_app(self):
        """Tạo Streamlit app file"""
        if os.path.exists("run_streamlit.py"):
            return True
            
        streamlit_code = '''
import streamlit as st
import sys
import os

# Add project path
sys.path.insert(0, os.getcwd())

st.set_page_config(
    page_title="🎭 AI video creator",
    page_icon="🎭",
    layout="wide"
)

st.title("🎭 AI video creator")
st.markdown("### AI-Powered Talking Head Video Generator")

# Check if inference module exists
try:
    from inference import StreamSDK
    st.success("✅ AI Core loaded successfully")
    
    # Basic UI
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        
    text_input = st.text_area("Enter text to speak:", height=100)
    
    if st.button("Generate Talking Head Video"):
        if uploaded_file and text_input:
            st.info("🚧 Video generation feature will be implemented here")
        else:
            st.warning("Please upload an image and enter text")
            
except ImportError as e:
    st.error(f"❌ Error loading AI Core: {e}")
    st.info("Please check the setup and try again")

st.markdown("---")
st.markdown("🔗 **Links:**")
st.markdown("- [GitHub Repository](https://github.com/linhcentrio/ditto-talkinghead)")
st.markdown("- [Ngrok Dashboard](https://dashboard.ngrok.com/)")
'''
        
        with open("run_streamlit.py", "w", encoding="utf-8") as f:
            f.write(streamlit_code)
            
        return True
    
    def start_streamlit_server(self):
        """Khởi động Streamlit server"""
        
        # Thiết lập environment variables
        os.environ.update({
            'STREAMLIT_SERVER_FILE_WATCHER_TYPE': 'none',
            'STREAMLIT_SERVER_HEADLESS': 'true',
            'STREAMLIT_SERVER_PORT': '8501',
            'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false'
        })
        
        # Tạo streamlit app
        if not self.create_streamlit_app():
            return False
            
        # Khởi chạy Streamlit trong thread riêng
        def run_streamlit():
            streamlit_cmd = [
                sys.executable, "-m", "streamlit", "run", "run_streamlit.py",
                "--server.port=8501", "--server.address=0.0.0.0", 
                "--server.headless=true", "--browser.gatherUsageStats=false",
                "--server.enableCORS=false", "--server.enableXsrfProtection=false"
            ]
            
            try:
                self.streamlit_process = subprocess.Popen(
                    streamlit_cmd, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )
                self.streamlit_process.wait()
            except:
                pass
                
        streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
        streamlit_thread.start()
        
        # Đợi server khởi động
        for attempt in range(10):
            time.sleep(3)
            try:
                response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
                if response.status_code == 200:
                    return True
            except:
                continue
                
        return False
    
    def create_ngrok_tunnel(self):
        """Tạo Ngrok tunnel và hiển thị URL"""
        
        try:
            # Dọn dẹp tunnel cũ
            try:
                ngrok.kill()
                time.sleep(2)
            except:
                pass
                
            # Tạo tunnel mới
            self.ngrok_tunnel = ngrok.connect(8501, "http")
            public_url = str(self.ngrok_tunnel.public_url)
            
            # Hiển thị kết quả cuối cùng
            print("\n" + "=" * 70)
            print("🎉 AI video creator ĐÃ KHỞI ĐỘNG THÀNH CÔNG!")
            print("=" * 70)
            print(f"🔗 Public URL: {public_url}")
            print(f"📱 Truy cập ứng dụng tại: {public_url}")
            print("💡 URL này sẽ hoạt động trong suốt phiên làm việc")
            print("⏹️ Để dừng, nhấn Ctrl+C hoặc restart runtime")
            print("=" * 70)
            
            # Giữ script chạy
            try:
                while True:
                    time.sleep(30)
                    # Health check im lặng
                    try:
                        requests.get(f"{public_url}/_stcore/health", timeout=5)
                    except:
                        pass
                        
            except KeyboardInterrupt:
                print("\n🔄 Đang tắt ứng dụng...")
                self.cleanup()
                
            return True
            
        except Exception as e:
            print(f"❌ Lỗi tạo Ngrok tunnel: {str(e)}")
            return False
    
    def cleanup(self):
        """Dọn dẹp resources"""
        try:
            if self.streamlit_process:
                self.streamlit_process.terminate()
            if self.ngrok_tunnel:
                ngrok.disconnect(self.ngrok_tunnel.public_url)
            ngrok.kill()
        except:
            pass
    
    def run_complete_setup(self):
        """Chạy toàn bộ quá trình setup với progress đơn giản"""
        
        print("🎭 AI video creator - Complete Setup")
        print("=" * 50)
        
        steps = [
            ("Kiểm tra hệ thống", self.check_system),
            ("Cài đặt dependencies", self.install_all_dependencies),
            ("Thiết lập repository", self.setup_repository),
            ("Tải models và config", self.download_models),
            ("Test AI Core", self.test_ai_core),
            ("Thiết lập API keys", self.setup_api_keys),
            ("Khởi động Streamlit", self.start_streamlit_server),
            ("Tạo Ngrok tunnel", self.create_ngrok_tunnel),
        ]
        
        try:
            for step_name, step_func in steps:
                self.logger.log_step(step_name, "progress")
                
                success = step_func()
                
                if not success:
                    self.logger.log_step(f"Lỗi tại bước: {step_name}", "error")
                    return False
                    
            return True
            
        except KeyboardInterrupt:
            print("\n🔄 Setup bị ngắt bởi người dùng")
            return False
        except Exception as e:
            print(f"\n❌ Lỗi: {str(e)}")
            return False
        finally:
            if not self.ngrok_tunnel:  # Only cleanup if not running
                self.cleanup()

def main():
    """Hàm main"""
    
    # Kiểm tra API keys đã được thiết lập chưa
    ngrok_token = os.environ.get('NGROK_TOKEN', '').strip()
    
    if not ngrok_token:
        print("❌ API Keys chưa được thiết lập!")
        print("💡 Vui lòng chạy cell 'Cấu hình API Keys' trước tiên")
        sys.exit(1)
    
    # Khởi tạo và chạy setup
    setup = DittoSimpleSetup()
    
    success = setup.run_complete_setup()
    
    if not success:
        print("\n❌ Setup thất bại!")
        sys.exit(1)

if __name__ == "__main__":
    main()
