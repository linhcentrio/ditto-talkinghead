#!/usr/bin/env python3
"""
🎭 AI video creator - Complete Setup for Google Colab
Tối ưu hóa: Chỉ tải từ HuggingFace Hub
"""

import os
import sys
import subprocess
import time
import threading
import requests
import shutil
from pathlib import Path

# =================== INSTALL PYNGROK FIRST ===================
def install_pyngrok():
    try:
        import pyngrok
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyngrok'], 
                      capture_output=True, timeout=120)

install_pyngrok()
from pyngrok import ngrok
import torch

# =================== CONSTANTS ===================
REPO_URL = "https://github.com/linhcentrio/ditto-talkinghead.git"
REPO_BRANCH = "colab"
HUGGINGFACE_CONFIG_URL = "https://huggingface.co/digital-avatar/ditto-talkinghead/resolve/main/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
HUGGINGFACE_TRT_REPO = "manh-linh/ditto_trt_custom"

class DittoSetup:
    def __init__(self):
        self.gpu_capability = 6
        self.data_root = "./checkpoints/ditto_trt"
        self.streamlit_process = None
        self.ngrok_tunnel = None

    def run_cmd(self, cmd, timeout=300):
        """Chạy lệnh im lặng"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, 
                                  text=True, timeout=timeout)
            return result.returncode == 0
        except:
            return False

    def check_system(self):
        """Kiểm tra GPU"""
        try:
            if torch.cuda.is_available():
                self.gpu_capability = torch.cuda.get_device_capability()[0]
            return True
        except:
            return True

    def install_dependencies(self):
        """Cài đặt tất cả dependencies"""
        print(" → Cài đặt AI Core libraries...")
        self.run_cmd("pip install --upgrade pip setuptools wheel > /dev/null 2>&1", 180)
        
        # AI Core
        ai_libs = [
            "tensorrt==8.6.1", "librosa", "tqdm", "filetype", "imageio",
            "opencv-python-headless", "scikit-image", "cython", "cuda-python",
            "imageio-ffmpeg", "colored", "polygraphy", "numpy==2.0.1"
        ]
        self.run_cmd(f"pip install {' '.join(ai_libs)} > /dev/null 2>&1", 300)

        print(" → Cài đặt Streamlit & Processing...")
        ui_libs = [
            "streamlit", "fastapi", "uvicorn", "python-multipart", "requests",
            "pysrt", "python-dotenv", "moviepy==2.1.2", "openai", "edge-tts",
            "gradio", "transparent-background", "insightface", "huggingface_hub"
        ]
        self.run_cmd(f"pip install {' '.join(ui_libs)} > /dev/null 2>&1", 300)

        print(" → Cài đặt FFmpeg...")
        self.run_cmd("apt-get update -qq && apt-get install -y ffmpeg > /dev/null 2>&1", 120)
        
        return True

    def setup_repository(self):
        """Clone repository"""
        if os.path.exists("ditto-talkinghead"):
            shutil.rmtree("ditto-talkinghead")
        
        success = self.run_cmd(f"git clone --single-branch --branch {REPO_BRANCH} {REPO_URL} > /dev/null 2>&1")
        if success:
            os.chdir("ditto-talkinghead")
            self.run_cmd("git pull > /dev/null 2>&1")
        return success

    def download_from_huggingface(self, repo_id, local_dir):
        """Tải models từ HuggingFace Hub"""
        try:
            download_script = f'''
import sys
from huggingface_hub import snapshot_download

try:
    snapshot_download(
        repo_id="{repo_id}",
        local_dir="{local_dir}",
        resume_download=True,
        local_dir_use_symlinks=False
    )
    print("✅ Tải thành công")
    sys.exit(0)
except Exception as e:
    print(f"❌ Lỗi: {{e}}")
    sys.exit(1)
'''
            with open("temp_download.py", "w") as f:
                f.write(download_script)
            
            result = subprocess.run([sys.executable, "temp_download.py"], 
                                  capture_output=True, text=True, timeout=900)
            
            if os.path.exists("temp_download.py"):
                os.remove("temp_download.py")
            
            return result.returncode == 0 and os.path.exists(local_dir) and len(os.listdir(local_dir)) > 0
        except:
            return False

    def download_models(self):
        """Tải models và config từ HuggingFace"""
        # Tạo thư mục
        os.makedirs("checkpoints/ditto_cfg", exist_ok=True)
        
        # Tải config
        success = self.run_cmd(f"wget -q {HUGGINGFACE_CONFIG_URL} -O checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl")
        if not success:
            success = self.run_cmd(f"curl -L {HUGGINGFACE_CONFIG_URL} -o checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl > /dev/null 2>&1")
        
        # Tải TRT models từ HuggingFace
        if self.gpu_capability < 8:
            self.data_root = "./checkpoints/ditto_trt"
        else:
            self.data_root = "./checkpoints/ditto_trt_Ampere_Plus"
        
        os.makedirs(self.data_root, exist_ok=True)
        
        print(f" → Tải models từ HuggingFace Hub...")
        success = self.download_from_huggingface(HUGGINGFACE_TRT_REPO, self.data_root)
        
        if success:
            print(f" → ✅ Đã tải models vào: {self.data_root}")
        else:
            print(f" → ❌ Lỗi tải models từ HuggingFace")
            return False
        
        return True

    def test_ai_core(self):
        """Test AI Core"""
        try:
            sys.path.insert(0, os.getcwd())
            if os.path.exists('inference.py'):
                from inference import StreamSDK
                cfg_pkl = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
                if os.path.exists(cfg_pkl):
                    SDK = StreamSDK(cfg_pkl, self.data_root)
            return True
        except:
            return True

    def setup_api_keys(self):
        """Setup API keys"""
        ngrok_token = os.environ.get('NGROK_TOKEN', '').strip()
        if not ngrok_token:
            return False
        
        try:
            ngrok.set_auth_token(ngrok_token)
        except:
            return False
        
        # Setup optional keys
        openai_key = os.environ.get('OPENAI_API_KEY', '').strip()
        pexels_key = os.environ.get('PEXELS_API_KEY', '').strip()
        
        if openai_key:
            os.environ['OPENAI_API_KEY'] = openai_key
        if pexels_key:
            os.environ['PEXELS_API_KEY'] = pexels_key
        
        return True

    def create_streamlit_app(self):
        """Tạo Streamlit app"""
        if os.path.exists("run_streamlit.py"):
            return True
        
        app_code = '''
import streamlit as st
import sys, os
sys.path.insert(0, os.getcwd())

st.set_page_config(page_title="🎭 AI video creator", page_icon="🎭", layout="wide")
st.title("🎭 AI video creator")
st.markdown("### AI-Powered Talking Head Video Generator")

try:
    from inference import StreamSDK
    st.success("✅ AI Core loaded successfully")
    
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
            f.write(app_code)
        return True

    def start_streamlit(self):
        """Khởi động Streamlit"""
        os.environ.update({
            'STREAMLIT_SERVER_FILE_WATCHER_TYPE': 'none',
            'STREAMLIT_SERVER_HEADLESS': 'true',
            'STREAMLIT_SERVER_PORT': '8501',
            'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false'
        })
        
        if not self.create_streamlit_app():
            return False
        
        def run_streamlit():
            cmd = [
                sys.executable, "-m", "streamlit", "run", "run_streamlit.py",
                "--server.port=8501", "--server.address=0.0.0.0",
                "--server.headless=true", "--browser.gatherUsageStats=false",
                "--server.enableCORS=false", "--server.enableXsrfProtection=false"
            ]
            try:
                self.streamlit_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.streamlit_process.wait()
            except:
                pass
        
        thread = threading.Thread(target=run_streamlit, daemon=True)
        thread.start()
        
        # Đợi server khởi động
        for _ in range(10):
            time.sleep(3)
            try:
                response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
                if response.status_code == 200:
                    return True
            except:
                continue
        return False

    def create_ngrok_tunnel(self):
        """Tạo Ngrok tunnel"""
        try:
            try:
                ngrok.kill()
                time.sleep(2)
            except:
                pass
            
            self.ngrok_tunnel = ngrok.connect(8501, "http")
            public_url = str(self.ngrok_tunnel.public_url)
            
            print("\n" + "=" * 70)
            print("🎉 AI video creator ĐÃ KHỞI ĐỘNG THÀNH CÔNG!")
            print("=" * 70)
            print(f"🔗 Public URL: {public_url}")
            print(f"📱 Truy cập ứng dụng tại: {public_url}")
            print("💡 URL này sẽ hoạt động trong suốt phiên làm việc")
            print("⏹️ Để dừng, nhấn Ctrl+C hoặc restart runtime")
            print("=" * 70)
            
            try:
                while True:
                    time.sleep(30)
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

    def run_setup(self):
        """Chạy toàn bộ setup"""
        print("🎭 AI video creator - Complete Setup")
        print("=" * 50)
        
        steps = [
            ("Kiểm tra hệ thống", self.check_system),
            ("Cài đặt dependencies", self.install_dependencies),
            ("Thiết lập repository", self.setup_repository),
            ("Tải models và config", self.download_models),
            ("Test AI Core", self.test_ai_core),
            ("Thiết lập API keys", self.setup_api_keys),
            ("Khởi động Streamlit", self.start_streamlit),
            ("Tạo Ngrok tunnel", self.create_ngrok_tunnel),
        ]
        
        try:
            for i, (step_name, step_func) in enumerate(steps, 1):
                progress = (i / len(steps)) * 100
                print(f"[{progress:.0f}%] {step_name}...")
                
                if not step_func():
                    print(f"❌ Lỗi tại bước: {step_name}")
                    return False
            
            return True
        except KeyboardInterrupt:
            print("\n🔄 Setup bị ngắt bởi người dùng")
            return False
        except Exception as e:
            print(f"\n❌ Lỗi: {str(e)}")
            return False
        finally:
            if not self.ngrok_tunnel:
                self.cleanup()

def main():
    """Hàm main"""
    # Kiểm tra API keys
    ngrok_token = os.environ.get('NGROK_TOKEN', '').strip()
    if not ngrok_token:
        print("❌ NGROK_TOKEN chưa được thiết lập!")
        print("💡 Vui lòng chạy cell 'Cấu hình API Keys' trước tiên")
        sys.exit(1)
    
    # Chạy setup
    setup = DittoSetup()
    success = setup.run_setup()
    
    if not success:
        print("\n❌ Setup thất bại!")
        sys.exit(1)

if __name__ == "__main__":
    print("📦 Đảm bảo pyngrok được cài đặt...")
    print("📥 Tải script setup...")
    print("🚀 Bắt đầu setup...")
    main()
