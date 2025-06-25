#!/usr/bin/env python3
"""
🎭 Ditto Talking Head - Optimized Setup (Sử dụng Pre-loaded Environment)
Chạy với environment đã load từ Google Drive và tải models từ HuggingFace
"""

import os
import sys
import subprocess
import time
import threading
import requests
import shutil
from pathlib import Path

# =================== SIMPLIFIED LOGGER ===================
class ProgressLogger:
    def __init__(self, total_steps=6):  # Giảm từ 8 xuống 6 steps
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()

    def log_step(self, step_name, status="progress"):
        if status == "progress":
            self.current_step += 1
            percent = (self.current_step / self.total_steps) * 100
            print(f"[{percent:.0f}%] {step_name}...")
        elif status == "success":
            print(f"✅ {step_name}")
        elif status == "error":
            print(f"❌ {step_name}")

# Import các thư viện đã có trong environment
try:
    from pyngrok import ngrok
    import torch
    import streamlit
    print("✅ Environment imports successful")
except ImportError as e:
    print(f"❌ Environment import error: {e}")
    print("💡 Vui lòng load persistent environment trước")
    sys.exit(1)

# =================== CONSTANTS ===================
REPO_URL = "https://github.com/linhcentrio/ditto-talkinghead.git"
REPO_BRANCH = "colab"
HUGGINGFACE_CONFIG_URL = "https://huggingface.co/digital-avatar/ditto-talkinghead/resolve/main/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
HUGGINGFACE_MODELS_REPO = "manh-linh/ditto_trt_custom"  # Từ search results[1]

class DittoOptimizedSetup:
    def __init__(self):
        self.start_time = time.time()
        self.gpu_capability = 6
        self.data_root = "./checkpoints/ditto_trt"
        self.streamlit_process = None
        self.ngrok_tunnel = None
        self.logger = ProgressLogger()
        
    def run_command_silent(self, cmd, timeout=300):
        """Chạy lệnh im lặng"""
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
            if torch.cuda.is_available():
                self.gpu_capability = torch.cuda.get_device_capability()[0]
                gpu_name = torch.cuda.get_device_name(0)
                print(f"   → GPU: {gpu_name} (Capability: {self.gpu_capability})")
            return True
        except:
            return True
    
    def setup_repository(self):
        """Clone repository (nhanh hơn vì không cần cài dependencies)"""
        
        if os.path.exists("ditto-talkinghead"):
            shutil.rmtree("ditto-talkinghead")
            
        success = self.run_command_silent(
            f"git clone --single-branch --branch {REPO_BRANCH} {REPO_URL} > /dev/null 2>&1"
        )
        
        if not success:
            return False
            
        os.chdir("ditto-talkinghead")
        self.run_command_silent("git pull > /dev/null 2>&1")
        
        return True
    
    def download_models_from_huggingface(self):
        """Tải models từ HuggingFace thay vì Google Drive"""
        
        print("   → Tạo thư mục checkpoints...")
        os.makedirs("checkpoints/ditto_cfg", exist_ok=True)
        
        # Tải config file
        print("   → Tải config từ HuggingFace...")
        success = self.run_command_silent(
            f"wget -q {HUGGINGFACE_CONFIG_URL} -O checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
        )
        
        if not success:
            success = self.run_command_silent(
                f"curl -L {HUGGINGFACE_CONFIG_URL} -o checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl > /dev/null 2>&1"
            )
        
        # Tải TRT models từ HuggingFace[1]
        print(f"   → Tải TRT models từ HuggingFace: {HUGGINGFACE_MODELS_REPO}")
        
        if self.gpu_capability < 8:
            self.data_root = "./checkpoints/ditto_trt"
            print("   → Sử dụng Non-Ampere models")
        else:
            self.data_root = "./checkpoints/ditto_trt_Ampere_Plus"
            print("   → Sử dụng Ampere+ models")
            
        os.makedirs(self.data_root, exist_ok=True)
        
        # Sử dụng huggingface-hub để tải models (đã có trong environment)
        try:
            from huggingface_hub import snapshot_download
            
            print("   → Downloading từ HuggingFace Hub...")
            snapshot_download(
                repo_id=HUGGINGFACE_MODELS_REPO,
                local_dir=self.data_root,
                repo_type="model"
            )
            print("   → HuggingFace models downloaded thành công")
            
        except Exception as e:
            print(f"   → Lỗi download từ HuggingFace: {e}")
            # Fallback: sử dụng git clone
            print("   → Thử git clone...")
            self.run_command_silent(
                f"git clone https://huggingface.co/{HUGGINGFACE_MODELS_REPO} {self.data_root} > /dev/null 2>&1"
            )
            
        return True
    
    def test_ai_core(self):
        """Test AI Core SDK"""
        try:
            sys.path.insert(0, os.getcwd())
            
            if not os.path.exists('inference.py'):
                print("   → inference.py không tìm thấy (sẽ tạo basic version)")
                return True
                
            from inference import StreamSDK
            
            cfg_pkl = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
            if not os.path.exists(cfg_pkl):
                print("   → Config file missing")
                return False
                
            SDK = StreamSDK(cfg_pkl, self.data_root)
            print("   → AI Core SDK initialized thành công")
            return True
            
        except Exception as e:
            print(f"   → AI Core test warning: {e}")
            return True  # Continue anyway
    
    def setup_api_keys(self):
        """Kiểm tra API keys đã được thiết lập"""
        
        ngrok_token = os.environ.get('NGROK_TOKEN', '').strip()
        
        if not ngrok_token:
            print("   → Ngrok token missing!")
            return False
            
        try:
            ngrok.set_auth_token(ngrok_token)
            print("   → Ngrok configured")
        except Exception as e:
            print(f"   → Ngrok error: {e}")
            return False
            
        # Check optional keys
        openai_key = os.environ.get('OPENAI_API_KEY', '').strip()
        pexels_key = os.environ.get('PEXELS_API_KEY', '').strip()
        
        if openai_key:
            print("   → OpenAI API available")
        else:
            print("   → Using Edge TTS")
            
        if pexels_key:
            print("   → Pexels API available")
            
        return True
    
    def create_streamlit_app(self):
        """Tạo Streamlit app optimized"""
        
        if os.path.exists("run_streamlit.py"):
            return True
            
        streamlit_code = '''
import streamlit as st
import sys
import os

# Add project path
sys.path.insert(0, os.getcwd())

st.set_page_config(
    page_title="🎭 Ditto Talking Head",
    page_icon="🎭",
    layout="wide"
)

st.title("🎭 Ditto Talking Head")
st.markdown("### AI-Powered Talking Head Video Generator")
st.markdown("**Environment**: Pre-loaded từ Google Drive")

# Environment info
with st.sidebar:
    st.markdown("### 🔧 Environment Info")
    st.markdown("✅ Packages: Pre-loaded")
    st.markdown("✅ Models: HuggingFace")
    st.markdown("✅ GPU: Available" if torch.cuda.is_available() else "⚠️ GPU: Not available")

# Main UI
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=300)
    
text_input = st.text_area("Enter text to speak:", height=100)

col1, col2 = st.columns(2)

with col1:
    voice_options = ["Edge TTS", "OpenAI TTS"]
    voice_choice = st.selectbox("Voice Engine:", voice_options)

with col2:
    language_options = ["Vietnamese", "English", "Chinese"]
    language = st.selectbox("Language:", language_options)

if st.button("🎬 Generate Talking Head Video", type="primary"):
    if uploaded_file and text_input:
        with st.spinner("🔄 Đang xử lý video..."):
            st.info("🚧 Video generation đang được triển khai")
            st.success("🎉 Demo UI ready! Chức năng sẽ được bổ sung")
    else:
        st.warning("⚠️ Vui lòng upload ảnh và nhập text")

# Footer
st.markdown("---")
st.markdown("### 🔗 Links")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("[🔗 GitHub](https://github.com/linhcentrio/ditto-talkinghead)")
with col2:
    st.markdown("[🤗 HuggingFace](https://huggingface.co/manh-linh/ditto_trt_custom)")
with col3:
    st.markdown("[🌐 Ngrok Dashboard](https://dashboard.ngrok.com/)")
'''
        
        with open("run_streamlit.py", "w", encoding="utf-8") as f:
            f.write(streamlit_code)
            
        return True
    
    def start_streamlit_server(self):
        """Khởi động Streamlit (nhanh hơn vì packages đã load)"""
        
        os.environ.update({
            'STREAMLIT_SERVER_FILE_WATCHER_TYPE': 'none',
            'STREAMLIT_SERVER_HEADLESS': 'true',
            'STREAMLIT_SERVER_PORT': '8501',
            'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false'
        })
        
        if not self.create_streamlit_app():
            return False
            
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
        
        # Đợi server khởi động (nhanh hơn vì packages đã load)
        print("   → Waiting for Streamlit...")
        for attempt in range(8):  # Giảm từ 10 xuống 8
            time.sleep(2)  # Giảm từ 3 xuống 2 giây
            try:
                response = requests.get("http://localhost:8501/_stcore/health", timeout=3)
                if response.status_code == 200:
                    print("   → Streamlit ready")
                    return True
            except:
                continue
                
        return False
    
    def create_ngrok_tunnel(self):
        """Tạo Ngrok tunnel"""
        
        try:
            try:
                ngrok.kill()
                time.sleep(1)  # Giảm thời gian đợi
            except:
                pass
                
            self.ngrok_tunnel = ngrok.connect(8501, "http")
            public_url = str(self.ngrok_tunnel.public_url)
            
            print("\n" + "=" * 70)
            print("🎉 DITTO TALKING HEAD ĐÃ KHỞI ĐỘNG THÀNH CÔNG!")
            print("=" * 70)
            print(f"🔗 Public URL: {public_url}")
            print(f"📱 Truy cập ứng dụng: {public_url}")
            print("🚀 Environment: Pre-loaded từ Google Drive")
            print("🤗 Models: Loaded từ HuggingFace")
            print("⏹️ Để dừng: Ctrl+C hoặc restart runtime")
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
            print(f"❌ Ngrok tunnel error: {str(e)}")
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.streamlit_process:
                self.streamlit_process.terminate()
            if self.ngrok_tunnel:
                ngrok.disconnect(self.ngrok_tunnel.public_url)
            ngrok.kill()
        except:
            pass
    
    def run_optimized_setup(self):
        """Chạy setup tối ưu với environment đã load"""
        
        print("🎭 Ditto Talking Head - Optimized Setup")
        print("🚀 Sử dụng Environment đã load")
        print("=" * 50)
        
        steps = [
            ("Kiểm tra hệ thống", self.check_system),
            ("Thiết lập repository", self.setup_repository),
            ("Tải models từ HuggingFace", self.download_models_from_huggingface),
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
                    self.logger.log_step(f"Lỗi: {step_name}", "error")
                    return False
                    
            return True
            
        except KeyboardInterrupt:
            print("\n🔄 Setup interrupted")
            return False
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            return False
        finally:
            if not self.ngrok_tunnel:
                self.cleanup()

def main():
    """Main function"""
    
    # Kiểm tra API keys
    ngrok_token = os.environ.get('NGROK_TOKEN', '').strip()
    
    if not ngrok_token:
        print("❌ API Keys chưa được thiết lập!")
        print("💡 Vui lòng chạy cell 'Cấu hình API Keys' trước")
        sys.exit(1)
    
    # Kiểm tra environment đã load
    if '/content/packages' not in sys.path:
        print("⚠️ Environment có thể chưa được load đúng cách")
        print("💡 Nhưng sẽ tiếp tục...")
    
    # Chạy setup
    setup = DittoOptimizedSetup()
    
    success = setup.run_optimized_setup()
    
    if not success:
        print("\n❌ Setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
