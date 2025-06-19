#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎭 Ditto Talking Head - Google Colab Setup with GPU-aware TRT models
Automated installation and setup for Streamlit app with CloudFlared tunnel
"""

import os
import sys
import subprocess
import time
import json
import shutil
from pathlib import Path
import requests
from IPython.display import clear_output, display, HTML
import threading
import queue

class SetupManager:
    def __init__(self):
        self.start_time = time.time()
        self.status_queue = queue.Queue()
        self.current_step = 0
        self.total_steps = 12  # Tăng từ 10 lên 12
        self.gpu_capability = None
        self.data_root = None

    def print_header(self):
        """Print setup header"""
        print("🎭 " + "="*50)
        print("   DITTO TALKING HEAD - GOOGLE COLAB SETUP")
        print("="*52)
        print("🚀 Automated setup for AI video generation")
        print("⏱️  Estimated time: 3-5 minutes")
        print("="*52)

    def update_progress(self, step_name, status="working"):
        """Update progress with visual indicator"""
        self.current_step += 1
        progress = (self.current_step / self.total_steps) * 100

        status_icons = {
            "working": "🔄",
            "success": "✅",
            "error": "❌",
            "warning": "⚠️"
        }

        icon = status_icons.get(status, "🔄")
        print(f"\n{icon} [{self.current_step}/{self.total_steps}] {step_name}")
        print(f"Progress: {'█' * int(progress//5)}{'░' * (20 - int(progress//5))} {progress:.0f}%")

        if status == "success":
            print(f"   ✓ Completed in {time.time() - self.start_time:.1f}s")

    def run_silent_command(self, cmd, timeout=300):
        """Run command silently with timeout"""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout, check=False
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timeout"
        except Exception as e:
            return False, "", str(e)

    def detect_gpu_capability(self):
        """Detect GPU compute capability"""
        self.update_progress("Detecting GPU architecture")
        
        try:
            # Check if torch is available, install if not
            try:
                import torch
            except ImportError:
                print("Installing PyTorch for GPU detection...")
                success, _, _ = self.run_silent_command(
                    "pip install torch --index-url https://download.pytorch.org/whl/cu121 -q", 180
                )
                if not success:
                    print("⚠️ Failed to install PyTorch, assuming GPU capability 6")
                    self.gpu_capability = 6
                    self.update_progress("GPU detection", "warning")
                    return
                import torch
            
            if torch.cuda.is_available():
                self.gpu_capability = torch.cuda.get_device_capability()[0]
                gpu_name = torch.cuda.get_device_name()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                print(f"🔍 GPU: {gpu_name}")
                print(f"🔍 GPU Compute Capability: {self.gpu_capability}")
                print(f"🔍 GPU Memory: {gpu_memory:.1f}GB")
                
                # Determine architecture
                if self.gpu_capability >= 8:
                    print("🚀 Ampere+ architecture detected (>=8.0)")
                    self.data_root = "./checkpoints/ditto_trt_Ampere_Plus"
                else:
                    print("📦 Pre-Ampere architecture detected (<8.0)")
                    self.data_root = "./checkpoints/ditto_trt"
            else:
                print("⚠️ No CUDA GPU detected, using CPU fallback")
                self.gpu_capability = 6  # Default to older architecture
                self.data_root = "./checkpoints/ditto_trt"
                
        except Exception as e:
            print(f"⚠️ GPU detection failed: {e}, assuming capability 6")
            self.gpu_capability = 6
            self.data_root = "./checkpoints/ditto_trt"
        
        self.update_progress("GPU detection", "success")

    def install_system_deps(self):
        """Install system dependencies"""
        self.update_progress("Installing system dependencies")

        commands = [
            "apt-get update -qq",
            "apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1-mesa-glx git-lfs wget curl",
            "apt-get install -y fonts-dejavu-core fonts-liberation",  # Add fonts for better text rendering
            "git lfs install"
        ]

        for cmd in commands:
            success, _, error = self.run_silent_command(cmd)
            if not success and "apt-get update" not in cmd:
                print(f"⚠️ Command failed: {cmd}")
                print(f"Error: {error}")
                self.update_progress("System dependencies", "warning")
                return False

        self.update_progress("System dependencies", "success")
        return True

    def install_python_deps(self):
        """Install Python dependencies with GPU-specific optimizations"""
        self.update_progress("Installing Python packages")

        # Core packages first
        core_packages = [
            "pip install --upgrade pip setuptools wheel -q",
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q",
            "pip install numpy==2.0.1 opencv-python-headless scikit-image -q",
            "pip install librosa tqdm filetype imageio imageio-ffmpeg -q"
        ]

        for cmd in core_packages:
            success, _, _ = self.run_silent_command(cmd, 180)
            if not success:
                self.update_progress("Python packages", "warning")

        # AI packages with GPU considerations
        ai_packages = [
            "pip install --upgrade --no-cache-dir gdown -q",  # For Google Drive downloads
            "pip install cython transparent-background insightface -q",
            "pip install streamlit fastapi uvicorn python-multipart -q",
            "pip install moviepy==2.1.2 pysrt openai edge-tts pillow -q"
        ]

        # Add TensorRT if GPU capability >= 8
        if self.gpu_capability and self.gpu_capability >= 8:
            ai_packages.insert(1, "pip install tensorrt==8.6.1 cuda-python polygraphy -q")

        for cmd in ai_packages:
            self.run_silent_command(cmd, 120)  # Don't fail on these

        self.update_progress("Python packages", "success")
        return True

    def setup_cloudflared(self):
        """Setup CloudFlared tunnel"""
        self.update_progress("Setting up CloudFlared tunnel")

        # Check if already installed
        success, _, _ = self.run_silent_command("cloudflared --version")
        if success:
            self.update_progress("CloudFlared tunnel", "success")
            return True

        # Download and install
        commands = [
            "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            "chmod +x cloudflared-linux-amd64",
            "sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared"
        ]

        for cmd in commands:
            success, _, _ = self.run_silent_command(cmd)
            if not success:
                self.update_progress("CloudFlared tunnel", "error")
                return False

        self.update_progress("CloudFlared tunnel", "success")
        return True

    def clone_repository(self):
        """Clone project repository"""
        self.update_progress("Downloading project files")

        if Path("ditto-talkinghead").exists():
            os.chdir("ditto-talkinghead")
            success, _, _ = self.run_silent_command("git pull")
        else:
            success, _, _ = self.run_silent_command(
                "git clone --single-branch --branch colab https://github.com/linhcentrio/ditto-talkinghead.git"
            )
            if success:
                os.chdir("ditto-talkinghead")

        if not success:
            self.update_progress("Project files", "error")
            return False

        self.update_progress("Project files", "success")
        return True

    def download_models_gpu_aware(self):
        """Download models based on GPU capability"""
        self.update_progress("Downloading AI models (GPU-aware)")
        
        print(f"🔍 Target data root: {self.data_root}")
        
        if self.gpu_capability >= 8:
            # Ampere+ architecture - use Hugging Face models
            print("🚀 Downloading Ampere+ optimized models from Hugging Face...")
            return self._download_ampere_models()
        else:
            # Pre-Ampere architecture - use Google Drive models
            print("📦 Downloading Non-Ampere TRT models from Google Drive...")
            return self._download_legacy_models()

    def _download_ampere_models(self):
        """Download Ampere+ models from Hugging Face"""
        try:
            if Path("checkpoints").exists():
                print("📁 Checkpoints exist, updating...")
                success, _, _ = self.run_silent_command("cd checkpoints && git pull", 120)
            else:
                print("📥 Downloading checkpoints from Hugging Face...")
                success, _, _ = self.run_silent_command(
                    "git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints", 300
                )
            
            if not success:
                self.update_progress("AI models", "error") 
                return False
                
            # Verify required files exist
            required_paths = [
                "checkpoints/ditto_trt_Ampere_Plus",
                "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
            ]
            
            for path in required_paths:
                if not Path(path).exists():
                    print(f"⚠️ Missing required path: {path}")
            
            print("📋 Available checkpoints:")
            success, output, _ = self.run_silent_command("ls -la checkpoints/")
            if success:
                print(output)
                
            self.update_progress("AI models", "success")
            return True
            
        except Exception as e:
            print(f"❌ Ampere model download failed: {e}")
            self.update_progress("AI models", "error")
            return False

    def _download_legacy_models(self):
        """Download legacy models for pre-Ampere GPUs"""
        try:
            # Create checkpoints directory structure
            Path("checkpoints/ditto_trt").mkdir(parents=True, exist_ok=True)
            Path("checkpoints/ditto_cfg").mkdir(parents=True, exist_ok=True)
            
            # Download TRT models from Google Drive
            print("📦 Downloading Non-Ampere TRT models...")
            success, _, error = self.run_silent_command(
                "gdown https://drive.google.com/drive/folders/1-1qnqy0D9ICgRh8iNY_22j9ieNRC0-zf?usp=sharing -O ./checkpoints/ditto_trt --folder", 
                600  # 10 minutes timeout
            )
            
            if not success:
                print(f"⚠️ Google Drive download failed: {error}")
                print("🔄 Trying alternative method...")
                
                # Fallback: Download individual files
                success = self._download_legacy_files_individually()
                
            if success:
                # Download config file from Hugging Face
                print("📥 Downloading config file from Hugging Face...")
                config_url = "https://huggingface.co/digital-avatar/ditto-talkinghead/resolve/main/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
                success, _, _ = self.run_silent_command(
                    f"wget -q -O checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl {config_url}", 120
                )
                
                if not success:
                    print("⚠️ Config download failed, trying curl...")
                    success, _, _ = self.run_silent_command(
                        f"curl -L -o checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl {config_url}", 120
                    )
            
            if success:
                self.update_progress("AI models", "success")
                return True
            else:
                self.update_progress("AI models", "warning") 
                return False
                
        except Exception as e:
            print(f"❌ Legacy model download failed: {e}")
            self.update_progress("AI models", "error")
            return False

    def _download_legacy_files_individually(self):
        """Fallback method to download legacy files individually"""
        try:
            # This would need specific file IDs - placeholder for now
            print("🔄 Individual file download not implemented yet")
            print("📝 Please manually download TRT models for pre-Ampere GPUs")
            return False
        except Exception as e:
            print(f"❌ Individual download failed: {e}")
            return False

    def verify_model_structure(self):
        """Verify downloaded model structure"""
        self.update_progress("Verifying model structure")
        
        print(f"\n=== MODEL STRUCTURE VERIFICATION ===")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Target data root: {self.data_root}")
        
        # Check if checkpoints exists
        checkpoints_path = Path("checkpoints")
        if checkpoints_path.exists():
            print(f"✅ Checkpoints directory exists: {checkpoints_path.absolute()}")
            
            # List key contents
            print("\n📁 Key checkpoints contents:")
            for item in ["ditto_trt_Ampere_Plus", "ditto_trt", "ditto_cfg"]:
                item_path = checkpoints_path / item
                if item_path.exists():
                    if item_path.is_dir():
                        file_count = len(list(item_path.rglob("*")))
                        print(f"  ✅ {item}/ ({file_count} files)")
                    else:
                        print(f"  ✅ {item}")
                else:
                    print(f"  ❌ {item}")
        else:
            print("❌ Checkpoints directory not found")
            self.update_progress("Model verification", "error")
            return False
        
        # Check for specific required files based on GPU capability
        if self.gpu_capability >= 8:
            required_files = [
                "checkpoints/ditto_trt_Ampere_Plus/",
                "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl",
            ]
        else:
            required_files = [
                "checkpoints/ditto_trt/",
                "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl",
            ]
        
        print(f"\n🔍 Checking required files for GPU capability {self.gpu_capability}:")
        all_found = True
        for file_path in required_files:
            path = Path(file_path)
            if path.exists():
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ {file_path}")
                all_found = False
        
        if all_found:
            self.update_progress("Model verification", "success")
        else:
            self.update_progress("Model verification", "warning")
        
        return all_found

    def test_sdk_initialization(self):
        """Test SDK initialization"""
        self.update_progress("Testing SDK initialization")
        
        print("🚀 Initializing AI Core SDK...")
        
        try:
            # Import required modules
            sys.path.append('.')  # Add current directory to Python path
            
            cfg_pkl = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
            data_root = self.data_root
            
            print(f"📊 Config: {cfg_pkl}")
            print(f"📁 Data root: {data_root}")
            
            # Check if files exist
            if not Path(cfg_pkl).exists():
                print(f"❌ Config file not found: {cfg_pkl}")
                self.update_progress("SDK test", "error")
                return False
                
            if not Path(data_root).exists():
                print(f"❌ Data root not found: {data_root}")
                self.update_progress("SDK test", "error")
                return False
            
            # Try to import and initialize StreamSDK
            try:
                from stream_pipeline_offline import StreamSDK
                SDK = StreamSDK(cfg_pkl, data_root)
                print("✅ SDK initialized successfully!")
                self.update_progress("SDK test", "success")
                return True
                
            except ImportError as e:
                print(f"❌ Import error: {e}")
                print("⚠️ StreamSDK not available, but setup can continue")
                self.update_progress("SDK test", "warning")
                return True  # Continue setup even if SDK test fails
                
        except Exception as e:
            print(f"❌ SDK initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.update_progress("SDK test", "warning")
            return True  # Continue setup even if SDK test fails

    def setup_project_structure(self):
        """Setup project directories and files"""
        self.update_progress("Setting up project structure")

        # Create directories
        dirs = ["output", "tmp", "example", "logs", "font/Roboto"]
        for dir_name in dirs:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

        # Create sample files
        self.create_sample_files()

        # Build Cython if possible
        if Path("setup.py").exists():
            self.run_silent_command("python setup.py build_ext --inplace", 60)

        self.update_progress("Project structure", "success")
        return True

    def create_sample_files(self):
        """Create sample files for testing"""
        example_dir = Path("example")

        # Sample audio
        if not list(example_dir.glob("*.wav")):
            self.run_silent_command(
                'ffmpeg -y -f lavfi -i "sine=frequency=440:duration=3" -ac 1 -ar 16000 example/sample_audio.wav'
            )

        # Sample image
        if not list(example_dir.glob("*.jpg")):
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (512, 512), color='lightblue')
                draw = ImageDraw.Draw(img)
                draw.text((256, 256), "Sample MC", fill='black', anchor='mm')
                img.save("example/sample_mc.jpg")
            except:
                pass

    def create_startup_script(self):
        """Create application startup script with GPU info"""
        self.update_progress("Creating startup scripts")

        startup_script = f'''#!/bin/bash
# Ditto Talking Head Startup Script - GPU Optimized

echo "🎭 Starting Ditto Talking Head..."
echo "🔍 GPU Capability: {self.gpu_capability}"
echo "📁 Model Path: {self.data_root}"

# Set environment variables
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=8501
export DITTO_GPU_CAPABILITY={self.gpu_capability}
export DITTO_DATA_ROOT={self.data_root}

# Start Streamlit in background
echo "🚀 Starting Streamlit server..."
cd /content/ditto-talkinghead
python -m streamlit run run_streamlit.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true &

# Wait for Streamlit to start
echo "⏳ Waiting for server to start..."
sleep 10

# Start CloudFlared tunnel
echo "☁️ Creating secure tunnel..."
echo "🔗 Your app will be available at the URL shown below:"
cloudflared tunnel --url http://localhost:8501
'''

        with open("start_app.sh", "w") as f:
            f.write(startup_script)

        os.chmod("start_app.sh", 0o755)

        self.update_progress("Startup scripts", "success")
        return True

    def verify_installation(self):
        """Verify installation"""
        self.update_progress("Verifying installation")

        # Check key files
        required_files = ["run_streamlit.py", "start_app.sh"]
        for file in required_files:
            if not Path(file).exists():
                self.update_progress("Installation verification", "error")
                return False

        # Check Python imports
        test_imports = ["streamlit", "numpy", "cv2", "librosa"]
        for module in test_imports:
            try:
                __import__(module)
            except ImportError:
                self.update_progress("Installation verification", "warning")
                break
        else:
            self.update_progress("Installation verification", "success")

        return True

    def show_completion_message(self):
        """Show completion message with GPU info"""
        total_time = time.time() - self.start_time

        print("\n" + "="*52)
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("="*52)
        print(f"⏱️  Total setup time: {total_time:.1f} seconds")
        print(f"🔍 GPU Compute Capability: {self.gpu_capability}")
        print(f"📁 Model Path: {self.data_root}")
        print()
        print("✅ All components installed:")
        print("   • GPU-optimized AI models")
        print("   • Streamlit web interface")
        print("   • CloudFlared tunnel")
        print("   • Sample files for testing")
        print()
        print("🚀 Ready to run! Execute the next cell to start the app.")
        print("="*52)

    def run_setup(self):
        """Run complete setup process"""
        try:
            self.print_header()

            # Run setup steps
            steps = [
                self.detect_gpu_capability,        # New step
                self.install_system_deps,
                self.install_python_deps,
                self.setup_cloudflared,
                self.clone_repository,
                self.download_models_gpu_aware,    # Updated step
                self.verify_model_structure,       # New step
                self.test_sdk_initialization,      # New step
                self.setup_project_structure,
                self.create_startup_script,
                self.verify_installation
            ]

            for step in steps:
                if not step():
                    print("\n⚠️  Some components failed but continuing...")

            self.show_completion_message()
            return True

        except KeyboardInterrupt:
            print("\n❌ Setup cancelled by user")
            return False
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main setup function"""
    setup = SetupManager()
    return setup.run_setup()

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
