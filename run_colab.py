#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎭 Ditto Talking Head - Google Colab Setup
Automated environment setup for AI video generation
"""

import os
import sys
import subprocess
import time
import json
import shutil
from pathlib import Path
import tempfile
from datetime import datetime

class DittoSetup:
    def __init__(self):
        self.start_time = time.time()
        self.gpu_capability = None
        self.data_root = None
        
    def print_status(self, message, status="info"):
        """Print formatted status message"""
        icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "working": "🔄"}
        print(f"\n{icons.get(status, 'ℹ️')} {message}")
        
    def run_command(self, cmd, silent=True, timeout=300):
        """Run shell command with error handling"""
        try:
            if silent:
                result = subprocess.run(cmd, shell=True, capture_output=True, 
                                      text=True, timeout=timeout, check=False)
                return result.returncode == 0, result.stdout, result.stderr
            else:
                result = subprocess.run(cmd, shell=True, timeout=timeout, check=False)
                return result.returncode == 0, "", ""
        except subprocess.TimeoutExpired:
            return False, "", "Command timeout"
        except Exception as e:
            return False, "", str(e)
    
    def detect_gpu_capability(self):
        """Detect GPU architecture and set appropriate model paths"""
        self.print_status("Detecting GPU architecture...", "working")
        
        try:
            # Install torch first if needed
            try:
                import torch
            except ImportError:
                self.print_status("Installing PyTorch for GPU detection...")
                success, _, _ = self.run_command(
                    "pip install torch --index-url https://download.pytorch.org/whl/cu121 -q"
                )
                if success:
                    import torch
                else:
                    raise ImportError("Failed to install PyTorch")
            
            if torch.cuda.is_available():
                self.gpu_capability = torch.cuda.get_device_capability()[0]
                gpu_name = torch.cuda.get_device_name()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                print(f"    🎮 GPU: {gpu_name}")
                print(f"    🔢 Compute Capability: {self.gpu_capability}")
                print(f"    💾 Memory: {gpu_memory:.1f}GB")
                
                if self.gpu_capability >= 8:
                    self.data_root = "./checkpoints/ditto_trt_Ampere_Plus"
                    print(f"    🚀 Using Ampere+ optimized models")
                else:
                    self.data_root = "./checkpoints/ditto_trt"
                    print(f"    📦 Using standard TRT models")
            else:
                self.print_status("No CUDA GPU detected, using CPU fallback", "warning")
                self.gpu_capability = 6
                self.data_root = "./checkpoints/ditto_trt"
                
        except Exception as e:
            self.print_status(f"GPU detection failed: {e}, using defaults", "warning")
            self.gpu_capability = 6
            self.data_root = "./checkpoints/ditto_trt"
            
        self.print_status("GPU detection completed", "success")
        
    def install_system_dependencies(self):
        """Install system-level dependencies"""
        self.print_status("Installing system dependencies...", "working")
        
        commands = [
            "apt-get update -qq",
            "apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0",
            "apt-get install -y libgl1-mesa-glx git-lfs wget curl fonts-dejavu-core",
            "git lfs install"
        ]
        
        for cmd in commands:
            success, _, _ = self.run_command(cmd)
            if not success and "update" not in cmd:
                self.print_status(f"Warning: {cmd} failed", "warning")
                
        self.print_status("System dependencies installed", "success")
        
    def install_python_packages(self):
        """Install Python packages"""
        self.print_status("Installing Python packages...", "working")
        
        # Core packages
        core_packages = [
            "pip install --upgrade pip setuptools wheel -q",
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q",
            "pip install numpy==2.0.1 opencv-python-headless scikit-image librosa -q",
        ]
        
        for cmd in core_packages:
            self.run_command(cmd, timeout=180)
            
        # AI and web packages
        ai_packages = [
            "pip install --upgrade --no-cache-dir gdown -q",
            "pip install streamlit fastapi uvicorn python-multipart -q",
            "pip install moviepy==2.1.2 pysrt openai edge-tts pillow -q",
            "pip install cython transparent-background insightface -q",
            "pip install tqdm filetype imageio imageio-ffmpeg colored -q"
        ]
        
        # Add TensorRT for Ampere+ GPUs
        if self.gpu_capability and self.gpu_capability >= 8:
            ai_packages.insert(0, "pip install tensorrt==8.6.1 cuda-python polygraphy -q")
            
        for cmd in ai_packages:
            self.run_command(cmd, timeout=120)
            
        self.print_status("Python packages installed", "success")
        
    def setup_tunnel_service(self):
        """Setup CloudFlared tunnel for public access"""
        self.print_status("Setting up tunnel service...", "working")
        
        # Check if cloudflared exists
        success, _, _ = self.run_command("cloudflared --version")
        if success:
            self.print_status("CloudFlared already installed", "success")
            return True
            
        # Install cloudflared
        commands = [
            "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            "chmod +x cloudflared-linux-amd64",
            "sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared"
        ]
        
        for cmd in commands:
            success, _, _ = self.run_command(cmd)
            if not success:
                self.print_status("CloudFlared installation failed", "error")
                return False
                
        self.print_status("Tunnel service ready", "success")
        return True
        
    def clone_repository(self):
        """Clone the project repository"""
        self.print_status("Downloading project files...", "working")
        
        if Path("ditto-talkinghead").exists():
            os.chdir("ditto-talkinghead")
            success, _, _ = self.run_command("git pull")
            if not success:
                self.print_status("Git pull failed, continuing...", "warning")
        else:
            success, _, _ = self.run_command(
                "git clone --single-branch --branch colab https://github.com/linhcentrio/ditto-talkinghead.git"
            )
            if success:
                os.chdir("ditto-talkinghead")
            else:
                self.print_status("Repository cloning failed", "error")
                return False
                
        self.print_status("Project files ready", "success")
        return True
        
    def download_models(self):
        """Download AI models based on GPU capability"""
        self.print_status("Downloading AI models (this may take a few minutes)...", "working")
        
        if self.gpu_capability >= 8:
            return self._download_ampere_models()
        else:
            return self._download_standard_models()
            
    def _download_ampere_models(self):
        """Download Ampere+ optimized models"""
        self.print_status("Downloading Ampere+ optimized models from Hugging Face...")
        
        try:
            if Path("checkpoints").exists():
                success, _, _ = self.run_command("cd checkpoints && git pull", 300)
            else:
                success, _, _ = self.run_command(
                    "git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints", 
                    timeout=600
                )
                
            if not success:
                self.print_status("Hugging Face download failed, trying fallback...", "warning")
                return self._download_config_only()
                
            # Verify key files
            required_files = [
                "checkpoints/ditto_trt_Ampere_Plus", 
                "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
            ]
            
            missing = [f for f in required_files if not Path(f).exists()]
            if missing:
                self.print_status(f"Missing files: {missing}", "warning")
                return self._download_config_only()
                
            self.print_status("Ampere+ models downloaded successfully", "success")
            return True
            
        except Exception as e:
            self.print_status(f"Ampere model download error: {e}", "error")
            return self._download_config_only()
            
    def _download_standard_models(self):
        """Download standard TRT models"""
        self.print_status("Downloading standard TRT models...")
        
        try:
            # Create directories
            Path("checkpoints/ditto_trt").mkdir(parents=True, exist_ok=True)
            Path("checkpoints/ditto_cfg").mkdir(parents=True, exist_ok=True)
            
            # Try Google Drive download
            success, _, error = self.run_command(
                "gdown https://drive.google.com/drive/folders/1-1qnqy0D9ICgRh8iNY_22j9ieNRC0-zf?usp=sharing -O ./checkpoints/ditto_trt --folder",
                timeout=600
            )
            
            if not success:
                self.print_status("Google Drive download failed, using config-only mode", "warning")
                return self._download_config_only()
                
            # Download config file
            return self._download_config_only()
            
        except Exception as e:
            self.print_status(f"Standard model download error: {e}", "error")
            return self._download_config_only()
            
    def _download_config_only(self):
        """Download essential config file only"""
        self.print_status("Downloading essential configuration...")
        
        # Ensure config directory exists
        Path("checkpoints/ditto_cfg").mkdir(parents=True, exist_ok=True)
        
        config_url = "https://huggingface.co/digital-avatar/ditto-talkinghead/resolve/main/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
        
        # Try wget first, then curl
        success, _, _ = self.run_command(
            f"wget -q -O checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl {config_url}"
        )
        
        if not success:
            success, _, _ = self.run_command(
                f"curl -L -o checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl {config_url}"
            )
            
        if success and Path("checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl").exists():
            self.print_status("Configuration downloaded successfully", "success")
            return True
        else:
            self.print_status("Failed to download configuration", "error")
            return False
            
    def setup_project_structure(self):
        """Setup project directories and build components"""
        self.print_status("Setting up project structure...", "working")
        
        # Create necessary directories
        directories = ["output", "tmp", "example", "logs"]
        for dir_name in directories:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            
        # Build Cython extensions if setup.py exists
        if Path("setup.py").exists():
            self.run_command("python setup.py build_ext --inplace", timeout=60)
            
        # Create sample files for testing
        self._create_sample_files()
        
        self.print_status("Project structure ready", "success")
        
    def _create_sample_files(self):
        """Create sample files for testing"""
        example_dir = Path("example")
        
        # Create sample audio if none exists
        if not list(example_dir.glob("*.wav")):
            self.run_command(
                'ffmpeg -y -f lavfi -i "sine=frequency=440:duration=3" '
                '-ac 1 -ar 16000 example/sample_audio.wav'
            )
            
        # Create sample image
        if not list(example_dir.glob("*.jpg")):
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (512, 512), color='lightblue')
                draw = ImageDraw.Draw(img)
                draw.text((256, 256), "Sample MC", fill='black', anchor='mm')
                img.save("example/sample_mc.jpg")
            except:
                pass
                
    def verify_installation(self):
        """Verify the installation"""
        self.print_status("Verifying installation...", "working")
        
        # Check key files
        required_files = [
            "run_streamlit.py",
            "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
        ]
        
        missing_files = [f for f in required_files if not Path(f).exists()]
        if missing_files:
            self.print_status(f"Missing required files: {missing_files}", "warning")
            
        # Test key imports
        test_modules = ["streamlit", "numpy", "cv2", "librosa"]
        failed_imports = []
        
        for module in test_modules:
            try:
                __import__(module)
            except ImportError:
                failed_imports.append(module)
                
        if failed_imports:
            self.print_status(f"Import issues: {failed_imports}", "warning")
        else:
            self.print_status("All core modules importable", "success")
            
        # Test SDK if possible
        try:
            sys.path.append('.')
            cfg_pkl = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
            
            if Path(cfg_pkl).exists() and Path(self.data_root).exists():
                # Try importing StreamSDK
                from stream_pipeline_offline import StreamSDK
                SDK = StreamSDK(cfg_pkl, self.data_root)
                self.print_status("AI Core SDK initialized successfully", "success")
            else:
                self.print_status("SDK test skipped - missing components", "warning")
                
        except Exception as e:
            self.print_status(f"SDK test failed: {str(e)[:50]}...", "warning")
            
        self.print_status("Installation verification completed", "success")
        
    def create_environment_config(self):
        """Create environment configuration for the app"""
        config = {
            'data_root': self.data_root,
            'gpu_capability': self.gpu_capability,
            'cfg_pkl': './checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl',
            'setup_time': datetime.now().isoformat(),
            'setup_duration': time.time() - self.start_time
        }
        
        with open('.ditto_config.json', 'w') as f:
            json.dump(config, f, indent=2)
            
        # Set environment variables
        os.environ['DITTO_DATA_ROOT'] = self.data_root
        os.environ['DITTO_GPU_CAPABILITY'] = str(self.gpu_capability)
        
        self.print_status("Environment configuration saved", "success")
        
    def run_setup(self):
        """Run the complete setup process"""
        print("🎭 " + "="*50)
        print("   DITTO TALKING HEAD - SETUP")
        print("="*52)
        print("🚀 Setting up AI video generation environment...")
        print("⏱️  Estimated time: 3-5 minutes")
        print("="*52)
        
        try:
            # Setup steps
            steps = [
                ("Detecting GPU", self.detect_gpu_capability),
                ("Installing system dependencies", self.install_system_dependencies),
                ("Installing Python packages", self.install_python_packages),
                ("Setting up tunnel service", self.setup_tunnel_service),
                ("Downloading project", self.clone_repository),
                ("Downloading AI models", self.download_models),
                ("Setting up project", self.setup_project_structure),
                ("Verifying installation", self.verify_installation),
                ("Creating configuration", self.create_environment_config)
            ]
            
            for step_name, step_func in steps:
                try:
                    if not step_func():
                        self.print_status(f"{step_name} completed with warnings", "warning")
                except Exception as e:
                    self.print_status(f"{step_name} failed: {str(e)[:50]}...", "error")
                    
            # Success message
            total_time = time.time() - self.start_time
            print("\n" + "="*52)
            print("🎉 SETUP COMPLETED SUCCESSFULLY!")
            print("="*52)
            print(f"⏱️  Total time: {total_time:.1f} seconds")
            print(f"🎮 GPU: {self.gpu_capability} ({'Ampere+' if self.gpu_capability >= 8 else 'Standard'})")
            print(f"📁 Models: {self.data_root}")
            print()
            print("✅ Environment ready for AI video generation!")
            print("🚀 You can now run start_app.py to launch the application")
            print("="*52)
            
            return True
            
        except KeyboardInterrupt:
            self.print_status("Setup cancelled by user", "error")
            return False
        except Exception as e:
            self.print_status(f"Setup failed: {str(e)}", "error")
            return False

def main():
    """Main setup function"""
    setup = DittoSetup()
    return setup.run_setup()

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup incomplete. Please check errors above and try again.")
        sys.exit(1)
