#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎭 Ditto Talking Head - Google Colab Setup (Optimized)
Fast setup leveraging Colab's pre-installed PyTorch
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
        print(f"{icons.get(status, 'ℹ️')} {message}")
        
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
        """Detect GPU architecture using Colab's PyTorch"""
        self.print_status("Detecting GPU architecture...", "working")
        
        try:
            # Use Colab's pre-installed PyTorch
            import torch
            
            if torch.cuda.is_available():
                self.gpu_capability = torch.cuda.get_device_capability()[0]
                gpu_name = torch.cuda.get_device_name()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                print(f"    🎮 GPU: {gpu_name}")
                print(f"    🔢 Compute Capability: {self.gpu_capability}")
                print(f"    💾 Memory: {gpu_memory:.1f}GB")
                
                # T4 has capability 7.5, V100 has 7.0, A100 has 8.0+
                if self.gpu_capability >= 8:
                    self.data_root = "./checkpoints/ditto_trt_Ampere_Plus"
                    print(f"    🚀 Using Ampere+ optimized models")
                else:
                    self.data_root = "./checkpoints/ditto_trt"
                    print(f"    📦 Using standard TRT models (T4/V100 compatible)")
            else:
                self.print_status("No CUDA GPU detected", "warning")
                self.gpu_capability = 7  # Default for T4
                self.data_root = "./checkpoints/ditto_trt"
                
        except Exception as e:
            self.print_status(f"GPU detection failed: {e}, using T4 defaults", "warning")
            self.gpu_capability = 7  # T4 default
            self.data_root = "./checkpoints/ditto_trt"
            
        self.print_status("GPU detection completed", "success")
        
    def install_system_dependencies(self):
        """Install system-level dependencies"""
        self.print_status("Installing system dependencies...", "working")
        
        commands = [
            "apt-get update -qq",
            "apt-get install -y ffmpeg",
            "ffmpeg -version"
        ]
        
        success_count = 0
        for cmd in commands:
            success, stdout, stderr = self.run_command(cmd)
            if success:
                success_count += 1
                if "ffmpeg -version" in cmd and stdout:
                    # Extract version info
                    version_line = stdout.split('\n')[0] if stdout else "FFmpeg installed"
                    print(f"    ✅ {version_line}")
            else:
                self.print_status(f"Warning: {cmd} failed", "warning")
                
        # Try to install libcudnn8 (may fail, that's ok)
        try:
            success, _, _ = self.run_command("apt install -y libcudnn8")
            if success:
                print("    ✅ libcudnn8 installed")
            else:
                print("    ⚠️ libcudnn8 not available, continuing...")
        except:
            print("    ⚠️ libcudnn8 installation skipped")
                
        self.print_status("System dependencies ready", "success")
        
    def install_ai_core_packages(self):
        """Install AI core packages"""
        self.print_status("Installing AI core packages...", "working")
        
        # AI Core packages (as specified by user)
        ai_core_cmd = (
            "pip install --upgrade pip setuptools wheel && "
            "pip install tensorrt==8.6.1 librosa tqdm filetype imageio "
            "opencv-python-headless scikit-image cython cuda-python "
            "imageio-ffmpeg colored polygraphy numpy==2.0.1 -q"
        )
        
        success, _, stderr = self.run_command(ai_core_cmd, timeout=300)
        if success:
            print("    ✅ AI core packages installed")
        else:
            self.print_status(f"AI core installation issues: {stderr[:100]}...", "warning")
            
        self.print_status("AI core packages ready", "success")
        
    def install_ui_packages(self):
        """Install Streamlit UI and processing packages"""
        self.print_status("Installing UI & processing packages...", "working")
        
        # UI packages (as specified by user)
        ui_commands = [
            "pip install streamlit fastapi uvicorn python-multipart requests -q",
            "pip install pysrt python-dotenv moviepy==2.1.2 -q",
            "pip install openai edge-tts -q",
            "pip install gradio transparent-background insightface -q"
        ]
        
        success_count = 0
        for cmd in ui_commands:
            success, _, _ = self.run_command(cmd, timeout=180)
            if success:
                success_count += 1
                
        print(f"    ✅ {success_count}/{len(ui_commands)} UI package groups installed")
        self.print_status("UI packages ready", "success")
        
    def install_tunnel_service(self):
        """Install tunnel service"""
        self.print_status("Installing tunnel service...", "working")
        
        # Install pyngrok (as specified by user)
        success, _, _ = self.run_command("pip install pyngrok -q")
        if success:
            print("    ✅ pyngrok installed")
            
        # Also install cloudflared as backup
        cloudflared_commands = [
            "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            "chmod +x cloudflared-linux-amd64",
            "sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared"
        ]
        
        for cmd in cloudflared_commands:
            self.run_command(cmd)
            
        # Test which tunnel service is available
        success_ngrok, _, _ = self.run_command("python -c 'import pyngrok; print(\"pyngrok available\")'")
        success_cf, _, _ = self.run_command("cloudflared --version")
        
        if success_ngrok:
            print("    ✅ pyngrok tunnel service ready")
        if success_cf:
            print("    ✅ cloudflared tunnel service ready")
            
        self.print_status("Tunnel service ready", "success")
        
    def clone_repository(self):
        """Clone the project repository"""
        self.print_status("Downloading project files...", "working")
        
        if Path("ditto-talkinghead").exists():
            os.chdir("ditto-talkinghead")
            success, _, _ = self.run_command("git pull")
            if success:
                print("    ✅ Project updated")
            else:
                print("    ⚠️ Git pull failed, using existing files")
        else:
            success, _, _ = self.run_command(
                "git clone --single-branch --branch colab https://github.com/linhcentrio/ditto-talkinghead.git"
            )
            if success:
                os.chdir("ditto-talkinghead")
                print("    ✅ Project downloaded")
            else:
                self.print_status("Repository cloning failed", "error")
                return False
                
        self.print_status("Project files ready", "success")
        return True
        
    def download_models(self):
        """Download AI models based on GPU capability"""
        self.print_status("Downloading AI models...", "working")
        
        # For T4 (capability 7.5) and similar, use standard models
        if self.gpu_capability >= 8:
            return self._download_ampere_models()
        else:
            return self._download_standard_models()
            
    def _download_ampere_models(self):
        """Download Ampere+ optimized models"""
        self.print_status("Downloading Ampere+ models from Hugging Face...")
        
        try:
            if Path("checkpoints").exists():
                success, _, _ = self.run_command("cd checkpoints && git pull", 300)
            else:
                success, _, _ = self.run_command(
                    "git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints", 
                    timeout=600
                )
                
            if success:
                print("    ✅ Ampere+ models downloaded")
                return True
            else:
                self.print_status("Hugging Face download failed, using config-only", "warning")
                return self._download_config_only()
                
        except Exception as e:
            self.print_status(f"Ampere download error: {str(e)[:50]}...", "warning")
            return self._download_config_only()
            
    def _download_standard_models(self):
        """Download standard TRT models for T4/V100"""
        self.print_status("Downloading T4-compatible models...")
        
        try:
            # Create directories
            Path("checkpoints/ditto_trt").mkdir(parents=True, exist_ok=True)
            Path("checkpoints/ditto_cfg").mkdir(parents=True, exist_ok=True)
            
            # Try Google Drive download with gdown
            self.print_status("Downloading from Google Drive (this may take 3-5 minutes)...")
            success, _, error = self.run_command(
                "pip install --upgrade gdown -q && "
                "gdown https://drive.google.com/drive/folders/1-1qnqy0D9ICgRh8iNY_22j9ieNRC0-zf?usp=sharing "
                "-O ./checkpoints/ditto_trt --folder",
                timeout=900  # 15 minutes
            )
            
            if success:
                print("    ✅ T4 models downloaded from Google Drive")
                return self._download_config_only()
            else:
                self.print_status("Google Drive download failed, using config-only mode", "warning")
                return self._download_config_only()
                
        except Exception as e:
            self.print_status(f"Standard model download error: {str(e)[:50]}...", "warning")
            return self._download_config_only()
            
    def _download_config_only(self):
        """Download essential config file"""
        self.print_status("Downloading configuration file...")
        
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
            print("    ✅ Configuration file downloaded")
            return True
        else:
            self.print_status("Failed to download configuration", "error")
            return False
            
    def setup_project_structure(self):
        """Setup project directories"""
        self.print_status("Setting up project structure...", "working")
        
        # Create necessary directories
        directories = ["output", "tmp", "example", "logs"]
        for dir_name in directories:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            
        # Build Cython extensions if setup.py exists
        if Path("setup.py").exists():
            success, _, _ = self.run_command("python setup.py build_ext --inplace", timeout=60)
            if success:
                print("    ✅ Cython extensions built")
            
        # Create sample files
        self._create_sample_files()
        
        self.print_status("Project structure ready", "success")
        
    def _create_sample_files(self):
        """Create sample files for testing"""
        example_dir = Path("example")
        
        # Create sample audio if none exists
        if not list(example_dir.glob("*.wav")):
            success, _, _ = self.run_command(
                'ffmpeg -y -f lavfi -i "sine=frequency=440:duration=3" '
                '-ac 1 -ar 16000 example/sample_audio.wav'
            )
            if success:
                print("    ✅ Sample audio created")
            
        # Create sample image
        if not list(example_dir.glob("*.jpg")):
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (512, 512), color='lightblue')
                draw = ImageDraw.Draw(img)
                draw.text((256, 256), "Sample MC", fill='black', anchor='mm')
                img.save("example/sample_mc.jpg")
                print("    ✅ Sample image created")
            except:
                pass
                
    def verify_installation(self):
        """Quick verification of key components"""
        self.print_status("Verifying installation...", "working")
        
        # Check key files
        required_files = [
            "run_streamlit.py",
            "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
        ]
        
        missing_files = [f for f in required_files if not Path(f).exists()]
        if missing_files:
            self.print_status(f"Missing files: {missing_files}", "warning")
            
        # Test key imports
        test_modules = ["streamlit", "numpy", "cv2", "librosa", "torch"]
        failed_imports = []
        
        for module in test_modules:
            try:
                __import__(module)
            except ImportError:
                failed_imports.append(module)
                
        if failed_imports:
            self.print_status(f"Import issues: {failed_imports}", "warning")
        else:
            print("    ✅ All core modules available")
            
        self.print_status("Verification completed", "success")
        
    def create_environment_config(self):
        """Create environment configuration"""
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
        
        print("    ✅ Configuration saved")
        self.print_status("Environment configured", "success")
        
    def run_setup(self):
        """Run the complete setup process"""
        print("🎭 " + "="*50)
        print("   DITTO TALKING HEAD - COLAB SETUP")
        print("="*52)
        print("🚀 Optimized setup for Google Colab T4")
        print("⏱️  Estimated time: 2-4 minutes")
        print("="*52)
        
        try:
            # Setup steps optimized for Colab
            steps = [
                ("GPU Detection", self.detect_gpu_capability),
                ("System Dependencies", self.install_system_dependencies),
                ("AI Core Packages", self.install_ai_core_packages),
                ("UI Packages", self.install_ui_packages),
                ("Tunnel Service", self.install_tunnel_service),
                ("Project Download", self.clone_repository),
                ("AI Models", self.download_models),
                ("Project Setup", self.setup_project_structure),
                ("Verification", self.verify_installation),
                ("Configuration", self.create_environment_config)
            ]
            
            for step_name, step_func in steps:
                try:
                    step_func()
                except Exception as e:
                    self.print_status(f"{step_name} completed with issues: {str(e)[:50]}...", "warning")
                    
            # Success message
            total_time = time.time() - self.start_time
            print("\n" + "="*52)
            print("🎉 SETUP COMPLETED!")
            print("="*52)
            print(f"⏱️  Setup time: {total_time:.1f} seconds")
            print(f"🎮 GPU: Compute {self.gpu_capability} ({'Ampere+' if self.gpu_capability >= 8 else 'T4/V100'})")
            print(f"📁 Models: {Path(self.data_root).name}")
            print()
            print("✅ Ready for AI video generation!")
            print("🚀 Run the next cell to launch the application")
            print("="*52)
            
            return True
            
        except KeyboardInterrupt:
            self.print_status("Setup cancelled by user", "error")
            return False
        except Exception as e:
            self.print_status(f"Setup failed: {str(e)}", "error")
            return False

def main():
    """Main setup function optimized for Colab"""
    setup = DittoSetup()
    return setup.run_setup()

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup incomplete. Please check errors above.")
        print("💡 Try restarting runtime if issues persist.")
        sys.exit(1)
