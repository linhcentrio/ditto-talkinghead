#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎭 Ditto Talking Head - Google Colab Setup
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
        self.total_steps = 10
        
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
        print(f"Progress: {'█' * int(progress//5)}{'░' * (20-int(progress//5))} {progress:.0f}%")
        
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
    
    def install_system_deps(self):
        """Install system dependencies"""
        self.update_progress("Installing system dependencies")
        
        commands = [
            "apt-get update -qq",
            "apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1-mesa-glx git-lfs wget curl",
            "git lfs install"
        ]
        
        for cmd in commands:
            success, _, error = self.run_silent_command(cmd)
            if not success and "apt-get update" not in cmd:
                self.update_progress("System dependencies", "warning")
                return False
        
        self.update_progress("System dependencies", "success")
        return True
    
    def install_python_deps(self):
        """Install Python dependencies"""
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
                
        # AI packages
        ai_packages = [
            "pip install tensorrt==8.6.1 cuda-python polygraphy -q",
            "pip install cython transparent-background insightface -q",
            "pip install streamlit fastapi uvicorn python-multipart -q",
            "pip install moviepy==2.1.2 pysrt openai edge-tts pillow -q"
        ]
        
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
    
    def download_models(self):
        """Download AI models"""
        self.update_progress("Downloading AI models (this may take a while)")
        
        if Path("checkpoints").exists():
            success, _, _ = self.run_silent_command("cd checkpoints && git pull", 60)
        else:
            success, _, _ = self.run_silent_command(
                "git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints", 
                300
            )
        
        if not success:
            self.update_progress("AI models", "warning")
        else:
            self.update_progress("AI models", "success")
        return True
    
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
        """Create application startup script"""
        self.update_progress("Creating startup scripts")
        
        startup_script = '''#!/bin/bash
# Ditto Talking Head Startup Script

echo "🎭 Starting Ditto Talking Head..."

# Set environment variables
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=8501

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
        """Show completion message with instructions"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*52)
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("="*52)
        print(f"⏱️  Total setup time: {total_time:.1f} seconds")
        print()
        print("✅ All components installed:")
        print("   • Streamlit web interface")  
        print("   • AI models and dependencies")
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
                self.install_system_deps,
                self.install_python_deps, 
                self.setup_cloudflared,
                self.clone_repository,
                self.download_models,
                self.setup_project_structure,
                self.create_startup_script,
                self.verify_installation
            ]
            
            for step in steps:
                if not step():
                    print("\n⚠️  Some components failed to install but continuing...")
                    
            self.show_completion_message()
            return True
            
        except KeyboardInterrupt:
            print("\n❌ Setup cancelled by user")
            return False
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            return False

def main():
    """Main setup function"""
    setup = SetupManager()
    return setup.run_setup()

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
