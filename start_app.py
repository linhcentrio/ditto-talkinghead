#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Ditto Talking Head - Standalone Application Launcher
No config file required - auto-detect everything!
"""

import os
import sys
import subprocess
import time
import threading
import signal
from pathlib import Path
from datetime import datetime

class StandaloneLauncher:
    def __init__(self):
        self.streamlit_process = None
        self.tunnel_process = None
        self.public_url = None
        self.config = None
        
    def print_status(self, message, status="info"):
        """Print formatted status message"""
        icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "working": "🔄"}
        print(f"{icons.get(status, 'ℹ️')} {message}")
        
    def auto_detect_environment(self):
        """Auto-detect environment without config file"""
        self.print_status("Auto-detecting environment...", "working")
        
        # Default configuration
        config = {
            'data_root': './checkpoints/ditto_trt',
            'gpu_capability': 7,  # Default for T4
            'cfg_pkl': './checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl',
            'setup_time': datetime.now().isoformat(),
            'setup_type': 'auto_detected'
        }
        
        try:
            # Try to detect GPU capability
            import torch
            if torch.cuda.is_available():
                gpu_capability = torch.cuda.get_device_capability()[0]
                gpu_name = torch.cuda.get_device_name()
                config['gpu_capability'] = gpu_capability
                
                # Choose model path based on GPU
                if gpu_capability >= 8:
                    config['data_root'] = './checkpoints/ditto_trt_Ampere_Plus'
                    print(f"    🚀 Detected: {gpu_name} (Ampere+)")
                else:
                    config['data_root'] = './checkpoints/ditto_trt' 
                    print(f"    🎮 Detected: {gpu_name} (T4/V100)")
            else:
                print(f"    ⚠️ No GPU detected, using CPU fallback")
                
        except Exception as e:
            print(f"    ⚠️ GPU detection failed: {str(e)[:50]}..., using defaults")
            
        # Set environment variables
        os.environ['DITTO_DATA_ROOT'] = config['data_root']
        os.environ['DITTO_GPU_CAPABILITY'] = str(config['gpu_capability'])
        
        self.config = config
        self.print_status("Environment auto-detected successfully", "success")
        return True
        
    def find_project_files(self):
        """Find and setup project files"""
        self.print_status("Locating project files...", "working")
        
        # Possible locations for run_streamlit.py
        possible_locations = [
            "run_streamlit.py",
            "ditto-talkinghead/run_streamlit.py",
            "./ditto-talkinghead/run_streamlit.py"
        ]
        
        # Find the main app file
        app_file = None
        for location in possible_locations:
            if Path(location).exists():
                app_file = location
                break
                
        if app_file:
            # Change to the correct directory
            if "ditto-talkinghead" in app_file:
                if not os.getcwd().endswith("ditto-talkinghead"):
                    os.chdir("ditto-talkinghead")
                    self.print_status("Changed to project directory", "info")
                app_file = "run_streamlit.py"
                
            self.print_status(f"Found app file: {app_file}", "success")
            return app_file
        else:
            self.print_status("Project files not found, attempting download...", "warning")
            return self._download_project_files()
            
    def _download_project_files(self):
        """Download project files if not found"""
        try:
            self.print_status("Downloading project from GitHub...", "working")
            
            # Clone repository
            cmd = "git clone --single-branch --branch colab https://github.com/linhcentrio/ditto-talkinghead.git"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and Path("ditto-talkinghead/run_streamlit.py").exists():
                os.chdir("ditto-talkinghead")
                self.print_status("Project downloaded successfully", "success")
                return "run_streamlit.py"
            else:
                self.print_status("Failed to download project", "error")
                return None
                
        except Exception as e:
            self.print_status(f"Download failed: {str(e)}", "error")
            return None
            
    def ensure_dependencies(self):
        """Ensure basic dependencies are installed"""
        self.print_status("Checking dependencies...", "working")
        
        # Check critical packages
        critical_packages = {
            'streamlit': 'streamlit',
            'cv2': 'opencv-python-headless', 
            'numpy': 'numpy',
            'PIL': 'pillow'
        }
        
        missing_packages = []
        for module, package in critical_packages.items():
            try:
                __import__(module)
            except ImportError:
                missing_packages.append(package)
                
        if missing_packages:
            self.print_status(f"Installing missing packages: {', '.join(missing_packages)}", "working")
            
            install_cmd = f"pip install {' '.join(missing_packages)} -q"
            result = subprocess.run(install_cmd, shell=True, capture_output=True)
            
            if result.returncode == 0:
                self.print_status("Dependencies installed", "success")
            else:
                self.print_status("Some dependencies failed to install", "warning")
        else:
            self.print_status("All critical dependencies available", "success")
            
        return True
        
    def create_minimal_structure(self):
        """Create minimal directory structure"""
        directories = ["output", "tmp", "example", "logs", "checkpoints/ditto_cfg"]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        # Create a minimal config file if it doesn't exist
        config_file = Path("checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl")
        if not config_file.exists():
            self.print_status("Downloading minimal configuration...", "working")
            
            config_url = "https://huggingface.co/digital-avatar/ditto-talkinghead/resolve/main/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
            
            # Try wget first, then curl
            for cmd in [
                f"wget -q -O {config_file} {config_url}",
                f"curl -sL -o {config_file} {config_url}"
            ]:
                result = subprocess.run(cmd, shell=True, capture_output=True)
                if result.returncode == 0:
                    self.print_status("Configuration downloaded", "success")
                    break
            else:
                self.print_status("Failed to download config, will use fallback", "warning")
                
        return True
        
    def start_streamlit(self):
        """Start Streamlit server with auto-configuration"""
        self.print_status("Starting Streamlit server...", "working")
        
        # Set comprehensive environment
        env = os.environ.copy()
        env.update({
            'STREAMLIT_SERVER_FILE_WATCHER_TYPE': 'none',
            'STREAMLIT_SERVER_HEADLESS': 'true',
            'STREAMLIT_SERVER_PORT': '8501',
            'STREAMLIT_SERVER_ADDRESS': '0.0.0.0',
            'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false',
            'STREAMLIT_SERVER_ENABLE_CORS': 'false',
            'STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION': 'false',
            'STREAMLIT_SERVER_MAX_UPLOAD_SIZE': '200'
        })
        
        # Add auto-detected config
        if self.config:
            env.update({
                'DITTO_DATA_ROOT': self.config['data_root'],
                'DITTO_GPU_CAPABILITY': str(self.config['gpu_capability'])
            })
        
        try:
            # Start Streamlit
            self.streamlit_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "run_streamlit.py",
                "--server.port=8501",
                "--server.address=0.0.0.0", 
                "--server.headless=true",
                "--browser.gatherUsageStats=false",
                "--server.fileWatcherType=none",
                "--server.maxUploadSize=200"
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for startup with progress
            self.print_status("Initializing application...", "working")
            startup_time = 0
            max_startup_time = 30
            
            while startup_time < max_startup_time:
                if self.streamlit_process.poll() is not None:
                    break
                    
                time.sleep(1)
                startup_time += 1
                
                if startup_time % 5 == 0:
                    print(f"    ⏳ Starting up... ({startup_time}s)")
                    
            # Check if process is running
            if self.streamlit_process.poll() is None:
                # Try to verify server
                if self._verify_server_basic():
                    self.print_status("Streamlit server started successfully", "success")
                    return True
                else:
                    self.print_status("Server started but verification failed", "warning")
                    return True  # Continue anyway
            else:
                # Process died, check logs
                stdout, stderr = self.streamlit_process.communicate()
                self.print_status("Streamlit failed to start", "error")
                if stderr:
                    print(f"Error: {stderr[:200]}...")
                return False
                
        except Exception as e:
            self.print_status(f"Failed to start server: {str(e)}", "error")
            return False
            
    def _verify_server_basic(self):
        """Basic server verification without external dependencies"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', 8501))
            sock.close()
            return result == 0
        except:
            return False
            
    def start_tunnel(self):
        """Start tunnel with multiple fallback options"""
        self.print_status("Creating public tunnel...", "working")
        
        # Try multiple tunnel services
        tunnel_methods = [
            self._try_ngrok,
            self._try_cloudflared,
            self._try_localtunnel
        ]
        
        for method in tunnel_methods:
            if method():
                return True
                
        self.print_status("All tunnel methods failed", "warning")
        self.print_status("Application available locally only", "info")
        return False
        
    def _try_ngrok(self):
        """Try ngrok tunnel"""
        try:
            # Check if ngrok token is available
            ngrok_token = os.environ.get('NGROK_AUTH_TOKEN')
            
            from pyngrok import ngrok
            
            if ngrok_token:
                ngrok.set_auth_token(ngrok_token)
                
            self.print_status("Creating ngrok tunnel...", "working")
            public_url = ngrok.connect(8501, "http")
            self.public_url = str(public_url)
            self.print_status("Ngrok tunnel created successfully", "success")
            return True
            
        except Exception as e:
            self.print_status(f"Ngrok failed: {str(e)[:50]}...", "warning")
            return False
            
    def _try_cloudflared(self):
        """Try cloudflared tunnel"""
        try:
            self.print_status("Creating CloudFlare tunnel...", "working")
            
            # Check if cloudflared is available
            result = subprocess.run(["cloudflared", "--version"], 
                                  capture_output=True, timeout=5)
            if result.returncode != 0:
                return False
                
            self.tunnel_process = subprocess.Popen([
                "cloudflared", "tunnel", "--url", "http://localhost:8501"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Parse tunnel URL with timeout
            start_time = time.time()
            while time.time() - start_time < 20:
                if self.tunnel_process.poll() is not None:
                    break
                    
                line = self.tunnel_process.stderr.readline()
                if line and "trycloudflare.com" in line:
                    import re
                    url_match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                    if url_match:
                        self.public_url = url_match.group(0)
                        self.print_status("CloudFlare tunnel created successfully", "success")
                        return True
                        
                time.sleep(0.5)
                
            self.print_status("CloudFlare tunnel timeout", "warning")
            return False
            
        except Exception as e:
            self.print_status(f"CloudFlare failed: {str(e)[:50]}...", "warning")
            return False
            
    def _try_localtunnel(self):
        """Try localtunnel as final fallback"""
        try:
            # Install and use localtunnel
            self.print_status("Trying localtunnel...", "working")
            
            # Check if npm/node is available
            result = subprocess.run(["npm", "--version"], capture_output=True, timeout=5)
            if result.returncode != 0:
                return False
                
            # Install localtunnel
            subprocess.run(["npm", "install", "-g", "localtunnel"], 
                         capture_output=True, timeout=30)
            
            # Start tunnel
            self.tunnel_process = subprocess.Popen([
                "lt", "--port", "8501"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Parse output for URL
            start_time = time.time()
            while time.time() - start_time < 15:
                line = self.tunnel_process.stdout.readline()
                if line and "https://" in line:
                    import re
                    url_match = re.search(r'https://[^\s]+', line)
                    if url_match:
                        self.public_url = url_match.group(0)
                        self.print_status("Localtunnel created successfully", "success")
                        return True
                        
            return False
            
        except Exception as e:
            self.print_status(f"Localtunnel failed: {str(e)[:50]}...", "warning")
            return False
            
    def display_access_info(self):
        """Display access information"""
        print("\n" + "="*60)
        print("🎭 AI VIDEO CREATOR - READY!")
        print("="*60)
        
        if self.public_url:
            print(f"\n🔗 **PUBLIC URL:**")
            print(f"   {self.public_url}")
            print(f"\n📱 Click the link above to access your AI Video Creator")
            print(f"🌍 Share this URL with others if needed")
        else:
            print(f"\n🔗 **LOCAL ACCESS:**")
            print(f"   http://localhost:8501")
            print(f"📱 Available within this environment only")
            
        if self.config:
            gpu_info = f"GPU Capability {self.config['gpu_capability']}"
            model_info = Path(self.config['data_root']).name
            print(f"\n🎮 **System**: {gpu_info}")
            print(f"🤖 **AI Models**: {model_info}")
            
        print(f"\n🎬 **Quick Start:**")
        print(f"   1. Upload MC image/video + background")
        print(f"   2. Add audio OR enter text for AI speech")  
        print(f"   3. Click 'Create Video' and wait for magic!")
        
        print(f"\n🛑 **To Stop**: Interrupt this cell (■ button)")
        print("="*60)
        
    def cleanup(self):
        """Clean up all processes"""
        processes = [
            ("Streamlit", self.streamlit_process),
            ("Tunnel", self.tunnel_process)
        ]
        
        for name, process in processes:
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except:
                    try:
                        process.kill()
                    except:
                        pass
                        
    def handle_interrupt(self, signum, frame):
        """Handle interrupt gracefully"""
        print("\n")
        self.print_status("Shutting down AI Video Creator...", "working")
        self.cleanup()
        self.print_status("Stopped successfully", "success")
        sys.exit(0)
        
    def run_app(self):
        """Run the complete standalone application"""
        signal.signal(signal.SIGINT, self.handle_interrupt)
        
        print("🎬 AI Video Creator - Standalone Launcher")
        print("="*45)
        
        try:
            # Step 1: Auto-detect environment
            if not self.auto_detect_environment():
                return False
                
            # Step 2: Find or download project files
            app_file = self.find_project_files()
            if not app_file:
                self.print_status("Cannot locate application files", "error")
                return False
                
            # Step 3: Ensure dependencies
            self.ensure_dependencies()
            
            # Step 4: Create minimal structure
            self.create_minimal_structure()
                
            # Step 5: Start Streamlit
            if not self.start_streamlit():
                self.print_status("Failed to start application", "error")
                return False
                
            # Step 6: Start tunnel (optional)
            tunnel_success = self.start_tunnel()
                
            # Step 7: Display access info
            self.display_access_info()
            
            # Step 8: Keep running
            self._run_main_loop()
            
            return True
            
        except Exception as e:
            self.print_status(f"Application error: {str(e)}", "error")
            return False
            
        finally:
            self.cleanup()
            
    def _run_main_loop(self):
        """Main loop with health monitoring"""
        try:
            while True:
                # Health check
                if self.streamlit_process and self.streamlit_process.poll() is not None:
                    self.print_status("Application stopped", "error")
                    break
                    
                time.sleep(10)
                
        except KeyboardInterrupt:
            pass

def main():
    """Standalone main function"""
    launcher = StandaloneLauncher()
    return launcher.run_app()

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n" + "="*45)
        print("❌ FAILED TO START")
        print("="*45)
        print("🔧 This launcher is completely standalone!")
        print("💡 It will auto-detect and setup everything needed")
        print("🚀 Try running again or check error messages above")
        print("="*45)
