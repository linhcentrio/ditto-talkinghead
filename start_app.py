#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Ditto Talking Head - Application Launcher (Enhanced)
Launch Streamlit app with better error handling and fallback options
"""

import os
import sys
import subprocess
import time
import json
import threading
import signal
from pathlib import Path
from datetime import datetime

class DittoLauncher:
    def __init__(self):
        self.streamlit_process = None
        self.tunnel_process = None
        self.public_url = None
        self.config = None
        
    def print_status(self, message, status="info"):
        """Print formatted status message"""
        icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "working": "🔄"}
        print(f"{icons.get(status, 'ℹ️')} {message}")
        
    def check_setup_status(self):
        """Enhanced setup status checking with fallback options"""
        self.print_status("Checking setup status...", "working")
        
        # Check for config file
        config_file = Path('.ditto_config.json')
        
        # Primary check: config file exists
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    self.config = json.load(f)
                self.print_status("Configuration found", "success")
                return True
            except Exception as e:
                self.print_status(f"Config file corrupted: {e}", "warning")
                
        # Secondary check: look for project structure
        project_indicators = [
            "ditto-talkinghead/run_streamlit.py",
            "run_streamlit.py",
            "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
        ]
        
        found_indicators = [p for p in project_indicators if Path(p).exists()]
        
        if found_indicators:
            self.print_status(f"Found project files: {len(found_indicators)}/3", "warning")
            self.print_status("Attempting to create fallback configuration...", "working")
            return self._create_fallback_config()
        
        # No setup found at all
        self.print_status("Setup not completed. Please run STEP 1 first.", "error")
        self._show_setup_instructions()
        return False
        
    def _create_fallback_config(self):
        """Create fallback configuration based on available files"""
        try:
            # Detect GPU capability
            gpu_capability = 7  # Default for T4
            data_root = "./checkpoints/ditto_trt"
            
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_capability = torch.cuda.get_device_capability()[0]
                    if gpu_capability >= 8:
                        data_root = "./checkpoints/ditto_trt_Ampere_Plus"
            except:
                pass
                
            # Create basic config
            self.config = {
                'data_root': data_root,
                'gpu_capability': gpu_capability,
                'cfg_pkl': './checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl',
                'setup_time': datetime.now().isoformat(),
                'setup_type': 'fallback'
            }
            
            # Save config
            with open('.ditto_config.json', 'w') as f:
                json.dump(self.config, f, indent=2)
                
            # Set environment variables
            os.environ['DITTO_DATA_ROOT'] = data_root
            os.environ['DITTO_GPU_CAPABILITY'] = str(gpu_capability)
            
            self.print_status("Fallback configuration created", "success")
            return True
            
        except Exception as e:
            self.print_status(f"Failed to create fallback config: {e}", "error")
            return False
            
    def _show_setup_instructions(self):
        """Show setup instructions"""
        print("\n" + "="*50)
        print("📋 SETUP REQUIRED")
        print("="*50)
        print("Please follow these steps:")
        print()
        print("1️⃣ Go back to the STEP 1 cell above")
        print("2️⃣ Click the ▶️ button to run setup")
        print("3️⃣ Wait for 'SETUP COMPLETED!' message")
        print("4️⃣ Then run this cell again")
        print()
        print("Or run this command manually:")
        print("!python run_colab.py")
        print("="*50)
        
    def verify_environment(self):
        """Verify the environment with enhanced checking"""
        self.print_status("Verifying environment...", "working")
        
        # Navigate to project directory
        if not Path("run_streamlit.py").exists():
            if Path("ditto-talkinghead/run_streamlit.py").exists():
                os.chdir("ditto-talkinghead")
                self.print_status("Changed to project directory", "info")
            else:
                self.print_status("Project files not found", "error")
                return False
                
        # Check critical files
        critical_files = ["run_streamlit.py"]
        missing_critical = [f for f in critical_files if not Path(f).exists()]
        
        if missing_critical:
            self.print_status(f"Missing critical files: {missing_critical}", "error")
            return False
            
        # Check optional files (warn but don't fail)
        optional_files = [
            "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl",
            "checkpoints/ditto_trt",
            "checkpoints/ditto_trt_Ampere_Plus"
        ]
        
        missing_optional = [f for f in optional_files if not Path(f).exists()]
        if missing_optional:
            self.print_status(f"Missing optional files: {len(missing_optional)} items", "warning")
            self.print_status("Some AI features may be limited", "warning")
            
        self.print_status("Environment verification completed", "success")
        return True
        
    def start_streamlit(self):
        """Start Streamlit server with enhanced configuration"""
        self.print_status("Starting Streamlit server...", "working")
        
        # Set comprehensive Streamlit environment
        env = os.environ.copy()
        env.update({
            'STREAMLIT_SERVER_FILE_WATCHER_TYPE': 'none',
            'STREAMLIT_SERVER_HEADLESS': 'true',
            'STREAMLIT_SERVER_PORT': '8501',
            'STREAMLIT_SERVER_ADDRESS': '0.0.0.0',
            'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false',
            'STREAMLIT_SERVER_ENABLE_CORS': 'false',
            'STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION': 'false'
        })
        
        # Add config-based environment variables
        if self.config:
            env.update({
                'DITTO_DATA_ROOT': self.config.get('data_root', './checkpoints/ditto_trt'),
                'DITTO_GPU_CAPABILITY': str(self.config.get('gpu_capability', 7))
            })
        
        try:
            self.streamlit_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "run_streamlit.py",
                "--server.port=8501",
                "--server.address=0.0.0.0", 
                "--server.headless=true",
                "--browser.gatherUsageStats=false",
                "--server.fileWatcherType=none"
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for server with progress indicator
            self.print_status("Initializing Streamlit (15s)...", "working")
            for i in range(15):
                if self.streamlit_process.poll() is not None:
                    break
                time.sleep(1)
                if i % 3 == 0:
                    print(f"    ⏳ {15-i}s remaining...")
            
            # Verify server
            if self.streamlit_process.poll() is None:
                if self._verify_server():
                    return True
                else:
                    self.print_status("Server started but not responding properly", "warning")
                    return True  # Continue anyway
            else:
                self.print_status("Streamlit failed to start", "error")
                return False
                
        except Exception as e:
            self.print_status(f"Failed to start Streamlit: {e}", "error")
            return False
            
    def _verify_server(self):
        """Verify Streamlit server is responding"""
        try:
            import requests
            for attempt in range(3):
                try:
                    response = requests.get("http://localhost:8501", timeout=5)
                    if response.status_code == 200:
                        self.print_status("Streamlit server verified", "success")
                        return True
                except:
                    pass
                time.sleep(2)
        except ImportError:
            # requests not available, skip verification
            pass
            
        return False
        
    def start_tunnel(self):
        """Start tunnel with multiple service options"""
        self.print_status("Creating public tunnel...", "working")
        
        # Try ngrok first (if available)
        if self._try_ngrok():
            return True
            
        # Fallback to cloudflared
        if self._try_cloudflared():
            return True
            
        self.print_status("Failed to create public tunnel", "error")
        self.print_status("App will be available locally only", "warning")
        return False
        
    def _try_ngrok(self):
        """Try to use ngrok tunnel"""
        try:
            from pyngrok import ngrok
            self.print_status("Using ngrok tunnel...", "working")
            
            # Create tunnel
            public_url = ngrok.connect(8501, "http")
            self.public_url = str(public_url)
            
            self.print_status("Ngrok tunnel created", "success")
            return True
            
        except Exception as e:
            self.print_status(f"Ngrok failed: {str(e)[:50]}...", "warning")
            return False
            
    def _try_cloudflared(self):
        """Try to use cloudflared tunnel"""
        try:
            self.print_status("Using CloudFlare tunnel...", "working")
            
            self.tunnel_process = subprocess.Popen([
                "cloudflared", "tunnel", "--url", "http://localhost:8501"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Parse tunnel URL
            start_time = time.time()
            while time.time() - start_time < 30:
                if self.tunnel_process.poll() is not None:
                    break
                    
                line = self.tunnel_process.stderr.readline()
                if line and "trycloudflare.com" in line:
                    import re
                    url_match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                    if url_match:
                        self.public_url = url_match.group(0)
                        self.print_status("CloudFlare tunnel created", "success")
                        return True
                        
                time.sleep(0.5)
                
            self.print_status("CloudFlare tunnel timeout", "warning")
            return False
            
        except Exception as e:
            self.print_status(f"CloudFlare failed: {str(e)[:50]}...", "warning")
            return False
            
    def display_access_info(self):
        """Display comprehensive access information"""
        print("\n" + "="*60)
        print("🎭 DITTO TALKING HEAD - READY!")
        print("="*60)
        
        if self.public_url:
            print(f"\n🔗 **PUBLIC ACCESS:**")
            print(f"   {self.public_url}")
            print(f"\n📱 Click the link above to access your app")
            print(f"🌍 Share this URL with others if needed")
        else:
            print(f"\n🔗 **LOCAL ACCESS ONLY:**")
            print(f"   http://localhost:8501")
            print(f"📱 Available within this Colab session only")
            
        print(f"\n🎬 **HOW TO USE:**")
        print(f"   1. Upload MC image/video + background video")
        print(f"   2. Add audio file OR enter text for TTS")  
        print(f"   3. Adjust position, size, and quality")
        print(f"   4. Click 'Create Video' and wait for result")
        
        if self.config and self.config.get('setup_type') == 'fallback':
            print(f"\n⚠️  **NOTE:** Using fallback configuration")
            print(f"   Some AI features may be limited")
            print(f"   For full features, run complete setup")
            
        print(f"\n🛑 **TO STOP:** Interrupt this cell (■ button)")
        print("="*60)
        
    def cleanup(self):
        """Enhanced cleanup"""
        processes = [
            ("Streamlit", self.streamlit_process),
            ("Tunnel", self.tunnel_process)
        ]
        
        for name, process in processes:
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                    self.print_status(f"{name} stopped", "success")
                except:
                    try:
                        process.kill()
                        self.print_status(f"{name} force stopped", "warning")
                    except:
                        pass
                        
    def handle_interrupt(self, signum, frame):
        """Handle keyboard interrupt gracefully"""
        print("\n")
        self.print_status("Stopping application...", "working")
        self.cleanup()
        self.print_status("Application stopped", "success")
        sys.exit(0)
        
    def run_app(self):
        """Run the complete application with enhanced error handling"""
        # Setup signal handler
        signal.signal(signal.SIGINT, self.handle_interrupt)
        
        print("🎭 " + "="*40)
        print("   DITTO TALKING HEAD LAUNCHER")
        print("="*42)
        
        try:
            # Check setup status with fallback options
            if not self.check_setup_status():
                return False
                
            # Verify environment
            if not self.verify_environment():
                self.print_status("Environment issues detected, but continuing...", "warning")
                
            # Start Streamlit
            if not self.start_streamlit():
                self.print_status("Failed to start Streamlit server", "error")
                return False
                
            # Start tunnel (optional)
            tunnel_success = self.start_tunnel()
            if not tunnel_success:
                self.print_status("Continuing without public tunnel", "warning")
                
            # Display access information
            self.display_access_info()
            
            # Keep running with health monitoring
            self._run_main_loop()
            
            return True
            
        except Exception as e:
            self.print_status(f"Application error: {e}", "error")
            return False
            
        finally:
            self.cleanup()
            
    def _run_main_loop(self):
        """Main application loop with health monitoring"""
        try:
            while True:
                # Check Streamlit process
                if self.streamlit_process and self.streamlit_process.poll() is not None:
                    self.print_status("Streamlit process stopped", "error")
                    break
                    
                # Check tunnel process (if using cloudflared)
                if (self.tunnel_process and 
                    self.tunnel_process.poll() is not None and 
                    "cloudflared" in str(self.tunnel_process.args)):
                    self.print_status("Tunnel disconnected, attempting restart...", "warning")
                    if not self._try_cloudflared():
                        self.print_status("Tunnel restart failed", "warning")
                        
                time.sleep(10)  # Check every 10 seconds
                
        except KeyboardInterrupt:
            self.print_status("Stopping application...", "working")

def main():
    """Enhanced main launcher function"""
    launcher = DittoLauncher()
    return launcher.run_app()

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n" + "="*50)
        print("❌ APPLICATION FAILED TO START")
        print("="*50)
        print("🔧 Troubleshooting steps:")
        print("1. Make sure you ran STEP 1 setup first")
        print("2. Check if setup completed successfully")
        print("3. Try restarting runtime and run setup again")
        print("4. If issues persist, check the error messages above")
        print("="*50)
        sys.exit(1)
