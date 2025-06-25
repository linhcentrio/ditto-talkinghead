#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Ditto Talking Head - Application Launcher
Launch Streamlit app with public tunnel access
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
        
    def print_status(self, message, status="info"):
        """Print formatted status message"""
        icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "working": "🔄"}
        print(f"{icons.get(status, 'ℹ️')} {message}")
        
    def load_config(self):
        """Load environment configuration"""
        config_file = Path('.ditto_config.json')
        
        if not config_file.exists():
            self.print_status("Configuration not found. Please run setup first.", "error")
            return False
            
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # Set environment variables
            os.environ['DITTO_DATA_ROOT'] = config.get('data_root', './checkpoints/ditto_trt')
            os.environ['DITTO_GPU_CAPABILITY'] = str(config.get('gpu_capability', 6))
            
            self.print_status(f"Configuration loaded (GPU: {config.get('gpu_capability')})", "success")
            return True
            
        except Exception as e:
            self.print_status(f"Failed to load configuration: {e}", "error")
            return False
            
    def verify_environment(self):
        """Verify the environment is ready"""
        self.print_status("Verifying environment...", "working")
        
        # Check if we're in the right directory
        if not Path("run_streamlit.py").exists():
            if Path("ditto-talkinghead/run_streamlit.py").exists():
                os.chdir("ditto-talkinghead")
                self.print_status("Changed to project directory", "info")
            else:
                self.print_status("Project files not found. Please run setup first.", "error")
                return False
                
        # Check key files
        required_files = [
            "run_streamlit.py",
            "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
        ]
        
        missing = [f for f in required_files if not Path(f).exists()]
        if missing:
            self.print_status(f"Missing files: {missing}", "warning")
            self.print_status("Some features may not work properly", "warning")
            
        self.print_status("Environment verification completed", "success")
        return True
        
    def start_streamlit(self):
        """Start Streamlit server"""
        self.print_status("Starting Streamlit server...", "working")
        
        # Set Streamlit environment variables
        env = os.environ.copy()
        env.update({
            'STREAMLIT_SERVER_FILE_WATCHER_TYPE': 'none',
            'STREAMLIT_SERVER_HEADLESS': 'true',
            'STREAMLIT_SERVER_PORT': '8501',
            'STREAMLIT_SERVER_ADDRESS': '0.0.0.0',
            'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false'
        })
        
        try:
            self.streamlit_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "run_streamlit.py",
                "--server.port=8501",
                "--server.address=0.0.0.0", 
                "--server.headless=true",
                "--browser.gatherUsageStats=false"
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for server to start
            self.print_status("Waiting for server to initialize...", "working")
            time.sleep(15)
            
            # Check if server is running
            if self.streamlit_process.poll() is None:
                self.verify_server()
                return True
            else:
                self.print_status("Streamlit failed to start", "error")
                return False
                
        except Exception as e:
            self.print_status(f"Failed to start Streamlit: {e}", "error")
            return False
            
    def verify_server(self):
        """Verify Streamlit server is responding"""
        import requests
        
        for attempt in range(5):
            try:
                response = requests.get("http://localhost:8501", timeout=5)
                if response.status_code == 200:
                    self.print_status("Streamlit server is ready", "success")
                    return True
            except:
                pass
                
            if attempt < 4:
                self.print_status(f"Server check attempt {attempt + 1}/5...", "working")
                time.sleep(3)
                
        self.print_status("Server verification failed, but continuing...", "warning")
        return True
        
    def start_tunnel(self):
        """Start CloudFlared tunnel"""
        self.print_status("Creating secure tunnel...", "working")
        
        try:
            # Start cloudflared tunnel
            self.tunnel_process = subprocess.Popen([
                "cloudflared", "tunnel", "--url", "http://localhost:8501"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Parse tunnel URL from output
            start_time = time.time()
            while time.time() - start_time < 30:  # 30 second timeout
                if self.tunnel_process.poll() is not None:
                    break
                    
                # Read stderr for tunnel URL
                line = self.tunnel_process.stderr.readline()
                if line and "trycloudflare.com" in line:
                    # Extract URL
                    import re
                    url_match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                    if url_match:
                        self.public_url = url_match.group(0)
                        break
                        
                time.sleep(0.5)
                
            if self.public_url:
                self.print_status("Tunnel created successfully", "success")
                return True
            else:
                self.print_status("Failed to get tunnel URL", "error")
                return False
                
        except Exception as e:
            self.print_status(f"Tunnel creation failed: {e}", "error")
            return False
            
    def display_access_info(self):
        """Display access information"""
        print("\n" + "="*60)
        print("🎭 DITTO TALKING HEAD - READY!")
        print("="*60)
        
        if self.public_url:
            print(f"\n🔗 Access your app at:")
            print(f"   {self.public_url}")
            print(f"\n📱 Click the link above to open the application")
        else:
            print(f"\n🔗 Local access (Colab only):")
            print(f"   http://localhost:8501")
            
        print(f"\n💡 Tips:")
        print(f"   • The app is fully loaded and ready to use")
        print(f"   • Upload your MC image/video and background")
        print(f"   • Choose text-to-speech or upload audio")  
        print(f"   • Adjust settings and create your video")
        
        if self.public_url:
            print(f"\n⚠️  Note: The URL is temporary and will change if restarted")
            
        print(f"\n🛑 To stop the app, interrupt this cell (■ button)")
        print("="*60)
        
    def cleanup(self):
        """Clean up processes"""
        if self.streamlit_process:
            try:
                self.streamlit_process.terminate()
                self.streamlit_process.wait(timeout=5)
            except:
                self.streamlit_process.kill()
                
        if self.tunnel_process:
            try:
                self.tunnel_process.terminate()
                self.tunnel_process.wait(timeout=5)
            except:
                self.tunnel_process.kill()
                
    def handle_interrupt(self, signum, frame):
        """Handle keyboard interrupt"""
        self.print_status("Shutting down...", "working")
        self.cleanup()
        sys.exit(0)
        
    def run_app(self):
        """Run the complete application"""
        # Setup signal handler
        signal.signal(signal.SIGINT, self.handle_interrupt)
        
        print("🎭 " + "="*40)
        print("   DITTO TALKING HEAD LAUNCHER")
        print("="*42)
        
        try:
            # Load configuration
            if not self.load_config():
                return False
                
            # Verify environment
            if not self.verify_environment():
                return False
                
            # Start Streamlit
            if not self.start_streamlit():
                return False
                
            # Start tunnel
            if not self.start_tunnel():
                self.print_status("Continuing without public tunnel", "warning")
                
            # Display access information
            self.display_access_info()
            
            # Keep running
            try:
                while True:
                    # Check if processes are still running
                    if self.streamlit_process and self.streamlit_process.poll() is not None:
                        self.print_status("Streamlit process stopped", "error")
                        break
                        
                    if self.tunnel_process and self.tunnel_process.poll() is not None:
                        self.print_status("Tunnel process stopped, restarting...", "warning")
                        if not self.start_tunnel():
                            self.print_status("Failed to restart tunnel", "warning")
                            
                    time.sleep(5)
                    
            except KeyboardInterrupt:
                self.print_status("Stopping application...", "working")
                
            return True
            
        except Exception as e:
            self.print_status(f"Application error: {e}", "error")
            return False
            
        finally:
            self.cleanup()

def main():
    """Main launcher function"""
    launcher = DittoLauncher()
    return launcher.run_app()

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Failed to start application")
        print("💡 Try running the setup cell again if you encounter issues")
        sys.exit(1)
