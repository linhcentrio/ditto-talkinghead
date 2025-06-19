#!/usr/bin/env python3
"""
🚀 Ditto Talking Head - App Launcher
Simple launcher script for Google Colab
"""

import subprocess
import time
import sys
from pathlib import Path

def start_application():
    """Start the Ditto Talking Head application"""
    
    print("🎭 " + "="*40)
    print("   DITTO TALKING HEAD LAUNCHER")  
    print("="*42)
    
    # Check if setup was completed
    if not Path("ditto-talkinghead/start_app.sh").exists():
        print("❌ Setup not completed. Please run the setup cell first.")
        return False
    
    # Change to project directory
    try:
        import os
        os.chdir("ditto-talkinghead")
        print("📁 Changed to project directory")
    except:
        print("❌ Could not access project directory")
        return False
    
    # Start the application
    print("🚀 Starting application...")
    print("⏳ This will take 10-15 seconds...")
    print("🔗 CloudFlared will show your public URL below:")
    print("="*42)
    
    try:
        # Run the startup script
        subprocess.run(["./start_app.sh"], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Application stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Failed to start application: {e}")
        return False
    
    return True

if __name__ == "__main__":
    start_application()
