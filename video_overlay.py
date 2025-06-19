#!/usr/bin/env python3
"""Video MC Overlay - Ghép MC vào video nền - Fixed version"""
import cv2, sys, subprocess, tempfile, os, shutil, json, argparse
from PIL import Image
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

class VideoOverlay:
    def __init__(self, mc_video, bg_video, position='Góc dưới phải', scale=0.25, output=None, quality='medium', bitrate=None):
        self.mc_video, self.bg_video = Path(mc_video), Path(bg_video)
        self.position, self.scale = position, scale
        self.output = Path(output or f"{self.mc_video.parent}/output_{self.mc_video.stem}_{self.bg_video.stem}.mp4")
        self.quality = {'low': 'ultrafast', 'medium': 'medium', 'high': 'slow'}.get(quality, quality)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.frames_dir = self.temp_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)

        # Initialize background remover
        try:
            from transparent_background import Remover
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.remover = Remover(mode='fast', device=self.device)
            self.has_remover = True
        except ImportError:
            print("⚠️ transparent-background không có, sử dụng fallback method")
            self.has_remover = False

        # Get video info
        self.mc_info = self._get_video_info(str(mc_video))
        self.bg_info = self._get_video_info(str(bg_video))
        
        # Extract video properties safely
        try:
            fps_str = self.mc_info.get('r_frame_rate', '30/1')
            if '/' in fps_str:
                num, den = fps_str.split('/')
                self.mc_fps = int(num) / max(int(den), 1)
            else:
                self.mc_fps = float(fps_str) if fps_str else 30.0
        except:
            self.mc_fps = 30.0
            
        self.bg_width = int(self.bg_info.get('width', 1280))
        self.bg_height = int(self.bg_info.get('height', 720))
        
        # Check for audio
        self.has_audio = {
            'mc': self._has_audio_stream(str(mc_video)),
            'bg': self._has_audio_stream(str(bg_video))
        }
        
        self.bitrate = bitrate or f"{max(1, int(self.bg_width * self.bg_height * self.mc_fps / 500000))}M"

    def __del__(self):
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    @lru_cache(maxsize=4)
    def _get_video_info(self, video_path):
        """Get video information using ffprobe"""
        cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of json "{video_path}"'
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                if data.get('streams'):
                    return data['streams'][0]
        except:
            pass
        return {'width': '1280', 'height': '720', 'r_frame_rate': '30/1'}

    def _has_audio_stream(self, video_path):
        """Check if video has audio stream"""
        cmd = f'ffprobe -v error -select_streams a -show_entries stream=codec_type -of json "{video_path}"'
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return len(data.get('streams', [])) > 0
        except:
            pass
        return False

    def process(self):
        """Main processing function"""
        print(f"🎬 Bắt đầu ghép video...")
        print(f"📊 MC: {self.mc_video.name} ({self.mc_fps:.1f}fps)")
        print(f"📊 BG: {self.bg_video.name} ({self.bg_width}x{self.bg_height})")
        
        # Process frames if using background remover
        if self.has_remover:
            print("🔄 Xử lý frames với AI background removal...")
            if not self._process_frames():
                print("⚠️ Lỗi xử lý frames, sử dụng fallback...")
                return self._fallback_overlay()
            
            return self._create_video_from_frames()
        else:
            print("🔄 Sử dụng overlay trực tiếp...")
            return self._direct_overlay()

    def _process_frames(self):
        """Process video frames to remove background"""
        cap = cv2.VideoCapture(str(self.mc_video))
        if not cap.isOpened():
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📹 Xử lý {total_frames} frames...")
        
        frame_count = 0
        success_count = 0
        
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # Remove background
                result_image = self.remover.process(pil_image, type="rgba")
                
                # Save frame
                frame_path = self.frames_dir / f"frame_{frame_count:06d}.png"
                result_image.save(frame_path, "PNG")
                success_count += 1
                
            except Exception as e:
                print(f"⚠️ Lỗi xử lý frame {frame_count}: {e}")
                
            frame_count += 1
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"📊 Tiến độ: {progress:.1f}% ({frame_count}/{total_frames})")

        cap.release()
        print(f"✅ Đã xử lý {success_count}/{frame_count} frames")
        return success_count > 0

    def _create_video_from_frames(self):
        """Create video from processed frames"""
        # Map position names to coordinates
        positions = {
            "Góc trên trái": "10:10",
            "Góc trên phải": f"{self.bg_width}-w-10:10", 
            "Góc dưới trái": f"10:{self.bg_height}-h-10",
            "Góc dưới phải": f"{self.bg_width}-w-10:{self.bg_height}-h-10",
            "Chính giữa": f"({self.bg_width}-w)/2:({self.bg_height}-h)/2"
        }
        pos = positions.get(self.position, positions["Góc dưới phải"])

        # Build FFmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(self.mc_fps),
            '-i', str(self.frames_dir / "frame_%06d.png"),
            '-i', str(self.bg_video)
        ]

        # Video filter
        video_filter = f"[0:v]scale=iw*{self.scale}:ih*{self.scale}[overlay];[1:v][overlay]overlay={pos}"
        cmd.extend(['-filter_complex', video_filter])

        # Audio handling
        if self.has_audio['mc']:
            cmd.extend(['-i', str(self.mc_video), '-map', '2:a'])
        elif self.has_audio['bg']:
            cmd.extend(['-map', '1:a'])
        else:
            cmd.append('-an')

        # Video encoding settings
        cmd.extend([
            '-c:v', 'libx264',
            '-preset', self.quality,
            '-crf', '23',
            '-pix_fmt', 'yuv420p'
        ])

        # Audio encoding
        if self.has_audio['mc'] or self.has_audio['bg']:
            cmd.extend(['-c:a', 'aac', '-b:a', '192k'])

        cmd.append(str(self.output))

        print(f"🔧 Chạy FFmpeg command...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0 and self.output.exists() and self.output.stat().st_size > 0:
                size_mb = self.output.stat().st_size / (1024*1024)
                return f"✅ Tạo video thành công! ({size_mb:.2f} MB)", str(self.output)
            else:
                print(f"❌ FFmpeg error: {result.stderr}")
                return self._fallback_overlay()
                
        except subprocess.TimeoutExpired:
            print("⏰ FFmpeg timeout, thử fallback...")
            return self._fallback_overlay()
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            return self._fallback_overlay()

    def _direct_overlay(self):
        """Direct overlay without background removal"""
        positions = {
            "Góc trên trái": "10:10",
            "Góc trên phải": f"{self.bg_width}-overlay_w-10:10",
            "Góc dưới trái": f"10:{self.bg_height}-overlay_h-10", 
            "Góc dưới phải": f"{self.bg_width}-overlay_w-10:{self.bg_height}-overlay_h-10",
            "Chính giữa": f"({self.bg_width}-overlay_w)/2:({self.bg_height}-overlay_h)/2"
        }
        pos = positions.get(self.position, positions["Góc dưới phải"])

        # Simple overlay command
        cmd = [
            'ffmpeg', '-y',
            '-i', str(self.bg_video),
            '-i', str(self.mc_video),
            '-filter_complex', f"[1:v]scale=iw*{self.scale}:ih*{self.scale}[overlay];[0:v][overlay]overlay={pos}",
            '-c:v', 'libx264',
            '-preset', self.quality,
            '-crf', '23'
        ]

        # Audio handling
        if self.has_audio['mc']:
            cmd.extend(['-map', '1:a', '-c:a', 'aac'])
        elif self.has_audio['bg']:
            cmd.extend(['-map', '0:a', '-c:a', 'aac'])
        else:
            cmd.append('-an')

        cmd.append(str(self.output))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and self.output.exists():
                size_mb = self.output.stat().st_size / (1024*1024)
                return f"✅ Ghép video thành công! ({size_mb:.2f} MB)", str(self.output)
            else:
                return self._fallback_overlay()
                
        except Exception as e:
            print(f"❌ Direct overlay error: {e}")
            return self._fallback_overlay()

    def _fallback_overlay(self):
        """Ultimate fallback method"""
        print("🔄 Sử dụng phương pháp fallback cuối cùng...")
        
        # Very simple overlay
        cmd = f'''ffmpeg -y -i "{self.bg_video}" -i "{self.mc_video}" \
-filter_complex "[1:v]scale=iw*{self.scale}:ih*{self.scale}[v];[0:v][v]overlay=W-w-10:H-h-10" \
-c:v libx264 -preset ultrafast -crf 28 \
{'-map 1:a -c:a copy' if self.has_audio['mc'] else '-map 0:a -c:a copy' if self.has_audio['bg'] else '-an'} \
"{self.output}"'''

        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0 and self.output.exists() and self.output.stat().st_size > 0:
                size_mb = self.output.stat().st_size / (1024*1024)
                return f"✅ Tạo video thành công (fallback)! ({size_mb:.2f} MB)", str(self.output)
            else:
                return "❌ Không thể tạo video", None
                
        except Exception as e:
            return f"❌ Lỗi fallback: {str(e)}", None

def main():
    parser = argparse.ArgumentParser(description="Video MC Overlay - Ghép MC vào video nền")
    parser.add_argument('-m', '--mc-video', required=True, help='Video MC')
    parser.add_argument('-b', '--bg-video', required=True, help='Video nền')
    parser.add_argument('-o', '--output', help='File đầu ra')
    parser.add_argument('-p', '--position', default='Góc dưới phải', 
                       choices=["Góc trên trái", "Góc trên phải", "Góc dưới trái", "Góc dưới phải", "Chính giữa"])
    parser.add_argument('-s', '--scale', type=float, default=0.25, help='Tỷ lệ (0.1-0.5)')
    parser.add_argument('-q', '--quality', default='medium', 
                       choices=['low', 'medium', 'high', 'ultrafast', 'slow', 'veryslow'])
    parser.add_argument('-r', '--bitrate', help='Bitrate (VD: 4M)')

    args = parser.parse_args()
    
    # Validate inputs
    if not (0.1 <= args.scale <= 0.5):
        print("❌ Tỷ lệ scale phải từ 0.1 đến 0.5")
        return 1
        
    if not Path(args.mc_video).exists():
        print(f"❌ Không tìm thấy file MC: {args.mc_video}")
        return 1
        
    if not Path(args.bg_video).exists():
        print(f"❌ Không tìm thấy file nền: {args.bg_video}")
        return 1

    # Check FFmpeg
    try:
        subprocess.run("ffmpeg -version", shell=True, capture_output=True, check=True)
    except:
        print("❌ Không tìm thấy FFmpeg")
        return 1

    # Create overlay processor
    overlay = VideoOverlay(
        args.mc_video, args.bg_video, args.position, 
        args.scale, args.output, args.quality, args.bitrate
    )
    
    # Process video
    status, output = overlay.process()
    
    print(f"\n{'✅' if output else '❌'} {status}")
    if output:
        print(f"📁 File: {output}")
        
    return 0 if output else 1

if __name__ == "__main__":
    sys.exit(main())
