#!/usr/bin/env python3
"""Video MC Overlay - Ghép MC vào video nền với xử lý lỗi cải tiến"""

import cv2
import sys
import subprocess
import tempfile
import os
import shutil
import json
import argparse
from PIL import Image
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np

class VideoOverlay:
    def __init__(self, mc_video, bg_video, position='Góc dưới phải', scale=0.25, output=None, quality='medium', bitrate=None):
        self.mc_video = Path(mc_video)
        self.bg_video = Path(bg_video)
        self.position = position
        self.scale = scale
        self.output = Path(output or f"{self.mc_video.parent}/output_{self.mc_video.stem}_{self.bg_video.stem}.mp4")
        self.quality = {'low': 'ultrafast', 'medium': 'medium', 'high': 'slow'}.get(quality, quality)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.frames_dir = self.temp_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        
        # Kiểm tra file input tồn tại
        if not self.mc_video.exists():
            raise FileNotFoundError(f"MC video không tồn tại: {self.mc_video}")
        if not self.bg_video.exists():
            raise FileNotFoundError(f"Background video không tồn tại: {self.bg_video}")
        
        # Khởi tạo transparent background remover
        try:
            from transparent_background import Remover
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.remover = Remover(mode='fast', device=self.device)
            self.use_ai_bg_removal = True
        except ImportError:
            print("⚠️ transparent-background không có, sử dụng chế độ fallback")
            self.use_ai_bg_removal = False
        
        # Lấy thông tin video
        self.mc_info = self._get_video_info(str(mc_video))
        self.bg_info = self._get_video_info(str(bg_video))
        
        # Xử lý thông tin video an toàn
        self.mc_fps = self._extract_fps(self.mc_info.get('r_frame_rate', '30/1'))
        self.bg_width = int(self.bg_info.get('width', 1280))
        self.bg_height = int(self.bg_info.get('height', 720))
        
        # Kiểm tra audio streams
        self.has_audio = {
            'mc': self._has_audio_stream(str(mc_video)),
            'bg': self._has_audio_stream(str(bg_video))
        }
        
        self.bitrate = bitrate or f"{max(1, int(self.bg_width * self.bg_height * self.mc_fps / 500000))}M"

    def _extract_fps(self, fps_str):
        """Trích xuất FPS từ string một cách an toàn"""
        try:
            if '/' in fps_str:
                nums = fps_str.split('/')
                return float(nums[0]) / float(nums[1]) if len(nums) == 2 and float(nums[1]) != 0 else 30.0
            return float(fps_str)
        except:
            return 30.0

    def _has_audio_stream(self, video_path):
        """Kiểm tra xem video có audio stream không"""
        try:
            cmd = f'ffprobe -v error -select_streams a -show_entries stream=codec_type -of json "{video_path}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return len(data.get('streams', [])) > 0
        except:
            pass
        return False

    def __del__(self):
        """Cleanup temp directory"""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    @lru_cache(maxsize=4)
    def _get_video_info(self, video_path):
        """Lấy thông tin video với cache"""
        cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of json "{video_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            try:
                data = json.loads(result.stdout)
                streams = data.get('streams', [{}])
                return streams[0] if streams else {}
            except:
                pass
        return {}

    def process(self):
        """Xử lý video overlay với nhiều phương án fallback"""
        
        # Kiểm tra xem MC là ảnh hay video
        mc_ext = self.mc_video.suffix.lower()
        is_image = mc_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        if is_image:
            return self._process_image_mc()
        else:
            return self._process_video_mc()
    
    def _process_image_mc(self):
        """Xử lý khi MC input là ảnh"""
        try:
            # Đọc ảnh MC
            mc_img = cv2.imread(str(self.mc_video))
            if mc_img is None:
                return "Lỗi: Không thể đọc ảnh MC", None
            
            # Xử lý background removal nếu có thể
            if self.use_ai_bg_removal:
                try:
                    # Chuyển BGR sang RGB cho PIL
                    mc_rgb = cv2.cvtColor(mc_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(mc_rgb)
                    
                    # Remove background
                    mc_transparent = self.remover.process(pil_img, type="rgba")
                    
                    # Chuyển về numpy array
                    mc_array = np.array(mc_transparent)
                    
                    # Tạo frame với alpha channel
                    overlay_frame = mc_array
                    
                except Exception as e:
                    print(f"⚠️ AI background removal failed, using original: {e}")
                    # Fallback: sử dụng ảnh gốc
                    overlay_frame = cv2.cvtColor(mc_img, cv2.COLOR_BGR2RGBA)
            else:
                # Không có AI removal, sử dụng ảnh gốc
                overlay_frame = cv2.cvtColor(mc_img, cv2.COLOR_BGR2RGBA)
            
            # Lưu frame overlay
            overlay_path = self.temp_dir / "overlay.png"
            if overlay_frame.shape[2] == 4:  # RGBA
                Image.fromarray(overlay_frame).save(overlay_path)
            else:  # RGB
                cv2.imwrite(str(overlay_path), overlay_frame)
            
            # Tạo video overlay với FFmpeg
            return self._create_overlay_video(overlay_path, is_static=True)
            
        except Exception as e:
            return f"Lỗi xử lý ảnh MC: {str(e)}", None
    
    def _process_video_mc(self):
        """Xử lý khi MC input là video"""
        if not self._process_frames():
            return "Lỗi: Không thể xử lý frames video", None
        
        # Sử dụng frames đã xử lý
        return self._create_overlay_video(self.frames_dir / "f_%06d.png", is_static=False)
    
    def _process_frames(self):
        """Xử lý từng frame của video MC"""
        cap = cv2.VideoCapture(str(self.mc_video))
        if not cap.isOpened():
            return False

        frame_count = 0
        success_count = 0
        
        with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 2, 8)) as executor:
            futures = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                future = executor.submit(self._process_frame, frame, frame_count)
                futures.append(future)
                frame_count += 1
            
            cap.release()
            
            # Đợi tất cả frames xử lý xong
            for future in futures:
                if future.result():
                    success_count += 1
        
        print(f"✅ Đã xử lý {success_count}/{frame_count} frames")
        return success_count > 0

    def _process_frame(self, frame, idx):
        """Xử lý một frame với background removal"""
        try:
            if frame is None:
                return False
            
            if self.use_ai_bg_removal:
                # Chuyển BGR sang RGB cho PIL
                rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
                
                # Remove background
                result = self.remover.process(pil_img, type="rgba")
                result.save(self.frames_dir / f'f_{idx:06d}.png')
            else:
                # Fallback: lưu frame gốc
                cv2.imwrite(str(self.frames_dir / f'f_{idx:06d}.png'), frame)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Lỗi xử lý frame {idx}: {e}")
            return False

    def _create_overlay_video(self, overlay_input, is_static=False):
        """Tạo video overlay với FFmpeg"""
        
        # Xác định vị trí overlay
        positions = {
            "Góc trên trái": "0:0",
            "Góc trên phải": f"{self.bg_width}-overlay_w:0", 
            "Góc dưới trái": f"0:{self.bg_height}-overlay_h",
            "Góc dưới phải": f"{self.bg_width}-overlay_w:{self.bg_height}-overlay_h",
            "Chính giữa": f"({self.bg_width}-overlay_w)/2:({self.bg_height}-overlay_h)/2"
        }
        pos = positions.get(self.position, positions["Góc dưới phải"])
        
        # Tạo filter complex
        if is_static:
            # Cho ảnh tĩnh
            filter_complex = f"[1]scale=iw*{self.scale}:-1[overlay];[0:v][overlay]overlay={pos}:enable='between(t,0,{self._get_bg_duration()})'"
        else:
            # Cho video frames
            filter_complex = f"[1]scale=iw*{self.scale}:-1[overlay];[0:v][overlay]overlay={pos}"
        
        # Lệnh FFmpeg chính
        if is_static:
            cmd = [
                'ffmpeg', '-y',
                '-i', str(self.bg_video),  # Background video
                '-loop', '1', '-i', str(overlay_input),  # Static overlay image
                '-filter_complex', filter_complex,
                '-shortest',  # End when shortest input ends
                '-c:v', 'libx264', '-preset', self.quality, '-crf', '23',
                '-pix_fmt', 'yuv420p'
            ]
        else:
            cmd = [
                'ffmpeg', '-y',
                '-i', str(self.bg_video),  # Background video
                '-framerate', str(self.mc_fps), '-i', str(overlay_input),  # Overlay frames
                '-filter_complex', filter_complex,
                '-c:v', 'libx264', '-preset', self.quality, '-crf', '23',
                '-pix_fmt', 'yuv420p'
            ]
        
        # Xử lý audio
        if self.has_audio['mc'] and not is_static:
            # Ưu tiên audio từ MC nếu có
            cmd.extend(['-i', str(self.mc_video), '-map', '2:a', '-c:a', 'aac', '-b:a', '192k'])
        elif self.has_audio['bg']:
            # Sử dụng audio từ background
            cmd.extend(['-map', '0:a', '-c:a', 'aac', '-b:a', '192k'])
        else:
            # Không có audio
            cmd.append('-an')
        
        # Thêm output path
        cmd.append(str(self.output))
        
        # Chạy lệnh
        print(f"🔄 Đang tạo video overlay...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and self.output.exists() and self.output.stat().st_size > 0:
            size_mb = self.output.stat().st_size / (1024 * 1024)
            return f"✅ Tạo video thành công! ({size_mb:.2f} MB)", str(self.output)
        
        # Fallback method
        print("⚠️ Phương pháp chính thất bại, thử phương án dự phòng...")
        return self._fallback_overlay()
    
    def _fallback_overlay(self):
        """Phương án dự phòng đơn giản hơn"""
        try:
            # Lệnh đơn giản hơn
            cmd = [
                'ffmpeg', '-y',
                '-i', str(self.bg_video),
                '-i', str(self.mc_video),
                '-filter_complex', f'[1:v]scale=iw*{self.scale}:-1[overlay];[0:v][overlay]overlay=W-w-10:H-h-10',
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28'
            ]
            
            # Audio handling
            if self.has_audio['bg']:
                cmd.extend(['-map', '0:a', '-c:a', 'copy'])
            else:
                cmd.append('-an')
            
            cmd.append(str(self.output))
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and self.output.exists() and self.output.stat().st_size > 0:
                size_mb = self.output.stat().st_size / (1024 * 1024)
                return f"✅ Tạo video thành công (fallback)! ({size_mb:.2f} MB)", str(self.output)
            else:
                return f"❌ Lỗi tạo video: {result.stderr}", None
                
        except Exception as e:
            return f"❌ Lỗi fallback: {str(e)}", None
    
    def _get_bg_duration(self):
        """Lấy độ dài của background video"""
        try:
            cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{self.bg_video}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return 10.0  # Default 10 seconds

def main():
    """Hàm chính để chạy từ command line"""
    parser = argparse.ArgumentParser(description="Video MC Overlay - Ghép MC vào video nền")
    parser.add_argument('-m', '--mc-video', required=True, help='Video/Ảnh MC')
    parser.add_argument('-b', '--bg-video', required=True, help='Video nền')
    parser.add_argument('-o', '--output', help='File đầu ra')
    parser.add_argument('-p', '--position', default='Góc dưới phải', 
                       choices=["Góc trên trái", "Góc trên phải", "Góc dưới trái", "Góc dưới phải", "Chính giữa"])
    parser.add_argument('-s', '--scale', type=float, default=0.25, help='Tỷ lệ (0.1-0.5)')
    parser.add_argument('-q', '--quality', default='medium', 
                       choices=['low', 'medium', 'high', 'ultrafast', 'slow', 'veryslow'])
    parser.add_argument('-r', '--bitrate', help='Bitrate (VD: 4M)')

    args = parser.parse_args()
    
    # Kiểm tra tỷ lệ scale
    if not (0.1 <= args.scale <= 0.5):
        print("❌ Tỷ lệ scale phải nằm trong khoảng 0.1-0.5")
        return 1

    # Kiểm tra FFmpeg
    if subprocess.run("ffmpeg -version", shell=True, capture_output=True).returncode != 0:
        print("❌ FFmpeg không được tìm thấy. Vui lòng cài đặt FFmpeg.")
        return 1

    try:
        overlay = VideoOverlay(
            args.mc_video, args.bg_video, args.position, 
            args.scale, args.output, args.quality, args.bitrate
        )
        status, output = overlay.process()

        print(f"\n{'✅' if output else '❌'} {status}")
        if output:
            print(f"📁 File: {output}")
        return 0 if output else 1
        
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
