#!/usr/bin/env python3
"""Video MC Overlay - Ghép MC vào video nền"""
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
        
        try: 
            from transparent_background import Remover
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.remover = Remover(mode='fast', device=self.device)
        except ImportError: 
            print("Cài đặt: pip install transparent-background"); sys.exit(1)
        
        # Lấy thông tin video
        self.mc_info = self._get_video_info(str(mc_video))[0]
        self.bg_info = self._get_video_info(str(bg_video))[0]
        fps_str = self.mc_info.get('r_frame_rate', '30/1')
        self.mc_fps = 30 if not fps_str else (int(p[0]) / int(p[1] or 1) if len(p := fps_str.split('/')) == 2 else 30)
        self.bg_width = int(self.bg_info.get('width', 1280))
        self.bg_height = int(self.bg_info.get('height', 720))
        self.has_audio = {
            'mc': bool(self._get_video_info(str(mc_video), 'audio')), 
            'bg': bool(self._get_video_info(str(bg_video), 'audio'))
        }
        self.bitrate = bitrate or f"{max(1, int(self.bg_width * self.bg_height * self.mc_fps / 500000))}M"
    
    def __del__(self): 
        self.temp_dir.exists() and shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @lru_cache(maxsize=4)
    def _get_video_info(self, video_path, info_type='video'):
        stream, entries = ("v:0", "width,height,r_frame_rate") if info_type == "video" else ("a", "codec_type")
        cmd = f'ffprobe -v error -select_streams {stream} -show_entries stream={entries} -of json "{video_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout).get('streams', [{}])
        return [{}]

    def process(self):
        if not self._process_frames(): 
            return "Lỗi: Không thể xử lý frames", None
        
        # Xác định vị trí overlay
        positions = {
            "Góc trên trái": "0:0", "Góc trên phải": f"{self.bg_width}-w:0", 
            "Góc dưới trái": f"0:{self.bg_height}-h", "Góc dưới phải": f"{self.bg_width}-w:{self.bg_height}-h", 
            "Chính giữa": f"({self.bg_width}-w)/2:({self.bg_height}-h)/2"
        }
        pos = positions.get(self.position, positions["Góc dưới phải"])
        
        # FFmpeg command
        cmd = ['ffmpeg', '-y', '-framerate', f"{self.mc_fps}", '-i', f"{self.frames_dir}/f_%06d.png", '-i', str(self.bg_video)]
        
        # Audio mapping
        if self.has_audio['mc']:
            cmd.extend(['-i', str(self.mc_video), '-map', '2:a', '-c:a', 'aac', '-b:a', '192k'])
        elif self.has_audio['bg']:
            cmd.extend(['-map', '1:a', '-c:a', 'aac', '-b:a', '192k'])
        else:
            cmd.append('-an')
            
        # Video settings
        cmd.extend([
            '-filter_complex', f"[0]scale=iw*{self.scale}:-1[fg];[1:v][fg]overlay={pos}", 
            '-c:v', 'libx264', '-preset', self.quality, '-crf', '23', 
            '-pix_fmt', 'yuv420p', '-b:v', self.bitrate, str(self.output)
        ])
        
        # Run main command
        subprocess.run(cmd, capture_output=True)
        if self.output.exists() and (size := self.output.stat().st_size) > 0:
            return f"Xử lý hoàn tất! ({size / 1_048_576:.2f} MB)", str(self.output)
        
        # Backup plan
        backup_cmd = (
            f'ffmpeg -y -i "{self.bg_video}" -framerate {self.mc_fps} -i "{self.frames_dir}/f_%06d.png" '
            f'-filter_complex "[1]scale=iw*{self.scale}:-1[fg];[0:v][fg]overlay={pos}" -c:v libx264 -preset {self.quality} '
            f'{"-map 0:a -c:a copy " if self.has_audio["bg"] else "-an "}"{self.output}"'
        )
        subprocess.run(backup_cmd, shell=True)
        
        if self.output.exists() and self.output.stat().st_size > 0:
            size = self.output.stat().st_size / 1_048_576
            return f"Xử lý hoàn tất (phương án dự phòng)! ({size:.2f} MB)", str(self.output)
        return "Lỗi: Không thể tạo video", None
    
    def _process_frames(self):
        cap = cv2.VideoCapture(str(self.mc_video))
        if not cap.isOpened(): 
            return False
            
        with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 2, 8)) as executor:
            futures, idx = [], 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                futures.append(executor.submit(self._process_frame, frame, idx))
                idx += 1
                
            cap.release()
            return any(f.result() for f in futures)
    
    def _process_frame(self, frame, idx):
        try:
            if frame is None:
                return False
            rgb_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.remover.process(rgb_img, type="rgba").save(self.frames_dir / f'f_{idx:06d}.png')
            return True
        except:
            return False

def main():
    parser = argparse.ArgumentParser(description="Video MC Overlay - Ghép MC vào video nền")
    parser.add_argument('-m', '--mc-video', required=True, help='Video MC')
    parser.add_argument('-b', '--bg-video', required=True, help='Video nền')
    parser.add_argument('-o', '--output', help='File đầu ra')
    parser.add_argument('-p', '--position', default='Góc dưới phải', 
                      choices=["Góc trên trái", "Góc trên phải", "Góc dưới trái", "Góc dưới phải", "Chính giữa"])
    parser.add_argument('-s', '--scale', type=float, default=0.25, help='Tỷ lệ (0.1-0.5)')
    parser.add_argument('-q', '--quality', default='medium', choices=['low', 'medium', 'high', 'ultrafast', 'slow', 'veryslow'])
    parser.add_argument('-r', '--bitrate', help='Bitrate (VD: 4M)')
    
    args = parser.parse_args()
    if not (0.1 <= args.scale <= 0.5) or subprocess.run("ffmpeg -version", shell=True, capture_output=True).returncode != 0:
        print("Lỗi: Tỷ lệ scale không hợp lệ hoặc không tìm thấy FFmpeg.")
        return 1
    
    overlay = VideoOverlay(args.mc_video, args.bg_video, args.position, args.scale, args.output, args.quality, args.bitrate)
    status, output = overlay.process()
    
    print(f"\n{'✅' if output else '❌'} {status}")
    if output:
        print(f"📁 File: {output}")
    return 0 if output else 1

if __name__ == "__main__": 
    sys.exit(main())
