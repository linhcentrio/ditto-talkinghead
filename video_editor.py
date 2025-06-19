#!/usr/bin/env python3
"""Video Editor - Module tạo audio và phụ đề chuyên nghiệp"""

import os
import uuid
import tempfile
import argparse
import re
import time
from pathlib import Path
from typing import Tuple, Optional, List, Union, Any, Dict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from moviepy import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
from moviepy.video.tools.subtitles import SubtitlesClip

class VideoEditor:
    """Class tích hợp tính năng xử lý audio và phụ đề chuyên nghiệp"""

    def __init__(self, output_dir: Union[str, Path] = "./output"):
        """Khởi tạo VideoEditor với thư mục đầu ra"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def create_sample_audio_if_missing(self, audio_path: Union[str, Path]) -> bool:
        """Tạo audio mẫu nếu không có audio input"""
        try:
            audio_path = Path(audio_path)
            
            # Tạo audio im lặng 5 giây
            cmd = [
                "ffmpeg", "-y", "-f", "lavfi", 
                "-i", "anullsrc=r=44100:cl=mono", 
                "-t", "5", "-q:a", "0", 
                "-map", "0", str(audio_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0 and audio_path.exists()
            
        except Exception as e:
            print(f"Lỗi tạo audio mẫu: {e}")
            return False
    
    def fix_audio_format(self, input_audio: Union[str, Path], output_audio: Union[str, Path]) -> bool:
        """Sửa format audio để đảm bảo tương thích"""
        try:
            cmd = [
                "ffmpeg", "-y", "-i", str(input_audio),
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                str(output_audio)
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0
            
        except Exception as e:
            print(f"Lỗi fix audio format: {e}")
            return False

    def generate_audio_from_text(self, text: str, output_path: Union[str, Path], 
                               srt_path: Optional[Union[str, Path]] = None, 
                               service: str = "edge", 
                               voice: str = "vi-VN-NamMinhNeural", 
                               speed: float = 1.2) -> Tuple[bool, str]:
        """Tạo audio từ văn bản và tuỳ chọn tạo file phụ đề"""
        output_path, srt_path = Path(output_path), Path(srt_path) if srt_path else None
        
        try:
            match service:
                case "edge":
                    return self._edge_tts(text, voice, output_path, srt_path)
                case "openai":
                    return self._openai_tts(text, voice, output_path, srt_path, speed)
                case _:
                    return False, f"Dịch vụ TTS không hỗ trợ: {service}"
        except Exception as e:
            return False, f"Lỗi khi tạo audio: {str(e)}"

    def _edge_tts(self, text: str, voice: str, output_path: Path, srt_path: Optional[Path]) -> Tuple[bool, str]:
        """Tạo audio và phụ đề sử dụng Edge TTS"""
        try:
            import edge_tts
            
            communicate = edge_tts.Communicate(text, voice)
            submaker = edge_tts.SubMaker()
            
            with open(output_path, "wb") as file:
                for chunk in communicate.stream_sync():
                    match chunk["type"]:
                        case "audio":
                            file.write(chunk["data"])
                        case "WordBoundary":
                            submaker.feed(chunk)
            
            # Nếu cần tạo file SRT
            if srt_path:
                with open(srt_path, "w", encoding="utf-8") as file:
                    file.write(submaker.get_srt())
            
            return True, ""
        except ImportError:
            return False, "Thư viện edge-tts chưa được cài đặt. Cài đặt với: pip install edge-tts"
        except Exception as e:
            return False, f"Lỗi Edge TTS: {str(e)}"

    def _openai_tts(self, text: str, voice: str, output_path: Path, 
                  srt_path: Optional[Path], speed: float = 1.2) -> Tuple[bool, str]:
        """Tạo audio sử dụng OpenAI TTS"""
        try:
            import openai
            from dotenv import load_dotenv
            
            load_dotenv()
            
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            if not OPENAI_API_KEY:
                return False, "OPENAI_API_KEY không được định nghĩa trong biến môi trường hoặc file .env"
            
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
            # Gọi API TTS
            response = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                speed=speed
            )
            
            # Lưu âm thanh
            response.write_to_file(str(output_path))
            
            # Tạo phụ đề nếu cần
            if srt_path:
                success, error = self.generate_srt_from_audio(output_path, srt_path)
                if not success:
                    return False, f"Tạo audio thành công nhưng tạo SRT thất bại: {error}"
            
            return True, ""
        except ImportError:
            return False, "Thư viện openai hoặc python-dotenv chưa được cài đặt."
        except Exception as e:
            return False, f"Lỗi OpenAI TTS: {str(e)}"

    def generate_srt_from_audio(self, audio_path: Union[str, Path], 
                              output_path: Union[str, Path]) -> Tuple[bool, str]:
        """Tạo phụ đề SRT từ file audio sử dụng Whisper"""
        try:
            import openai
            from dotenv import load_dotenv
            
            load_dotenv()
            
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            if not OPENAI_API_KEY:
                return False, "OPENAI_API_KEY không được định nghĩa trong biến môi trường hoặc file .env"
            
            openai.api_key = OPENAI_API_KEY
            
            audio_path, output_path = Path(audio_path), Path(output_path)
            
            # Kiểm tra định dạng file
            if (audio_format := audio_path.suffix.lower()) not in ['.mp3', '.wav', '.m4a']:
                return False, f"Định dạng file không được hỗ trợ: {audio_format}"
            
            # Gọi Whisper API
            with open(audio_path, "rb") as audio:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    response_format="srt"
                )
            
            # Lưu phụ đề SRT
            with open(output_path, "w", encoding="utf-8") as srt_file:
                srt_file.write(transcript)
            
            return True, ""
        except ImportError:
            return False, "Thư viện openai hoặc python-dotenv chưa được cài đặt."
        except Exception as e:
            return False, f"Lỗi khi tạo phụ đề: {str(e)}"

    def apply_caption_style_01(self, video_path: Union[str, Path], srt_path: Union[str, Path], 
                             output_path: Union[str, Path], audio_path: Optional[Union[str, Path]] = None, 
                             fontsize: int = 48, position: str = "bottom") -> Tuple[bool, str]:
        """Áp dụng phụ đề kiểu style 01 (từng từ) vào video - Tối ưu căn giữa và vị trí"""
        try:
            from moviepy import VideoFileClip, CompositeVideoClip, AudioFileClip
            
            video_path, srt_path, output_path = Path(video_path), Path(srt_path), Path(output_path)
            audio_path = Path(audio_path) if audio_path else None
            
            font = "font/Roboto/Roboto-Bold.ttf"
            
            with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as temp_file:
                processed_srt_path = Path(temp_file.name)
            
            success, error = self._process_srt_for_style_01(srt_path, processed_srt_path)
            if not success:
                processed_srt_path.unlink(missing_ok=True)
                return False, error
            
            word_groups, timing_groups, start_times, error = self._parse_srt_to_word_groups(processed_srt_path)
            if error:
                processed_srt_path.unlink(missing_ok=True)
                return False, error
            
            video = VideoFileClip(str(video_path))
            text_clips = []
            
            # Tính vị trí Y dựa trên position
            video_height = video.h
            line_spacing = fontsize * 0.5
            
            # Sử dụng tỷ lệ thay vì công thức cứng
            position_factor = {
                "top": 0.15,       # 15% từ trên xuống 
                "center": 0.5,     # Giữa màn hình
                "bottom": 0.75     # 75% từ trên xuống
            }.get(position, 0.75)  # Mặc định là bottom
            
            start_y = video_height * position_factor
            
            start_i = 0
            
            for idx, texts in enumerate(word_groups):
                start_time, end_time = timing_groups[idx]
                
                for i, text in enumerate(texts):
                    y_pos = start_y + i * (fontsize + line_spacing)
                    words = text.split()
                    default_color = "#FCBB09" if len(words) != 4 and len(words) != 2 else "#FFFFFF" if len(words) == 4 else "#00DCBF"
                    
                    # Tính tổng độ rộng CHÍNH XÁC bao gồm khoảng cách giữa các từ
                    word_widths = [self._measure_text_width(word, font, fontsize) for word in words]
                    word_spacing = 10  # Khoảng cách giữa các từ
                    total_spacing = word_spacing * (len(words) - 1) if len(words) > 1 else 0
                    line_width = sum(word_widths) + total_spacing
                    
                    # Căn giữa chính xác hơn
                    x_pos = (video.w - line_width) / 2
                    
                    for word in words:
                        start_times_word = start_times[start_i]
                        word_clip = self._create_text_with_glow(
                            word, font, fontsize, "#000000", 1, default_color, True, default_color, 10
                        )
                        
                        word_clip = word_clip.with_position((x_pos, y_pos)).with_start(start_times_word).with_end(end_time)
                        text_clips.append(word_clip)
                        x_pos += self._measure_text_width(word, font, fontsize) + word_spacing
                        start_i += 1
            
            result = CompositeVideoClip([video] + text_clips)
            if audio_path and audio_path.exists():
                audio_clip = AudioFileClip(str(audio_path))
                result = result.with_audio(audio_clip)
            
            result.write_videofile(str(output_path), fps=24)
            
            processed_srt_path.unlink(missing_ok=True)
            
            return True, ""
        except ImportError as e:
            return False, f"Thiếu thư viện: {str(e)}. Cài đặt với: pip install moviepy pillow numpy"
        except Exception as e:
            if 'processed_srt_path' in locals():
                Path(processed_srt_path).unlink(missing_ok=True)
            return False, f"Lỗi khi tạo phụ đề style 01: {str(e)}"

    def apply_caption_style_02(self, video_path: Union[str, Path], srt_path: Union[str, Path], 
                             output_path: Union[str, Path], audio_path: Optional[Union[str, Path]] = None, 
                             fontsize: int = 75, position: str = "center", zoom_effect: bool = True, 
                             zoom_size: float = 0.01) -> Tuple[bool, str]:
        """Áp dụng phụ đề kiểu style 02 (gradient) vào video - Tối ưu vị trí và căn chỉnh"""
        try:
            import pysrt
            from math import sin, pi
            from moviepy import VideoFileClip, ImageClip, CompositeVideoClip, AudioFileClip
            
            video_path, srt_path, output_path = Path(video_path), Path(srt_path), Path(output_path)
            audio_path = Path(audio_path) if audio_path else None
            
            gradient_colors = [
                ("#8A2BE2", "#7FFF00"),  # Tím - Xanh lá
                ("#00DCBF", "#FFFFFF")   # Xanh ngọc - Trắng
            ]
            
            font_path = "font/Roboto/Roboto-Bold.ttf"
            
            with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as temp_file:
                processed_srt_path = Path(temp_file.name)
            
            success, error = self._process_srt_for_style_02(srt_path, processed_srt_path, [4, 4])
            if not success:
                processed_srt_path.unlink(missing_ok=True)
                return False, error
            
            video_clip = VideoFileClip(str(video_path))
            subtitle_clips = []
            
            VIDEO_W, VIDEO_H = video_clip.size
            
            # Tính vị trí dựa trên tỷ lệ màn hình thay vì giá trị cứng
            spacing_factor = fontsize / VIDEO_H * 1.5  # Tỷ lệ dựa trên fontsize và chiều cao video
            
            subs = pysrt.open(str(processed_srt_path), encoding='utf-8')
            
            for sub in subs:
                start_s = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000.0
                end_s = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000.0
                duration = end_s - start_s
                
                top_text, bottom_text = self._split_subtitle_in_half(sub.text)
                
                # Tính vị trí dựa trên position
                match position:
                    case "center":
                        # Giữa màn hình, hai dòng cách nhau spacing_factor * VIDEO_H
                        top_y = 0.5 - spacing_factor  # Vị trí tương đối
                        bottom_y = 0.5 + spacing_factor
                    case "top":
                        # Phần trên màn hình
                        top_y = 0.15  # 15% từ trên xuống
                        bottom_y = 0.15 + 2 * spacing_factor
                    case "bottom":
                        # Phần dưới màn hình
                        top_y = 0.85 - 2 * spacing_factor  # 85% từ trên xuống
                        bottom_y = 0.85
                
                # Chuyển thành vị trí tuyệt đối
                top_y_pos = int(VIDEO_H * top_y)
                bottom_y_pos = int(VIDEO_H * bottom_y)
                
                # Sử dụng vị trí chính xác
                top_text_image = self._create_gradient_text(
                    top_text, (VIDEO_W, VIDEO_H), font_path, fontsize, gradient_colors, 
                    {"x": VIDEO_W // 2, "y": top_y_pos}  # Truyền dict với x, y cụ thể
                )
                
                bot_text_image = self._create_gradient_text(
                    bottom_text, (VIDEO_W, VIDEO_H), font_path, fontsize, gradient_colors, 
                    {"x": VIDEO_W // 2, "y": bottom_y_pos}
                )
                
                top_clip = ImageClip(top_text_image, duration=duration).with_start(start_s)
                bot_clip = ImageClip(bot_text_image, duration=duration).with_start(start_s)
                
                if zoom_effect:
                    top_clip = top_clip.resize(lambda t: 1 + zoom_size * sin(2 * pi * t / duration))
                    bot_clip = bot_clip.resize(lambda t: 1 + zoom_size * sin(2 * pi * t / duration))
                
                subtitle_clips.extend([top_clip, bot_clip])
            
            final_clip = CompositeVideoClip([video_clip] + subtitle_clips, size=video_clip.size)
            if audio_path and audio_path.exists():
                audio_clip = AudioFileClip(str(audio_path))
                final_clip = final_clip.with_audio(audio_clip)
            
            final_clip.write_videofile(str(output_path), fps=24)
            
            processed_srt_path.unlink(missing_ok=True)
            
            return True, ""
        except ImportError as e:
            return False, f"Thiếu thư viện: {str(e)}. Cài đặt: pip install moviepy pillow numpy pysrt"
        except Exception as e:
            if 'processed_srt_path' in locals():
                Path(processed_srt_path).unlink(missing_ok=True)
            return False, f"Lỗi khi tạo phụ đề style 02: {str(e)}"

    def _process_srt_for_style_01(self, srt_path: Union[str, Path], 
                                output_path: Union[str, Path]) -> Tuple[bool, str]:
        """Xử lý file SRT cho style 01 - tách từng từ thành dòng riêng"""
        try:
            def parse_timestamp(timestamp):
                h, m, s = map(float, re.split('[:,]', timestamp.replace(',', '.')))
                return h * 3600 + m * 60 + s
            
            def format_timestamp(seconds):
                ms = int((seconds % 1) * 1000)
                seconds = int(seconds)
                h, remainder = divmod(seconds, 3600)
                m, s = divmod(remainder, 60)
                return f"{h:02}:{m:02}:{s:02},{ms:03}"
            
            with open(srt_path, 'r', encoding='utf-8') as file:
                srt_content = file.read()
            
            pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)\n\n', re.DOTALL)
            matches = pattern.findall(srt_content)
            
            new_srt = []
            index = 1
            
            # Giảm khoảng dừng giữa các từ
            pause_duration = 0.00
            
            for match in matches:
                _, start_time, end_time, text = match
                start_seconds = parse_timestamp(start_time)
                end_seconds = parse_timestamp(end_time)
                duration = end_seconds - start_seconds
                
                words = text.split()
                word_duration = (duration - pause_duration * (len(words) - 1)) / len(words) if words else 0
                
                for i, word in enumerate(words):
                    word_start = start_seconds + i * (word_duration + pause_duration)
                    word_end = word_start + word_duration
                    new_srt.append(f"{index}\n{format_timestamp(word_start)} --> {format_timestamp(word_end)}\n{word}\n")
                    index += 1
            
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write("\n".join(new_srt))
            
            return True, ""
        except Exception as e:
            return False, f"Lỗi khi xử lý SRT cho style 01: {str(e)}"

    def _process_srt_for_style_02(self, srt_path: Union[str, Path], 
                                output_path: Union[str, Path], 
                                group_sizes: List[int] = [4, 4]) -> Tuple[bool, str]:
        """Xử lý file SRT cho style 02 - tách thành các nhóm từ"""
        try:
            def time_to_seconds(time_str):
                h, m, s = time_str.split(':')
                s, ms = s.split(',')
                return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
            
            def seconds_to_time(seconds):
                from datetime import timedelta
                td = timedelta(seconds=seconds)
                return str(td).split('. ')[0].zfill(8) + ',' + str(int((seconds - int(seconds)) * 1000)).zfill(3)
            
            with open(srt_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
            
            result = []
            
            i = 0
            line_number = 1
            
            while i < len(lines):
                line = lines[i].strip()
                
                if re.match(r'^\d+$', line) and i + 2 < len(lines):
                    i += 1
                    time_line = lines[i].strip()
                    
                    if " --> " in time_line:
                        start_time, end_time = time_line.split(" --> ")
                        start_seconds = time_to_seconds(start_time)
                        end_seconds = time_to_seconds(end_time)
                        duration = end_seconds - start_seconds
                        
                        i += 1
                        content = lines[i].strip()
                        
                        words = content.split()
                        groups = []
                        word_index = 0
                        while word_index < len(words):
                            for size in group_sizes:
                                if word_index < len(words):
                                    group = words[word_index:word_index + size]
                                    groups.append(group)
                                    word_index += size
                        
                        num_groups = len(groups)
                        if num_groups > 0:
                            time_per_group = duration / num_groups
                            current_time = start_seconds
                            
                            for group in groups:
                                group_text = " ".join(group)
                                group_end_time = current_time + time_per_group
                                result.append(f"{line_number}\n{seconds_to_time(current_time)} --> {seconds_to_time(group_end_time)}\n{group_text}\n\n")
                                current_time = group_end_time
                                line_number += 1
                i += 1
            
            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.writelines(result)
            
            return True, ""
        except Exception as e:
            return False, f"Lỗi khi xử lý SRT cho style 02: {str(e)}"

    def _parse_srt_to_word_groups(self, srt_path: Union[str, Path]) -> Tuple[Optional[List], Optional[List], Optional[List], str]:
        """Phân tích SRT thành các nhóm từ với thời gian - Tối ưu cho phụ đề căn giữa"""
        try:
            def parse_srt_to_words_and_durations(srt_text):
                pattern = r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+)"
                matches = re.findall(pattern, srt_text)
                
                results = []
                start_times = []
                for start, end, word in matches:
                    start_sec = int(start[0:2]) * 3600 + int(start[3:5]) * 60 + float(start[6:].replace(',', '.'))
                    end_sec = int(end[0:2]) * 3600 + int(end[3:5]) * 60 + float(end[6:].replace(',', '.'))
                    duration = end_sec - start_sec
                    
                    results.append({"word": word, "start_time": start_sec, "duration": duration})
                    start_times.append(start_sec)
                
                return results, start_times
            
            def split_words_with_timing(parsed_data):
                words = [entry["word"] for entry in parsed_data]
                timings = [entry["start_time"] for entry in parsed_data]
                durations = [entry["duration"] for entry in parsed_data]
                
                result_groups = []
                timing_groups = []
                
                i = 0
                
                while i < len(words):
                    if len(result_groups) % 2 == 0:  # Nhóm chẵn: 6 từ
                        group_words = words[i:i+6]
                        end_idx = min(i+5, len(words)-1)
                        group_timings = (timings[i], timings[end_idx] + durations[end_idx])
                        i += 6
                    else:  # Nhóm lẻ: 3 từ
                        group_words = words[i:i+3]
                        end_idx = min(i+2, len(words)-1)
                        group_timings = (timings[i], timings[end_idx] + durations[end_idx])
                        i += 3
                    
                    if group_words:
                        if len(group_words) == 6:
                            result_groups.append([
                                " ".join(group_words[:4]),
                                " ".join(group_words[4:])
                            ])
                        else:
                            result_groups.append([" ".join(group_words)])
                        
                        timing_groups.append(group_timings)
                
                return result_groups, timing_groups
            
            with open(srt_path, 'r', encoding='utf-8') as file:
                srt_content = file.read()
            
            parsed_data, start_times = parse_srt_to_words_and_durations(srt_content)
            result_groups, timing_groups = split_words_with_timing(parsed_data)
            
            return result_groups, timing_groups, start_times, ""
        except Exception as e:
            return None, None, None, f"Lỗi khi phân tích SRT: {str(e)}"

    def _create_text_with_glow(self, text: str, font_path: str, fontsize: int, 
                             outline_color: str, outline_offset: int, text_color: str, 
                             glow: bool = True, glow_color: Union[str, Tuple[int, int, int]] = (255, 255, 0, 255), 
                             glow_radius: int = 8) -> Any:
        """Tạo text với hiệu ứng glow và viền - Tối ưu căn chỉnh và hiển thị"""
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageFilter
            from moviepy import ImageClip
            
            # Xử lý màu dạng tên thành mã hex
            if not outline_color.startswith("#"):
                outline_color = self._color_name_to_hex(outline_color)
            if not text_color.startswith("#"):
                text_color = self._color_name_to_hex(text_color)
            
            # Chuyển đổi màu sang RGB
            outline_color_rgb = self._hex_to_rgb(outline_color)
            text_color_rgb = self._hex_to_rgb(text_color)
            
            # Xử lý glow_color
            if isinstance(glow_color, str):
                if not glow_color.startswith("#"):
                    glow_color = self._color_name_to_hex(glow_color)
                glow_color = self._hex_to_rgb(glow_color) + (255,)
            
            # Tạo font
            try:
                font = ImageFont.truetype(font_path, fontsize)
            except:
                # Fallback font nếu không tìm thấy
                font = ImageFont.load_default()
                fontsize = int(fontsize * 0.8)
            
            # Tính kích thước text
            temp_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            text_bbox = temp_draw.textbbox((0, 0), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            
            # Tạo image với buffer cho effect
            padding = max(outline_offset, glow_radius) + fontsize // 2
            w = text_w + padding * 2  # Thêm padding vì text có thể bị cắt nếu quá sát cạnh
            h = text_h + padding * 2
            base_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(base_img)
            
            # Vẽ outline
            for dx in [-outline_offset, 0, outline_offset]:
                for dy in [-outline_offset, 0, outline_offset]:
                    if dx != 0 or dy != 0:
                        draw.text((padding + dx, padding + dy), text, font=font, fill=outline_color_rgb)
            
            # Vẽ text chính
            draw.text((padding, padding), text, font=font, fill=text_color_rgb)
            
            # Thêm hiệu ứng glow nếu cần
            if glow:
                glow_layer = base_img.copy()
                alpha = glow_layer.split()[-1]
                solid_color = Image.new("RGBA", glow_layer.size, glow_color)
                glow_layer = Image.composite(solid_color, Image.new("RGBA", glow_layer.size, (0, 0, 0, 0)), alpha)
                glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(glow_radius))
                base_img = Image.alpha_composite(glow_layer, base_img)
            
            return ImageClip(np.array(base_img))
        except Exception as e:
            raise Exception(f"Lỗi khi tạo text với glow: {str(e)}")

    def _create_gradient_text(self, text: str, size: Tuple[int, int], font_path: str, 
                            font_size: int, gradient_colors: List[Tuple[str, str]], 
                            text_position: Union[str, Dict[str, int], int] = "center", 
                            glow: bool = True, glow_radius: int = 50, 
                            stroke_width: int = 1, stroke_fill: str = "#000000") -> np.ndarray:
        """Tạo gradient text với hiệu ứng glow - Tối ưu căn chỉnh và vị trí"""
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageFilter
            import random
            
            img = Image.new("RGBA", size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Xử lý màu stroke
            if not stroke_fill.startswith("#"):
                stroke_fill = self._color_name_to_hex(stroke_fill)
            stroke_fill_rgb = self._hex_to_rgb(stroke_fill)
            
            # Tải font và xử lý fallback
            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                # Fallback nếu không tìm thấy font
                print(f"Không tìm thấy font {font_path}, sử dụng font mặc định")
                font = ImageFont.load_default()
                font_size = int(font_size * 0.8)  # Giảm kích thước để phù hợp với font mặc định
            
            # Đo kích thước text chính xác
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Xử lý vị trí linh hoạt hơn
            if isinstance(text_position, dict) and 'x' in text_position and 'y' in text_position:
                # Dict với x, y cụ thể
                x_pos = text_position['x'] - text_width // 2  # Vẫn căn giữa theo X
                y_pos = text_position['y'] - text_height // 2  # Căn giữa theo Y
                text_position = (x_pos, y_pos)
            elif text_position == "center":
                # Chính giữa màn hình
                text_position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
            else:
                # Giá trị số đơn lẻ - coi là vị trí Y, X vẫn căn giữa
                y_pos = int(text_position) if isinstance(text_position, (int, float)) else (size[1] - text_height) // 2
                text_position = ((size[0] - text_width) // 2, y_pos)
            
            # Chọn màu gradient ngẫu nhiên
            color1, color2 = random.choice(gradient_colors)
            color1 = self._hex_to_rgb(color1)
            color2 = self._hex_to_rgb(color2)
            
            # Tạo gradient
            gradient = Image.new("RGBA", size, (0, 0, 0, 0))
            for y in range(size[1]):
                r = int(color1[0] + (color2[0] - color1[0]) * y / size[1])
                g = int(color1[1] + (color2[1] - color1[1]) * y / size[1])
                b = int(color1[2] + (color2[2] - color1[2]) * y / size[1])
                ImageDraw.Draw(gradient).line([(0, y), (size[0], y)], fill=(r, g, b, 255))
            
            # Vẽ viền cho text
            for dx in range(-stroke_width, stroke_width+1):
                for dy in range(-stroke_width, stroke_width+1):
                    if dx != 0 or dy != 0:
                        draw.text((text_position[0] + dx, text_position[1] + dy), text, font=font, fill=stroke_fill_rgb)
            
            # Vẽ text chính - để alpha đúng cho gradient
            draw.text(text_position, text, font=font, fill="white")
            
            # Áp dụng gradient
            gradient_array = np.array(gradient)
            text_array = np.array(img)
            gradient_array[:, :, 3] = text_array[:, :, 3]
            gradient_img = Image.fromarray(gradient_array)
            
            # Thêm hiệu ứng glow nếu cần
            if glow:
                glow_effect = gradient_img.filter(ImageFilter.GaussianBlur(glow_radius))
                result = Image.alpha_composite(glow_effect, gradient_img)
            else:
                result = gradient_img
            
            return np.array(result, dtype=np.uint8)
        except Exception as e:
            raise Exception(f"Lỗi khi tạo gradient text: {str(e)}")

    def _measure_text_width(self, text: str, font_path: str, fontsize: int) -> int:
        """Đo độ rộng của text với font đã cho - Tối ưu độ chính xác"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Tải font an toàn với fallback
            try:
                font = ImageFont.truetype(font_path, fontsize)
            except:
                font = ImageFont.load_default()
                fontsize = int(fontsize * 0.8)
                
            # Sử dụng textbbox cho kết quả chính xác hơn
            # Đặc biệt quan trọng cho các ngôn ngữ có dấu như tiếng Việt
            text_bbox = draw.textbbox((0, 0), text, font=font)
            width = text_bbox[2] - text_bbox[0]
            
            # Thêm padding nhỏ để tránh cắt text khi hiển thị
            return width + int(fontsize * 0.05)
        except Exception as e:
            print(f"Lỗi đo text: {e}")
            return len(text) * fontsize // 2  # Ước tính đơn giản

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Chuyển đổi màu HEX sang RGB"""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) != 6:
            raise ValueError(f"Mã màu không hợp lệ: {hex_color}")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _color_name_to_hex(self, color_name: str) -> str:
        """Chuyển đổi tên màu sang mã hex"""
        colors = {
            "black": "#000000",
            "white": "#FFFFFF",
            "red": "#FF0000",
            "green": "#00FF00",
            "blue": "#0000FF",
            "yellow": "#FFFF00",
            "cyan": "#00FFFF",
            "magenta": "#FF00FF",
            "silver": "#C0C0C0",
            "gray": "#808080",
            "maroon": "#800000",
            "olive": "#808000",
            "purple": "#800080",
            "teal": "#008080",
            "navy": "#000080"
        }
        return colors.get(color_name.lower(), "#FFFFFF")  # Trả về trắng nếu không tìm thấy

    def _split_subtitle_in_half(self, content: str) -> Tuple[str, str]:
        """Chia nội dung phụ đề thành 2 phần - Cải tiến giữ nguyên ý nghĩa"""
        try:
            words = content.split()
            half = len(words) // 2
            
            # Nếu chuỗi quá ngắn, không chia
            if len(words) <= 3:
                return content, ""
                
            return " ".join(words[:half]), " ".join(words[half:])
        except:
            return content, ""

    def preview_caption_positions(self, video_path: Union[str, Path], fontsize: int = 48, 
                               output_path: Optional[Union[str, Path]] = None) -> str:
        """Tạo video xem trước vị trí phụ đề để kiểm tra căn chỉnh"""
        try:
            from moviepy import VideoFileClip, TextClip, CompositeVideoClip
            import tempfile
            
            video = VideoFileClip(str(video_path))
            VIDEO_W, VIDEO_H = video.size
            
            # Tạo các phụ đề mẫu cho các vị trí
            clips = [video]
            durations = min(video.duration, 10)  # Giới hạn 10 giây
            
            # Hiển thị grid tỷ lệ
            for i in range(1, 10):
                ratio = i/10
                y_pos = int(VIDEO_H * ratio)
                text = f"{ratio*100:.0f}%"
                
                clip = TextClip(text, fontsize=fontsize//2, color='white')
                clip = clip.with_position((50, y_pos)).with_duration(durations)
                clips.append(clip)
                
                # Vẽ đường ngang
                line = TextClip("—" * 100, fontsize=fontsize//3, color='gray')
                line = line.with_position((VIDEO_W//2, y_pos)).with_duration(durations)
                clips.append(line)
                
            # Demo các vị trí phụ đề
            positions = ["top", "center", "bottom"]
            colors = ['yellow', 'cyan', 'lime']
            
            for i, pos in enumerate(positions):
                # Style 01
                y_factor = 0.15 if pos == "top" else 0.5 if pos == "center" else 0.75
                y_pos = int(VIDEO_H * y_factor)
                
                text = f"Style 01 - {pos.upper()}"
                clip = TextClip(text, fontsize=fontsize, color=colors[i])
                clip = clip.with_position(('center', y_pos)).with_duration(durations)
                clips.append(clip)
                
                # Style 02 - 2 dòng
                spacing = fontsize / VIDEO_H * 1.5
                if pos == "center":
                    top_y = 0.5 - spacing
                    bottom_y = 0.5 + spacing
                elif pos == "top":
                    top_y = 0.15
                    bottom_y = 0.15 + 2*spacing
                else:  # bottom
                    top_y = 0.85 - 2*spacing
                    bottom_y = 0.85
                    
                top_clip = TextClip(f"Style 02 {pos.upper()} - Dòng 1", fontsize=fontsize, color=colors[i])
                bottom_clip = TextClip(f"Style 02 {pos.upper()} - Dòng 2", fontsize=fontsize, color=colors[i])
                
                top_clip = top_clip.with_position(('center', int(VIDEO_H * top_y))).with_duration(durations)
                bottom_clip = bottom_clip.with_position(('center', int(VIDEO_H * bottom_y))).with_duration(durations)
                
                clips.extend([top_clip, bottom_clip])
            
            # Tạo video cuối
            result = CompositeVideoClip(clips)
            
            # Lưu preview
            if not output_path:
                output_path = f"./output/caption_preview_{int(time.time())}.mp4"
            
            output_path = Path(output_path)
            result.write_videofile(str(output_path), fps=24)
            
            return f"Đã tạo preview tại: {str(output_path)}"
        except Exception as e:
            return f"Lỗi preview: {str(e)}"

def main():
    """Hàm chính để chạy từ command line"""
    parser = argparse.ArgumentParser(description="Video Editor - Công cụ TTS và phụ đề chuyên nghiệp")
    
    # Tạo nhóm lệnh
    subparsers = parser.add_subparsers(dest="command", help="Lệnh")
    
    # Lệnh tạo audio
    audio_parser = subparsers.add_parser("audio", help="Tạo audio từ văn bản")
    audio_parser.add_argument("--text", required=True, help="Văn bản đầu vào")
    audio_parser.add_argument("--output", required=True, help="Đường dẫn file audio đầu ra")
    audio_parser.add_argument("--srt", help="Đường dẫn file SRT đầu ra (tùy chọn)")
    audio_parser.add_argument("--service", choices=["edge", "openai"], default="edge", help="Dịch vụ TTS")
    audio_parser.add_argument("--voice", default="vi-VN-NamMinhNeural", help="Giọng đọc")
    audio_parser.add_argument("--speed", type=float, default=1.2, help="Tốc độ đọc (OpenAI TTS)")
    
    # Lệnh tạo phụ đề
    srt_parser = subparsers.add_parser("srt", help="Tạo phụ đề SRT từ audio")
    srt_parser.add_argument("--audio", required=True, help="Đường dẫn file audio")
    srt_parser.add_argument("--output", required=True, help="Đường dẫn file SRT đầu ra")
    
    # Lệnh tạo phụ đề style 01
    style01_parser = subparsers.add_parser("style01", help="Tạo phụ đề style 01 (từng từ)")
    style01_parser.add_argument("--video", required=True, help="Đường dẫn video đầu vào")
    style01_parser.add_argument("--srt", required=True, help="Đường dẫn file SRT")
    style01_parser.add_argument("--output", required=True, help="Đường dẫn video đầu ra")
    style01_parser.add_argument("--audio", help="Đường dẫn file audio (tùy chọn)")
    style01_parser.add_argument("--fontsize", type=int, default=48, help="Kích thước font")
    style01_parser.add_argument("--position", choices=["top", "center", "bottom"], default="bottom", help="Vị trí phụ đề")
    
    # Lệnh tạo phụ đề style 02
    style02_parser = subparsers.add_parser("style02", help="Tạo phụ đề style 02 (gradient)")
    style02_parser.add_argument("--video", required=True, help="Đường dẫn video đầu vào")
    style02_parser.add_argument("--srt", required=True, help="Đường dẫn file SRT")
    style02_parser.add_argument("--output", required=True, help="Đường dẫn video đầu ra")
    style02_parser.add_argument("--audio", help="Đường dẫn file audio (tùy chọn)")
    style02_parser.add_argument("--fontsize", type=int, default=75, help="Kích thước font")
    style02_parser.add_argument("--position", choices=["center", "top", "bottom"], default="center", help="Vị trí phụ đề")
    style02_parser.add_argument("--zoom", action="store_true", help="Bật hiệu ứng zoom")
    style02_parser.add_argument("--zoom-size", type=float, default=0.01, help="Độ lớn hiệu ứng zoom")
    
    # Lệnh preview vị trí phụ đề
    preview_parser = subparsers.add_parser("preview", help="Tạo video xem trước vị trí phụ đề")
    preview_parser.add_argument("--video", required=True, help="Đường dẫn video đầu vào")
    preview_parser.add_argument("--output", help="Đường dẫn video đầu ra")
    preview_parser.add_argument("--fontsize", type=int, default=48, help="Kích thước font xem trước")
    
    args = parser.parse_args()
    
    # Khởi tạo VideoEditor
    editor = VideoEditor()
    
    # Xử lý lệnh
    match args.command:
        case "audio":
            success, error = editor.generate_audio_from_text(
                args.text, args.output, args.srt, args.service, args.voice, args.speed
            )
            print(f"✅ Tạo audio thành công: {args.output}" if success else f"❌ Lỗi: {error}")
        
        case "srt":
            success, error = editor.generate_srt_from_audio(args.audio, args.output)
            print(f"✅ Tạo phụ đề thành công: {args.output}" if success else f"❌ Lỗi: {error}")
        
        case "style01":
            success, error = editor.apply_caption_style_01(
                args.video, args.srt, args.output, args.audio, args.fontsize, args.position
            )
            print(f"✅ Tạo video với phụ đề style 01 thành công: {args.output}" if success else f"❌ Lỗi: {error}")
        
        case "style02":
            success, error = editor.apply_caption_style_02(
                args.video, args.srt, args.output, args.audio, args.fontsize, args.position, args.zoom, args.zoom_size
            )
            print(f"✅ Tạo video với phụ đề style 02 thành công: {args.output}" if success else f"❌ Lỗi: {error}")
            
        case "preview":
            result = editor.preview_caption_positions(args.video, args.fontsize, args.output)
            print(result)
            
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
