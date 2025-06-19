#!/usr/bin/env python3
"""Streamlit UI tối ưu cho việc tạo video MC với nền và audio thoại - Phiên bản tối ưu"""

import streamlit as st
import subprocess
import tempfile
import shutil
import os
import queue
import threading
import time
import re
import traceback
import asyncio
import sys
import numpy as np
import pickle
import json
import librosa
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import cv2
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from video_editor import VideoEditor

# Nhận API keys từ môi trường
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

if OPENAI_API_KEY is None or PEXELS_API_KEY is None:
    raise ValueError("OPENAI_API_KEY và PEXELS_API_KEY chưa được thiết lập.")

# Tắt chế độ theo dõi file của Streamlit để tránh lỗi segmentation fault
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

# Thêm thư viện OpenAI mới
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

# === Định nghĩa các bước quy trình ===
WORKFLOW_STEPS = {
    "prepare_files": "Chuẩn bị files",
    "tts_generation": "Tạo âm thanh từ văn bản",
    "subtitle_generation": "Tạo phụ đề từ audio",
    "talking_head_generation": "Tạo video khuôn mặt nói",
    "video_overlay": "Ghép video MC và nền",
    "caption_application": "Thêm phụ đề vào video"
}

# === Khởi tạo OpenAI API client ===
openai_client = AsyncOpenAI()

# === Định nghĩa thông tin mô tả giọng nói ===
VOICE_DESCRIPTIONS = {
    "Ash": "Giọng nam trưởng thành, hơi trầm, phù hợp cho phim tài liệu",
    "Ballad": "Giọng nữ mềm mại, ấm áp, phù hợp cho nội dung tư vấn",
    "Coral": "Giọng nữ trẻ, rõ ràng, tự tin, phù hợp cho nội dung giáo dục",
    "Echo": "Giọng nam trẻ, năng động, phù hợp cho quảng cáo",
    "Fable": "Giọng nam uy tín, phù hợp cho thông báo chính thức",
    "Onyx": "Giọng nam trầm, sang trọng, phù hợp cho thuyết trình",
    "Nova": "Giọng nữ chuyên nghiệp, phù hợp cho tin tức",
    "Sage": "Giọng nữ từng trải, ấm áp, phù hợp cho podcast",
    "Shimmer": "Giọng nữ tươi sáng, năng động, phù hợp cho giải trí",
    "Verse": "Giọng nam tự nhiên, cân bằng, phù hợp cho đa dạng nội dung"
}

# === Hàm xác thực tham số khẩu hình ===
def validate_mouth_params(vad_alpha=1.0, exp_components=None, exp_scale=1.0, 
                          pose_scale=1.0, delta_exp_enabled=False, delta_exp_value=0.0):
    """Xác thực các tham số khẩu hình để đảm bảo chúng trong phạm vi an toàn"""
    validated = {}
    
    # Xác thực vad_alpha (giữ trong khoảng 0.0-1.0)
    validated['vad_alpha'] = max(0.0, min(1.0, float(vad_alpha)))
    
    # Xác thực exp_components (đảm bảo là list hợp lệ)
    if exp_components and isinstance(exp_components, list):
        validated['exp_components'] = [str(comp) for comp in exp_components if comp in ["exp", "pitch", "yaw", "roll", "t"]]
    else:
        validated['exp_components'] = None
    
    # Xác thực exp_scale và pose_scale (giữ trong khoảng 0.5-1.5)
    validated['exp_scale'] = max(0.5, min(1.5, float(exp_scale)))
    validated['pose_scale'] = max(0.5, min(1.5, float(pose_scale)))
    
    # Xác thực delta_exp_enabled và delta_exp_value
    validated['delta_exp_enabled'] = bool(delta_exp_enabled)
    validated['delta_exp_value'] = max(-0.2, min(0.2, float(delta_exp_value)))
    
    return validated

# === Khởi tạo session state ===
def init_session_state():
    """Khởi tạo toàn bộ session state cần thiết"""
    defaults = {
        'processing': False,
        'complete': False,
        'output_file': None,
        'history': [],
        'process_start_time': None,
        'auto_scale': True,
        'auto_fontsize': True,
        'logs': [],
    }

    # Khởi tạo workflow_steps riêng để đảm bảo nó được tạo đúng
    if 'workflow_steps' not in st.session_state:
        st.session_state['workflow_steps'] = {k: True for k in WORKFLOW_STEPS}

    # Khởi tạo các biến khác
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# === Các hàm tiện ích ===
@lru_cache(maxsize=32)
def get_video_resolution(video_path: str) -> Tuple[int, int]:
    """Lấy độ phân giải của video với cache"""
    try:
        with contextmanager(lambda: cv2.VideoCapture(str(video_path)))() as cap:
            return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080) if cap.isOpened() else (1920, 1080)
    except Exception:
        return 1920, 1080

def calculate_auto_fontsize(video_width: int, video_height: int) -> int:
    """Tính toán font size phù hợp dựa trên độ phân giải video"""
    return max(24, min(min(video_width * 24 // 1280, video_height * 24 // 720), 72))

def calculate_auto_scale(mc_path: Union[str, Any], bg_width: int, bg_height: int) -> float:
    """Tính toán tỉ lệ scale phù hợp cho MC"""
    try:
        mc_width, mc_height = 0, 0

        # Xử lý các loại đầu vào khác nhau với walrus operator
        if hasattr(mc_path, 'getbuffer'):  # UploadedFile
            suffix = Path(mc_path.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                temp.write(mc_path.getbuffer())
                temp_path = temp.name

            try:
                match suffix:
                    case ext if ext in ['.jpg', '.jpeg', '.png']:
                        if img := cv2.imread(temp_path):
                            mc_width, mc_height = img.shape[1], img.shape[0]
                    case _:  # Xử lý video
                        if cap := cv2.VideoCapture(temp_path):
                            if cap.isOpened():
                                mc_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                mc_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            cap.release()
            finally:
                os.unlink(temp_path)
        else:  # Path string
            mc_path_str = str(mc_path)
            suffix = Path(mc_path_str).suffix.lower()

            match suffix:
                case ext if ext in ['.jpg', '.jpeg', '.png']:
                    if img := cv2.imread(mc_path_str):
                        mc_width, mc_height = img.shape[1], img.shape[0]
                case _:  # Xử lý video
                    if cap := cv2.VideoCapture(mc_path_str):
                        if cap.isOpened():
                            mc_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            mc_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()

        if not mc_width or not mc_height:
            return 0.25

        # Tính toán tỉ lệ tối ưu
        width_scale = bg_width / mc_width / 3
        height_scale = bg_height / mc_height / 1.5

        return min(round(min(width_scale, height_scale), 2), 0.5)
    except Exception:
        return 0.25

def update_history_from_folder():
    """Cập nhật lịch sử từ thư mục output sử dụng pathlib và comprehension"""
    if not (output_folder := Path("./output")).exists():
        return

    # Lấy danh sách đường dẫn hiện có
    existing_paths = {item.get('path', '') for item in st.session_state.history}

    # Tìm các file mới
    new_files = [
        {
            'path': str(file),
            'created': datetime.fromtimestamp(file.stat().st_mtime),
            'size': file.stat().st_size / (1024*1024)
        }
        for file in output_folder.glob("final_mc_*.mp4")
        if file.exists() and str(file) not in existing_paths
    ]

    # Thêm vào lịch sử nếu có file mới
    if new_files:
        st.session_state.history.extend(new_files)

def create_empty_srt(srt_path: str):
    """Tạo file SRT trống khi bỏ qua bước tạo phụ đề"""
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write("""1
00:00:00,000 --> 00:00:05,000
Video MC Creator

2
00:00:05,000 --> 00:00:10,000
Tạo video MC với nền và audio thoại """)

def use_sample_audio(audio_path: str) -> str:
    """Sử dụng audio mẫu khi bỏ qua bước tạo audio"""
    # Tìm file audio mẫu
    if sample_paths := list(Path("./example").glob("*.wav")) + list(Path("./example").glob("*.mp3")):
        shutil.copy(str(sample_paths[0]), str(audio_path))
        return str(sample_paths[0])

    # Tạo audio im lặng 5 giây nếu không có mẫu
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", "5", "-q:a", "0", "-map", "0", str(audio_path)
    ], capture_output=True)
    return str(audio_path)

# === Hàm tạo audio bằng GPT-4o-mini-TTS ===
async def generate_gpt4o_tts(text: str, output_path: str, instructions: str, voice: str = "shimmer") -> bool:
    """Tạo audio từ văn bản bằng GPT-4o-mini-TTS với hướng dẫn về giọng điệu"""
    try:
        # Tạo file PCM tạm
        temp_pcm = output_path + ".pcm"

        # Tạo audio với streaming
        async with openai_client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice.lower(),  # API yêu cầu tên giọng viết thường
            input=text,
            instructions=instructions,
            response_format="pcm",
        ) as response:
            # Lưu nội dung PCM vào file
            with open(temp_pcm, 'wb') as f:
                async for chunk in response.iter_bytes():
                    f.write(chunk)

        # Chuyển đổi PCM sang MP3 bằng ffmpeg
        result = subprocess.run([
            "ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1",
            "-i", temp_pcm, "-acodec", "libmp3lame", "-b:a", "192k", output_path
        ], capture_output=True)

        # Xóa file tạm
        if os.path.exists(temp_pcm):
            os.remove(temp_pcm)

        return result.returncode == 0

    except Exception as e:
        print(f"Lỗi tạo GPT-4o TTS: {str(e)}")
        return False

# === Hàm nghe thử giọng nói ===
async def preview_audio_tts(text, instructions, voice, message_placeholder=None):
    """Tạo và phát mẫu giọng nói từ GPT-4o-mini-TTS"""
    try:
        if message_placeholder:
            message_placeholder.write("⏳ Đang tạo mẫu giọng nói...")

        # Tạo tệp tạm thời
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
            temp_path = temp.name

        # Tạo audio với streaming
        async with openai_client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice.lower(),  # API yêu cầu tên giọng viết thường
            input=text,
            instructions=instructions,
            response_format="pcm",
        ) as response:
            # Lưu nội dung PCM vào file
            with open(temp_path + ".pcm", 'wb') as f:
                async for chunk in response.iter_bytes():
                    f.write(chunk)

        # Chuyển đổi PCM sang MP3 bằng ffmpeg
        result = subprocess.run([
            "ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1",
            "-i", temp_path + ".pcm", "-acodec", "libmp3lame", "-b:a", "192k", temp_path
        ], capture_output=True)

        if result.returncode != 0:
            if message_placeholder:
                message_placeholder.error("Không thể chuyển đổi âm thanh. Vui lòng thử lại.")
            return None

        # Đọc file MP3 để hiển thị
        with open(temp_path, "rb") as f:
            audio_bytes = f.read()

        # Xóa files tạm
        try:
            os.unlink(temp_path)
            os.unlink(temp_path + ".pcm")
        except:
            pass

        return audio_bytes

    except Exception as e:
        if message_placeholder:
            message_placeholder.error(f"Lỗi: {str(e)}")
        return None

# === Handler cho các messages từ processing thread ===
def handle_message(msg_type: str, content: Any, containers: Dict[str, Any], show_logs: bool = True):
    """Xử lý messages từ queue dựa trên loại"""
    match msg_type:
        case 'status':
            containers['status'].write(content)
        case 'progress':
            containers['progress'].progress(content)
        case 'log':
            st.session_state.logs.append(content)
            if show_logs and 'log_content' in containers and containers['log_content']:
                containers['log_content'].code("\n".join(st.session_state.logs[-20:]))
        case 'metrics':
            with containers['metrics']:
                cols = st.columns(len(content))
                for i, (key, value) in enumerate(content.items()):
                    cols[i].metric(key, value)
        case 'error':
            st.error(content)
            st.session_state.processing = False
        case 'complete':
            st.session_state.processing = False
            st.session_state.complete = True
            st.session_state.output_file = content['output_file']

            # Thêm vào lịch sử
            st.session_state.history.append({
                'path': content['output_file'],
                'created': datetime.now(),
                'size': content.get('file_size', 0)
            })

# === Hàm xử lý video chung ===
def process_video(workflow_dict, mc_path_final, bg_path_final, audio_path_final, text_prompt, temp_dir, msg_queue, cancel_event, editor, timestamp, tts_service_val, tts_voice_val, tts_speed_val, tts_instructions_val="", position_val="Góc dưới phải", scale_val=0.25, caption_style_val="Style 01 (từng từ)", fontsize_val=48, caption_position_val="center", caption_zoom_val=False, zoom_size_val=0.01, quality_val="medium", ai_model_val="Mô hình mặc định", 
                  # Thêm các tham số khẩu hình
                  vad_alpha=1.0, exp_components=None, exp_scale=1.0, pose_scale=1.0, 
                  delta_exp_enabled=False, delta_exp_value=0.0):
    """Xử lý video trong thread riêng biệt, truyền vào đầy đủ tham số"""
    try:
        # Xác thực tham số khẩu hình để tránh lỗi
        mouth_params = validate_mouth_params(
            vad_alpha, exp_components, exp_scale, pose_scale, delta_exp_enabled, delta_exp_value
        )
        
        # Chuẩn bị files
        if workflow_dict.get("prepare_files", True):
            msg_queue.put(('status', "⏳ Đang chuẩn bị files..."))
            msg_queue.put(('progress', 5))
            msg_queue.put(('log', "Bắt đầu chuẩn bị files..."))

            # Xác định đường dẫn files tạm và đầu ra
            mc_suffix = Path(mc_path_final.name).suffix if hasattr(mc_path_final, 'name') else Path(str(mc_path_final)).suffix
            bg_suffix = Path(bg_path_final.name).suffix if hasattr(bg_path_final, 'name') and bg_path_final else ".mp4"

            mc_temp_path = temp_dir / f"mc{mc_suffix}"
            bg_temp_path = temp_dir / f"bg{bg_suffix}" if bg_path_final else None
            audio_temp_path = temp_dir / "audio.mp3"
            srt_path = temp_dir / "subtitle.srt"
            talking_path = temp_dir / "talking.mp4"
            output_file = editor.output_dir / f"video_mc_{timestamp}.mp4"
            final_output = editor.output_dir / f"final_mc_{timestamp}.mp4"

            # Đảm bảo thư mục output tồn tại
            os.makedirs(editor.output_dir, exist_ok=True)

            # Lưu files tạm
            if hasattr(mc_path_final, 'getbuffer'):  # UploadedFile
                with open(mc_temp_path, "wb") as f:
                    f.write(mc_path_final.getbuffer())
                actual_mc_path = mc_temp_path
            else:
                actual_mc_path = mc_path_final

            if bg_path_final:
                if hasattr(bg_path_final, 'getbuffer'):  # UploadedFile
                    with open(bg_temp_path, "wb") as f:
                        f.write(bg_path_final.getbuffer())
                    actual_bg_path = bg_temp_path
                else:
                    actual_bg_path = bg_path_final
            else:
                actual_bg_path = None

            msg_queue.put(('progress', 10))
        else:
            msg_queue.put(('log', "⏩ Bỏ qua bước chuẩn bị files"))
            # Đảm bảo các biến cần thiết được khởi tạo
            mc_temp_path = temp_dir / f"mc{Path(mc_path_final.name).suffix if hasattr(mc_path_final, 'name') else Path(str(mc_path_final)).suffix}"
            bg_temp_path = temp_dir / f"bg{Path(bg_path_final.name).suffix if hasattr(bg_path_final, 'name') else Path(str(bg_path_final)).suffix}" if bg_path_final else None
            audio_temp_path = temp_dir / "audio.mp3"
            srt_path = temp_dir / "subtitle.srt"
            talking_path = temp_dir / "talking.mp4"
            output_file = editor.output_dir / f"video_mc_{timestamp}.mp4"
            final_output = editor.output_dir / f"final_mc_{timestamp}.mp4"

            # Đảm bảo thư mục output tồn tại
            os.makedirs(editor.output_dir, exist_ok=True)

            # Copy files
            if hasattr(mc_path_final, 'getbuffer'):
                with open(mc_temp_path, "wb") as f:
                    f.write(mc_path_final.getbuffer())
                actual_mc_path = mc_temp_path
            else:
                actual_mc_path = mc_path_final

            if bg_path_final:
                if hasattr(bg_path_final, 'getbuffer'):
                    with open(bg_temp_path, "wb") as f:
                        f.write(bg_path_final.getbuffer())
                    actual_bg_path = bg_temp_path
                else:
                    actual_bg_path = bg_path_final
            else:
                actual_bg_path = None

        # Xử lý audio và phụ đề
        if audio_path_final:  # Upload file
            if workflow_dict.get("subtitle_generation", True):
                msg_queue.put(('status', "🔊 Đang xử lý audio..."))
                msg_queue.put(('log', "Xử lý audio từ file..."))

                if hasattr(audio_path_final, 'getbuffer'):
                    with open(audio_temp_path, "wb") as f:
                        f.write(audio_path_final.getbuffer())
                    actual_audio_path = audio_temp_path
                else:
                    actual_audio_path = audio_path_final

                # Tạo phụ đề từ audio
                msg_queue.put(('status', "📝 Đang tạo phụ đề từ audio..."))
                msg_queue.put(('log', "Bắt đầu tạo phụ đề..."))
                success, error = editor.generate_srt_from_audio(actual_audio_path, srt_path)

                if not success:
                    msg_queue.put(('error', f"Lỗi tạo phụ đề: {error}"))
                    return
            else:
                msg_queue.put(('log', "⏩ Bỏ qua bước tạo phụ đề từ audio"))

                # Vẫn xử lý audio file
                if hasattr(audio_path_final, 'getbuffer'):
                    with open(audio_temp_path, "wb") as f:
                        f.write(audio_path_final.getbuffer())
                    actual_audio_path = audio_temp_path
                else:
                    actual_audio_path = audio_path_final

                # Tạo SRT trống hoặc mẫu nếu bỏ qua bước tạo phụ đề
                if not workflow_dict.get("subtitle_generation", True):
                    create_empty_srt(srt_path)
        else:  # Tạo từ văn bản
            if workflow_dict.get("tts_generation", True):
                msg_queue.put(('status', "🎙️ Đang tạo audio từ văn bản..."))
                msg_queue.put(('log', "Bắt đầu tạo audio từ văn bản..."))

                # Lấy cài đặt TTS từ các tham số
                tts_service = "edge" if tts_service_val == "Edge TTS" else "openai"

                # Xử lý trường hợp GPT-4o-mini-TTS
                if tts_service_val == "GPT-4o-mini-TTS":
                    msg_queue.put(('log', f"Sử dụng GPT-4o-mini-TTS với giọng {tts_voice_val} để tạo giọng nói biểu cảm"))

                    # Sử dụng asyncio để chạy function async trong thread đồng bộ
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    success = loop.run_until_complete(
                        generate_gpt4o_tts(
                            text_prompt,
                            str(audio_temp_path),
                            tts_instructions_val,
                            tts_voice_val
                        )
                    )
                    loop.close()

                    if not success:
                        msg_queue.put(('error', f"Lỗi tạo audio với GPT-4o-mini-TTS"))
                        return

                    # Tạo phụ đề từ audio đã tạo
                    if workflow_dict.get("subtitle_generation", True):
                        msg_queue.put(('status', "📝 Đang tạo phụ đề từ audio..."))
                        success, error = editor.generate_srt_from_audio(str(audio_temp_path), srt_path)
                        if not success:
                            msg_queue.put(('error', f"Lỗi tạo phụ đề: {error}"))
                            create_empty_srt(srt_path)
                    else:
                        create_empty_srt(srt_path)
                else:
                    # Sử dụng phương thức tạo audio thông thường
                    success, error = editor.generate_audio_from_text(
                        text_prompt,
                        audio_temp_path,
                        srt_path,
                        service=tts_service,
                        voice=tts_voice_val,
                        speed=tts_speed_val
                    )

                    if not success:
                        msg_queue.put(('error', f"Lỗi tạo audio: {error}"))
                        return

                actual_audio_path = audio_temp_path
            else:
                msg_queue.put(('log', "⏩ Bỏ qua bước tạo audio từ văn bản"))
                actual_audio_path = use_sample_audio(audio_temp_path)

            if not workflow_dict.get("subtitle_generation", True):
                msg_queue.put(('log', "⏩ Bỏ qua bước tạo phụ đề"))
                create_empty_srt(srt_path)

        msg_queue.put(('progress', 30))

        # Kiểm tra nếu đã hủy quá trình
        if cancel_event.is_set():
            msg_queue.put(('error', "Quá trình đã bị hủy bởi người dùng"))
            return

        # Tạo video khuôn mặt nói
        if workflow_dict.get("talking_head_generation", True):
            msg_queue.put(('status', "🎭 Đang tạo video khuôn mặt nói..."))
            msg_queue.put(('log', "Bắt đầu tạo video khuôn mặt..."))

            # Chọn đường dẫn mô hình dựa trên lựa chọn của người dùng
            model_path = "./checkpoints/trt_Ampere_Plus"
            if ai_model_val == "Mô hình tối ưu hóa":
                model_path = "./checkpoints/trt_custom"
                msg_queue.put(('log', "Sử dụng mô hình tối ưu hóa"))
            else:
                msg_queue.put(('log', "Sử dụng mô hình mặc định"))
            
            # Chuẩn bị tham số khẩu hình
            # 1. Tạo use_d_keys với tỷ lệ phù hợp
            use_d_keys_dict = {}
            if mouth_params['exp_components']:
                if "exp" in mouth_params['exp_components']:
                    use_d_keys_dict["exp"] = mouth_params['exp_scale']
                for k in ["pitch", "yaw", "roll"]:
                    if k in mouth_params['exp_components']:
                        use_d_keys_dict[k] = mouth_params['pose_scale']
                if "t" in mouth_params['exp_components']:
                    use_d_keys_dict["t"] = 1.0
            
            # 2. Tạo ctrl_info
            ctrl_info = {}
            if mouth_params['vad_alpha'] < 1.0:
                msg_queue.put(('log', f"Áp dụng mức độ chuyển động môi: {mouth_params['vad_alpha']}"))
                for i in range(10000):  # Đủ cho hầu hết video
                    ctrl_info[i] = {"vad_alpha": mouth_params['vad_alpha']}
            
            # 3. Thêm delta_exp nếu được kích hoạt
            if mouth_params['delta_exp_enabled'] and mouth_params['delta_exp_value'] != 0.0:
                msg_queue.put(('log', f"Áp dụng offset biểu cảm môi: {mouth_params['delta_exp_value']}"))
                for i in range(10000):
                    if i in ctrl_info:
                        ctrl_info[i]["delta_exp"] = mouth_params['delta_exp_value']
                    else:
                        ctrl_info[i] = {"delta_exp": mouth_params['delta_exp_value']}
            
            # 4. Tạo more_kwargs
            more_kwargs = {
                "setup_kwargs": {},
                "run_kwargs": {
                    "fade_in": 1,
                    "fade_out": 1,
                }
            }
            
            # Thêm use_d_keys nếu có
            if use_d_keys_dict:
                msg_queue.put(('log', f"Áp dụng thành phần biểu cảm tùy chỉnh: {use_d_keys_dict}"))
                more_kwargs["setup_kwargs"]["use_d_keys"] = use_d_keys_dict
            
            # Thêm ctrl_info nếu có
            if ctrl_info:
                more_kwargs["run_kwargs"]["ctrl_info"] = ctrl_info
            
            # Lưu more_kwargs vào file pickle để truyền vào inference.py
            more_kwargs_path = temp_dir / "more_kwargs.pkl"
            with open(more_kwargs_path, 'wb') as f:
                pickle.dump(more_kwargs, f)
            
            # Ước tính số frame để thêm vào log và tiến trình
            audio, sr = librosa.core.load(str(actual_audio_path), sr=16000)
            num_frames = int(len(audio) / 16000 * 25)
            msg_queue.put(('log', f"Ước tính video sẽ có khoảng {num_frames} frames"))
            
            # Sử dụng subprocess để gọi inference.py thay vì import trực tiếp SDK
            cmd = [
                "python",
                "inference.py",
                "--data_root", model_path,
                "--cfg_pkl", "./checkpoints/cfg/v0.4_hubert_cfg_trt.pkl",
                "--audio_path", str(actual_audio_path),
                "--source_path", str(actual_mc_path),
                "--output_path", str(talking_path),
                "--more_kwargs", str(more_kwargs_path)
            ]
            
            msg_queue.put(('log', f"Chạy lệnh: {' '.join(cmd)}"))
            
            # Khởi chạy tiến trình inference với theo dõi output
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
            )
            
            # Xử lý output
            frame_count, fps = 0, 0
            
            # Thiết lập timeout
            start_time = time.time()
            max_wait_time = 3600  # 1 giờ
            
            while process.poll() is None:
                # Kiểm tra timeout và hủy
                if time.time() - start_time > max_wait_time or cancel_event.is_set():
                    process.terminate()
                    msg_queue.put(('error', "Quá thời gian xử lý" if time.time() - start_time > max_wait_time else "Quá trình đã bị hủy"))
                    return
                
                # Đọc một dòng từ stdout
                if line := process.stdout.readline():
                    # Lọc ANSI escape sequences
                    clean = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-? ]*[ -/]*[@-~])', '', line.strip())
                    if not clean or "aligned" in clean:
                        continue
                    
                    # Xử lý thông tin audio processing
                    if "dit:" in clean and (m := re.search(r'dit: (\d+)it . *? (\d+\.\d+)?it/s', clean)):
                        step, speed = int(m.group(1)), float(m.group(2) or 0)
                        progress_value = min(30 + step/10*10, 40)
                        msg_queue.put(('progress', int(progress_value)))
                        msg_queue.put(('status', f"Đang xử lý âm thanh → chuyển động ({step}/10)"))
                        msg_queue.put(('metrics', {'Tiến độ âm thanh': f"{step*10}%", 'Tốc độ': f"{speed:.1f}it/s"}))
                        if step % 2 == 0:
                            msg_queue.put(('log', f"➡️ Audio processing: {step}/10 ({speed:.1f}it/s)"))
                    
                    # Xử lý thông tin frame video
                    elif "writer:" in clean and (m := re.search(r'writer: (\d+)it . *? (\d+\.\d+)?it/s', clean)):
                        frame, speed = int(m.group(1)), float(m.group(2) or 0)
                        frame_count, fps = frame, speed
                        progress_value = min(40 + frame/400*20, 60)
                        msg_queue.put(('progress', int(progress_value)))
                        msg_queue.put(('status', f"Đang tạo video (frame {frame})"))
                        msg_queue.put(('metrics', {'Frames': frame, 'FPS': f"{speed:.1f}"}))
                        if frame % 50 == 0 or frame <= 5:
                            msg_queue.put(('log', f"🎬 Video: frame {frame} ({speed:.1f} fps)"))
                else:
                    # Tránh tiêu tốn CPU
                    time.sleep(0.1)
            
            # Đọc stderr output để ghi log
            stderr_output = process.stderr.read()
            
            # Kiểm tra file đầu ra
            if process.returncode != 0 or not os.path.exists(talking_path):
                # Fallback: Tạo video trực tiếp bằng ffmpeg
                msg_queue.put(('log', f"Lỗi khi tạo video khuôn mặt nói: {stderr_output}"))
                msg_queue.put(('log', "Dùng ffmpeg trực tiếp để tạo video khuôn mặt nói"))
                
                fallback_cmd = [
                    "ffmpeg", "-y", 
                    "-loop", "1" if Path(str(actual_mc_path)).suffix.lower() in ['.jpg', '.jpeg', '.png'] else "-i",
                    str(actual_mc_path),
                    "-i", str(actual_audio_path),
                    "-c:v", "libx264",
                    "-tune", "stillimage" if Path(str(actual_mc_path)).suffix.lower() in ['.jpg', '.jpeg', '.png'] else None,
                    "-c:a", "aac",
                    "-shortest", str(talking_path)
                ]
                # Lọc bỏ các tham số None
                fallback_cmd = [cmd for cmd in fallback_cmd if cmd is not None]
                
                msg_queue.put(('log', f"Lệnh fallback: {' '.join(fallback_cmd)}"))
                result = subprocess.run(fallback_cmd, capture_output=True, text=True)
                
                if result.returncode != 0 or not os.path.exists(talking_path):
                    msg_queue.put(('error', f"Lỗi khi tạo video với phương án dự phòng: {result.stderr}"))
                    return
                
                msg_queue.put(('log', "Đã tạo video bằng phương án dự phòng"))
            
            msg_queue.put(('log', f"✅ Đã tạo thành công video khuôn mặt nói: {talking_path}"))
            msg_queue.put(('progress', 60))
        else:
            msg_queue.put(('log', "⏩ Bỏ qua bước tạo video khuôn mặt nói"))
            # Nếu bỏ qua, sử dụng MC gốc làm video khuôn mặt
            if Path(str(actual_mc_path)).suffix.lower() in ['.mp4']:
                shutil.copy(str(actual_mc_path), str(talking_path))
            else:
                # Nếu là ảnh, tạo video tĩnh từ ảnh
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-loop", "1", "-i", str(actual_mc_path), 
                    "-i", str(actual_audio_path), "-c:v", "libx264", 
                    "-tune", "stillimage", "-c:a", "aac", "-shortest", str(talking_path)
                ]
                subprocess.run(ffmpeg_cmd, capture_output=True)
        
        msg_queue.put(('progress', 60))
        
        # Nếu đang ở tab "Tạo Video Khuôn Mặt AI", hoặc bỏ qua bước ghép video,
        # thì dùng talking_path làm kết quả cuối cùng
        if actual_bg_path is None or not workflow_dict.get("video_overlay", True):
            # Đây là tab "Tạo Video Khuôn Mặt AI" hoặc bỏ qua bước ghép video
            msg_queue.put(('log', "Dùng video khuôn mặt nói làm kết quả cuối cùng"))
            shutil.copy(str(talking_path), str(final_output))
            
            # Hoàn tất
            msg_queue.put(('progress', 100))
            msg_queue.put(('status', "✅ Hoàn thành!"))
            msg_queue.put(('log', "Xử lý video hoàn tất!"))
            
            # Thêm vào lịch sử và hoàn thành
            msg_queue.put(('complete', {
                'output_file': str(final_output),
                'file_size': os.path.getsize(final_output) / (1024*1024),
                'frame_count': frame_count,
                'fps': fps
            }))
            return
        
        # Kiểm tra nếu quá trình tiếp theo có được thực hiện hay không
        if cancel_event.is_set():
            msg_queue.put(('error', "Quá trình đã bị hủy bởi người dùng"))
            return
        
        # Ghép video MC và nền
        if workflow_dict.get("video_overlay", True):
            msg_queue.put(('status', "🎬 Đang ghép video..."))
            msg_queue.put(('log', "Bắt đầu ghép video..."))
            
            # Truyền trực tiếp tên tiếng Việt (không chuyển sang tiếng Anh)
            overlay_cmd = [
                "python", "video_overlay.py", 
                "-m", str(talking_path), 
                "-b", str(actual_bg_path), 
                "-o", str(output_file),
                "-p", position_val,  # Truyền trực tiếp tên tiếng Việt
                "-s", str(scale_val),
                "-q", quality_val
            ]
            
            msg_queue.put(('log', f"Chạy lệnh ghép video: {' '.join(overlay_cmd)}"))
            
            try:
                # Chạy lệnh với timeout để tránh treo
                result = subprocess.run(
                    overlay_cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=1800  # 30 phút
                )
                
                # Log output đầy đủ để debug
                if result.stdout.strip():
                    msg_queue.put(('log', f"Output: {result.stdout}"))
                if result.stderr.strip():
                    msg_queue.put(('log', f"Error: {result.stderr}"))
                
                if result.returncode != 0:
                    # Fallback: Sử dụng ffmpeg trực tiếp
                    msg_queue.put(('log', "Dùng ffmpeg trực tiếp để ghép video"))
                    
                    # Ánh xạ vị trí cho ffmpeg
                    positions = {
                        "Góc trên trái": "10:10",
                        "Góc trên phải": "main_w-overlay_w-10:10",
                        "Góc dưới trái": "10:main_h-overlay_h-10",
                        "Góc dưới phải": "main_w-overlay_w-10:main_h-overlay_h-10",
                        "Chính giữa": "(main_w-overlay_w)/2:(main_h-overlay_h)/2"
                    }
                    pos = positions.get(position_val, positions["Góc dưới phải"])
                    
                    fallback_cmd = [
                        "ffmpeg", "-y",
                        "-i", str(actual_bg_path),
                        "-i", str(talking_path),
                        "-filter_complex", f"[1:v]scale=iw*{scale_val}:-1[overlay];[0:v][overlay]overlay={pos}"
                    ]
                    
                    # Thêm audio mapping
                    try:
                        # Kiểm tra stream audio
                        audio_check = subprocess.run(
                            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(talking_path)],
                            capture_output=True, text=True
                        )
                        
                        if "audio" in audio_check.stdout:
                            fallback_cmd.extend(["-map", "1:a", "-c:a", "aac", "-b:a", "192k"])
                        else:
                            audio_check_bg = subprocess.run(
                                ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(actual_bg_path)],
                                capture_output=True, text=True
                            )
                            if "audio" in audio_check_bg.stdout:
                                fallback_cmd.extend(["-map", "0:a", "-c:a", "aac", "-b:a", "192k"])
                            else:
                                fallback_cmd.append("-an")
                    except Exception:
                        # Mặc định giữ audio từ background nếu kiểm tra thất bại
                        fallback_cmd.extend(["-map", "0:a? ", "-c:a", "aac"])
                    
                    # Thêm cài đặt video và đường dẫn output
                    fallback_cmd.extend([
                        "-c:v", "libx264",
                        "-preset", {"low": "ultrafast", "medium": "medium", "high": "slow"}.get(quality_val, "medium"),
                        "-crf", "23",
                        str(output_file)
                    ])
                    
                    msg_queue.put(('log', f"Lệnh fallback: {' '.join(fallback_cmd)}"))
                    fallback_result = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=1800)
                    
                    if fallback_result.returncode != 0:
                        msg_queue.put(('error', f"Lỗi khi ghép video với phương án dự phòng: {fallback_result.stderr}"))
                        return
                    
                    msg_queue.put(('log', "Đã ghép video bằng phương án dự phòng"))
            except subprocess.TimeoutExpired:
                msg_queue.put(('error', "Quá thời gian xử lý khi ghép video (30 phút)"))
                return
            except Exception as e:
                error_details = traceback.format_exc()
                msg_queue.put(('error', f"Lỗi không xác định khi ghép video: {str(e)}\n{error_details}"))
                return
            
            # Kiểm tra file đầu ra
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                msg_queue.put(('error', f"File đầu ra không hợp lệ: {output_file}"))
                return
            
            msg_queue.put(('log', f"✅ Đã ghép video thành công: {output_file}"))
            msg_queue.put(('progress', 80))
        else:
            msg_queue.put(('log', "⏩ Bỏ qua bước ghép video"))
            # Nếu bỏ qua, sử dụng video khuôn mặt làm kết quả
            shutil.copy(str(talking_path), str(output_file))
            msg_queue.put(('progress', 80))
        
        # Kiểm tra hủy
        if cancel_event.is_set():
            msg_queue.put(('error', "Quá trình đã bị hủy bởi người dùng"))
            return
        
        # Áp dụng phụ đề theo style đã chọn
        if workflow_dict.get("caption_application", True):
            msg_queue.put(('status', "🔤 Đang thêm phụ đề..."))
            msg_queue.put(('log', f"Thêm phụ đề kiểu: {caption_style_val}"))
            
            try:
                # Kiểm tra file SRT
                if not os.path.exists(srt_path):
                    msg_queue.put(('error', f"Không tìm thấy file phụ đề SRT: {srt_path}"))
                    return
                
                if caption_style_val == "Style 01 (từng từ)":
                    success, error = editor.apply_caption_style_01(
                        output_file,
                        srt_path,
                        final_output,
                        actual_audio_path,
                        fontsize_val
                    )
                else:  # Style 02 (gradient)
                    success, error = editor.apply_caption_style_02(
                        output_file,
                        srt_path,
                        final_output,
                        actual_audio_path,
                        fontsize_val,
                        caption_position_val,
                        caption_zoom_val,
                        zoom_size_val
                    )
                
                if not success:
                    msg_queue.put(('error', f"Lỗi khi thêm phụ đề: {error}"))
                    return
            except Exception as e:
                error_details = traceback.format_exc()
                msg_queue.put(('error', f"Lỗi không xác định khi thêm phụ đề: {str(e)}\n{error_details}"))
                return
        else:
            msg_queue.put(('log', "⏩ Bỏ qua bước thêm phụ đề"))
            # Sử dụng video không có phụ đề làm kết quả cuối cùng
            shutil.copy(str(output_file), str(final_output))
        
        # Hoàn tất
        msg_queue.put(('progress', 100))
        msg_queue.put(('status', "✅ Hoàn thành!"))
        msg_queue.put(('log', "Xử lý video hoàn tất!"))
        
        # Kiểm tra file đầu ra cuối cùng
        if not os.path.exists(final_output) or os.path.getsize(final_output) == 0:
            msg_queue.put(('error', f"File đầu ra cuối cùng không hợp lệ: {final_output}"))
            # Thử copy file output nếu có
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                shutil.copy(output_file, final_output)
                msg_queue.put(('log', f"Đã sao chép video không có phụ đề làm kết quả cuối cùng"))
            else:
                return
        
        # Thêm vào lịch sử và hoàn thành
        msg_queue.put(('complete', {
            'output_file': str(final_output),
            'file_size': os.path.getsize(final_output) / (1024*1024),
            'frame_count': frame_count if 'frame_count' in locals() else 0,
            'fps': fps if 'fps' in locals() else 0
        }))
        
    except Exception as e:
        error_details = traceback.format_exc()
        msg_queue.put(('error', f"Lỗi không xác định: {str(e)}\n{error_details}"))
    finally:
        # Dọn dẹp
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            msg_queue.put(('log', f"Lỗi khi dọn dẹp: {str(e)}"))

def main():
    # Đảm bảo session_state được khởi tạo
    init_session_state()
    
    st.set_page_config(page_title="Video AI Creator", page_icon="🎬", layout="wide")
    
    # Hiển thị logo và tiêu đề ứng dụng
    # col1, col2 = st.columns([1, 5])
    # with col1:
    #     st.image("/home/image_talking/aiclip_logo.png", width=100)
    # with col2:
    #     st.title("🎬 Video AI Creator")
    #     st.caption("Powered by [aiclip.ai](https://aiclip.ai/)")
    
    # Khởi tạo editor
    editor = VideoEditor(output_dir="./output")

    # === Sidebar cho cài đặt ===
    with st.sidebar:
        st.title("⚙️ Cài đặt")
        show_logs = st.checkbox("Hiển thị logs", value=False)  # Mặc định không hiển thị logs
        quality = st.select_slider(
            "Chất lượng video",
            options=["low", "medium", "high"],
            value="medium"
        )
        st.divider()
        
        with st.expander("💡 Mẹo sử dụng", expanded=False):
            st.markdown("""
            **Mẹo tối ưu:**
            - MC nên có nền đồng màu hoặc trong suốt
            - Video nền nên có định dạng 16:9
            - Audio nên rõ ràng, không nhiễu
            
            **Định dạng hỗ trợ:**
            - MC: JPG, PNG, MP4
            - Nền: MP4
            - Audio: WAV, MP3
            """)

    # === Tabs chính ===
    tabs = st.tabs(["Tạo Video MC", "Tạo Video Khuôn Mặt AI", "Quy Trình", "Lịch Sử", "Hướng Dẫn"])

    # === Tab 0: Tạo Video MC ===
    with tabs[0]:
        # Chia thành 2 cột
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Subtabs cho input
            input_tabs = st.tabs(["Input Files", "MC Settings", "Caption Settings"])
            
            # === Tab Input Files ===
            with input_tabs[0]:
                st.subheader("Tải lên và cài đặt")
                
                # MC uploader
                mc_file = st.file_uploader("Tải lên Ảnh/Video MC", type=["png", "jpg", "jpeg", "mp4"])
                if mc_file:
                    # Hiển thị preview cho file đã upload
                    if Path(mc_file.name).suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        st.image(mc_file, use_container_width=True, caption="MC Preview")
                    else:
                        st.video(mc_file)
                else:
                    mc_path = st.selectbox(
                        "Hoặc chọn file mẫu:",
                        options=[""] + [str(p) for p in Path("./example").glob("*.[jp][pn]g")] + [str(p) for p in Path("./example").glob("*mc*.mp4")],
                        format_func=lambda x: Path(x).name if x else "Chọn file mẫu..."
                    )
                    if mc_path:
                        if Path(mc_path).suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            st.image(mc_path, use_container_width=True, caption="MC Preview")
                        else:
                            st.video(mc_path)
                
                # BG uploader
                bg_file = st.file_uploader("Tải lên Video Nền", type=["mp4"])
                if bg_file:
                    # Hiển thị preview cho video nền đã upload
                    st.video(bg_file)
                else:
                    bg_path = st.selectbox(
                        "Hoặc chọn video nền mẫu:",
                        options=[""] + [str(p) for p in Path("./example").glob("*bg*.mp4")],
                        format_func=lambda x: Path(x).name if x else "Chọn video nền mẫu...",
                        key="bg_sample"
                    )
                    if bg_path:
                        st.video(bg_path)
                
                # Audio source
                audio_source = st.radio("Nguồn audio:", ["Upload file", "Tạo từ văn bản"], horizontal=True)
                
                if audio_source == "Upload file":
                    audio_file = st.file_uploader("Tải lên Audio thoại", type=["wav", "mp3"])
                    if audio_file:
                        # Hiển thị preview cho audio đã upload
                        st.audio(audio_file)
                    else:
                        audio_path = st.selectbox(
                            "Hoặc chọn audio mẫu:",
                            options=[""] + [str(p) for p in Path("./example").glob("*.wav")] + [str(p) for p in Path("./example").glob("*.mp3")],
                            format_func=lambda x: Path(x).name if x else "Chọn audio mẫu..."
                        )
                        if audio_path:
                            st.audio(audio_path)
                    text_prompt = None
                else:  # Tạo từ văn bản
                    audio_file = None
                    audio_path = None
                    text_prompt = st.text_area("Nhập văn bản thoại:", height=150)
                
                # TTS settings - CHỈ HIỂN THỊ KHI CHỌN "TẠO TỪ VĂN BẢN"
                if audio_source == "Tạo từ văn bản":
                    with st.expander("Cài đặt TTS", expanded=True):
                        tts_service = st.selectbox(
                            "Dịch vụ TTS:",
                            options=["Edge TTS", "OpenAI TTS", "GPT-4o-mini-TTS"],
                            index=2,  # Mặc định chọn GPT-4o-mini-TTS
                            key="tts_service"
                        )
                        
                        # Hiển thị các tùy chọn giọng nói dựa trên dịch vụ
                        if tts_service == "Edge TTS":
                            voice_options = ["vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"]
                            
                            tts_voice = st.selectbox(
                                "Giọng đọc:",
                                options=voice_options,
                                index=0,
                                key="tts_voice"
                            )
                            
                            # Hiển thị điều chỉnh tốc độ cho Edge TTS
                            tts_speed = st.slider(
                                "Tốc độ đọc:",
                                min_value=0.8,
                                max_value=1.5,
                                value=1.2,
                                step=0.1,
                                key="tts_speed"
                            )
                            
                            # Đặt giá trị mặc định cho tts_instructions
                            tts_instructions = ""
                            
                        elif tts_service == "OpenAI TTS":
                            voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
                            
                            tts_voice = st.selectbox(
                                "Giọng đọc:",
                                options=voice_options,
                                index=0,
                                key="tts_voice"
                            )
                            
                            # Hiển thị điều chỉnh tốc độ cho OpenAI TTS
                            tts_speed = st.slider(
                                "Tốc độ đọc:",
                                min_value=0.8,
                                max_value=1.5,
                                value=1.2,
                                step=0.1,
                                key="tts_speed"
                            )
                            
                            # Đặt giá trị mặc định cho tts_instructions
                            tts_instructions = ""
                            
                        else:  # GPT-4o-mini-TTS
                            voice_options = ["Ash", "Ballad", "Coral", "Echo", "Fable", "Onyx", "Nova", "Sage", "Shimmer", "Verse"]
                            
                            st.write("**🔊 Chọn giọng nói và nghe thử:**")
                            
                            tts_voice = st.selectbox(
                                "Giọng đọc:",
                                options=voice_options,
                                index=voice_options.index("Shimmer") if "Shimmer" in voice_options else 0,
                                key="tts_voice"
                            )
                            
                            # Hiển thị mô tả ngắn gọn của giọng đọc đã chọn
                            st.caption(f"**{tts_voice}**: {VOICE_DESCRIPTIONS.get(tts_voice, '')}")
                            
                            # Thêm trường hướng dẫn giọng nói cho GPT-4o-mini-TTS
                            tts_instructions = st.text_area(
                                "Hướng dẫn về giọng điệu:",
                                value="""Tone: Tự nhiên, trôi chảy, chuyên nghiệp
Emotion: Nhiệt tình, tự tin
Delivery: Rõ ràng, nhịp độ vừa phải, nhấn mạnh từ khóa quan trọng""",
                                height=100,
                                key="tts_instructions",
                                help="Mô tả tông giọng, cảm xúc và cách truyền đạt mong muốn"
                            )
                        
                            # Tạo mẫu văn bản để nghe thử
                            if text_prompt:
                                sample_text = text_prompt[:200] + "..." if len(text_prompt) > 200 else text_prompt
                            else:
                                sample_text = "Xin chào! Đây là mẫu thử giọng nói từ GPT-4o. Bạn có thể điều chỉnh văn bản này để nghe thử trước khi tạo video."
                            
                            preview_text = st.text_area(
                                "Văn bản mẫu để nghe thử:",
                                value=sample_text,
                                height=80,
                                key="preview_text"
                            )
                            
                            preview_message = st.empty()
                            preview_audio = st.empty()
                            
                            # Nút nghe thử
                            if st.button("🔊 Nghe thử giọng nói", use_container_width=True, key="tts_preview"):
                                if not preview_text.strip():
                                    preview_message.warning("Vui lòng nhập văn bản mẫu để nghe thử")
                                else:
                                    # Sử dụng asyncio
                                    audio_bytes = asyncio.run(preview_audio_tts(
                                        preview_text,
                                        tts_instructions,
                                        tts_voice,
                                        preview_message
                                    ))
                                    
                                    if audio_bytes:
                                        preview_message.success("✅ Tạo mẫu giọng nói thành công!")
                                        preview_audio.audio(audio_bytes, format="audio/mp3")
                        
                        # Hiển thị các thông tin tham khảo về giọng nói và hướng dẫn (bên ngoài expander)
                        if tts_service == "GPT-4o-mini-TTS":
                            st.divider()
                            st.subheader("🔊 Tham khảo về giọng nói GPT-4o-mini-TTS")
                            
                            voice_info_col, examples_col = st.columns(2)
                            
                            with voice_info_col:
                                st.markdown("**Đặc điểm của các giọng nói:**")
                                st.markdown("""
                                - **Ash**: Giọng nam trưởng thành, hơi trầm, phù hợp cho phim tài liệu
                                - **Ballad**: Giọng nữ mềm mại, ấm áp, phù hợp cho nội dung tư vấn
                                - **Coral**: Giọng nữ trẻ, rõ ràng, tự tin, phù hợp cho nội dung giáo dục
                                - **Echo**: Giọng nam trẻ, năng động, phù hợp cho quảng cáo
                                - **Fable**: Giọng nam uy tín, phù hợp cho thông báo chính thức
                                - **Onyx**: Giọng nam trầm, sang trọng, phù hợp cho thuyết trình
                                - **Nova**: Giọng nữ chuyên nghiệp, phù hợp cho tin tức
                                - **Sage**: Giọng nữ từng trải, ấm áp, phù hợp cho podcast
                                - **Shimmer**: Giọng nữ tươi sáng, năng động, phù hợp cho giải trí
                                - **Verse**: Giọng nam tự nhiên, cân bằng, phù hợp cho đa dạng nội dung
                                """)
                            
                            with examples_col:
                                st.markdown("**Ví dụ về hướng dẫn giọng nói:**")
                                st.markdown("""
                                **Giọng diễn thuyết:**
                                ```
                                Tone: Đĩnh đạc, trang trọng, đầy tự tin
                                Emotion: Nhiệt huyết, quyết đoán
                                Delivery: Nhịp độ vừa phải với các ngắt quãng, nhấn mạnh từ khóa quan trọng
                                ```
                                
                                **Giọng tư vấn:**
                                ```
                                Tone: Ấm áp, thân thiện, đáng tin cậy
                                Emotion: Thấu hiểu, quan tâm
                                Delivery: Nhẹ nhàng, rõ ràng, tạo cảm giác an tâm
                                ```
                                
                                **Giọng quảng cáo:**
                                ```
                                Tone: Sôi nổi, cuốn hút, năng động
                                Emotion: Phấn khích, hào hứng
                                Delivery: Nhanh, đầy năng lượng, với cường độ tăng dần
                                ```
                                """)
            
            # === Tab MC Settings ===
            with input_tabs[1]:
                st.subheader("Tùy chỉnh MC")
                
                # Vị trí và kích thước
                p_col, s_col = st.columns(2)
                with p_col:
                    position = st.selectbox(
                        "Vị trí MC",
                        ["Góc trên trái", "Góc trên phải", "Góc dưới trái", "Góc dưới phải", "Chính giữa"],
                        index=3
                    )
                
                with s_col:
                    # Auto scale checkbox
                    auto_scale = st.checkbox(
                        "Tự động điều chỉnh kích thước",
                        value=st.session_state.auto_scale
                    )
                    st.session_state.auto_scale = auto_scale
                    
                    # Tính toán scale
                    scale = 0.25  # Mặc định
                    if auto_scale and bg_file:
                        try:
                            # Lưu file bg tạm và lấy kích thước
                            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(bg_file.name).suffix) as temp:
                                temp.write(bg_file.getbuffer())
                                bg_temp_path = temp.name
                            
                            width, height = get_video_resolution(bg_temp_path)
                            
                            os.unlink(bg_temp_path)
                            
                            # Tính scale tự động
                            scale = calculate_auto_scale(mc_file if mc_file else mc_path, width, height)
                            st.write(f"Kích thước tự động: {scale:.2f}")
                        except Exception:
                            scale = 0.25
                    elif auto_scale and bg_path:
                        try:
                            width, height = get_video_resolution(bg_path)
                            scale = calculate_auto_scale(mc_file if mc_file else mc_path, width, height)
                            st.write(f"Kích thước tự động: {scale:.2f}")
                        except Exception:
                            scale = 0.25
                    else:
                        scale = st.slider("Kích thước", 0.1, 0.5, 0.25, 0.05)
                
                # Thêm phần tùy chỉnh khẩu hình
                with st.expander("🗣️ Tùy chỉnh khẩu hình", expanded=False):
                    # Điều chỉnh mức độ chuyển động môi
                    vad_alpha = st.slider(
                        "Mức độ chuyển động môi:",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0,
                        step=0.05,
                        help="Giá trị thấp hơn sẽ làm giảm chuyển động môi, giá trị cao hơn sẽ tăng chuyển động"
                    )
                    
                    # Tùy chọn nâng cao
                    mouth_advanced = st.checkbox("Tùy chọn nâng cao cho khẩu hình", value=False)
                    if mouth_advanced:
                        # Chọn các thành phần biểu cảm
                        exp_components = st.multiselect(
                            "Thành phần biểu cảm:",
                            options=["exp", "pitch", "yaw", "roll", "t"],
                            default=["exp", "pitch", "yaw", "roll", "t"],
                            help="Chọn các thành phần biểu cảm để sử dụng từ mô hình"
                        )
                        
                        # Điều chỉnh tỷ lệ cho các thành phần
                        exp_scale = st.slider(
                            "Tỷ lệ biểu cảm miệng (exp):",
                            min_value=0.5,
                            max_value=1.5,
                            value=1.0,
                            step=0.1,
                            help="Điều chỉnh tỷ lệ biểu cảm miệng"
                        )
                        
                        pose_scale = st.slider(
                            "Tỷ lệ chuyển động đầu (pitch, yaw, roll):",
                            min_value=0.5,
                            max_value=1.5,
                            value=1.0,
                            step=0.1,
                            help="Điều chỉnh tỷ lệ chuyển động đầu"
                        )
                        
                        # Điều chỉnh offset biểu cảm môi
                        delta_exp_enabled = st.checkbox("Thêm offset biểu cảm môi", value=False)
                        if delta_exp_enabled:
                            delta_exp_value = st.slider(
                                "Giá trị offset:",
                                min_value=-0.2,
                                max_value=0.2,
                                value=0.0,
                                step=0.01
                            )
            
            # === Tab Caption Settings ===
            with input_tabs[2]:
                st.subheader("Tùy chỉnh phụ đề")
                
                # Style phụ đề
                caption_style = st.radio(
                    "Kiểu phụ đề:",
                    ["Style 01 (từng từ)", "Style 02 (gradient)"],
                    horizontal=True,
                    index=0
                )
                
                # Auto fontsize
                auto_fontsize = st.checkbox(
                    "Tự động điều chỉnh kích thước phụ đề",
                    value=st.session_state.auto_fontsize
                )
                st.session_state.auto_fontsize = auto_fontsize
                
                # Tính toán fontsize
                fontsize = 48  # Mặc định
                if auto_fontsize and bg_file:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(bg_file.name).suffix) as temp:
                            temp.write(bg_file.getbuffer())
                            bg_temp_path = temp.name
                        
                        width, height = get_video_resolution(bg_temp_path)
                        
                        os.unlink(bg_temp_path)
                        
                        fontsize = calculate_auto_fontsize(width, height)
                        st.write(f"Kích thước phụ đề tự động: {fontsize}")
                    except Exception:
                        fontsize = 48
                elif auto_fontsize and bg_path:
                    try:
                        width, height = get_video_resolution(bg_path)
                        fontsize = calculate_auto_fontsize(width, height)
                        st.write(f"Kích thước phụ đề tự động: {fontsize}")
                    except Exception:
                        fontsize = 48
                else:
                    fontsize = st.slider("Kích thước phụ đề:", 24, 100, 48, 2)
                
                # Cài đặt cho Style 02
                caption_position, caption_zoom, zoom_size = "center", False, 0.01
                if caption_style == "Style 02 (gradient)":
                    caption_position = st.selectbox("Vị trí phụ đề:", ["center", "top", "bottom"], index=0)
                    caption_zoom = st.checkbox("Hiệu ứng zoom phụ đề", value=True)
                    zoom_size = st.slider("Độ lớn hiệu ứng zoom:", 0.005, 0.05, 0.01, 0.005) if caption_zoom else 0.01
            
            # Nút "Tạo Video"
            submitted = st.button(
                "🚀 Tạo Video",
                use_container_width=True,
                type="primary",
                disabled=st.session_state.processing
            )
        
        # === Cột hiển thị tiến trình và kết quả ===
        with col2:
            # Tạo các placeholder cho UI trạng thái
            elapsed_time_container = st.empty()
            status_container = st.empty()
            progress_container = st.empty()
            metrics_container = st.container()
            cancel_container = st.empty()
            log_container = st.container()
            
            if st.session_state.processing:
                status_container.subheader("⏳ Đang xử lý...")
                
                # Hiển thị thời gian xử lý
                if st.session_state.process_start_time:
                    elapsed = time.time() - st.session_state.process_start_time
                    elapsed_time_container.caption(f"Thời gian xử lý: {int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
                
                progress = progress_container.progress(0)
                
                # Nút hủy xử lý
                cancel_button = cancel_container.button("🛑 Hủy xử lý", use_container_width=True)
                
                # Hiển thị logs
                if show_logs:
                    log_container.markdown("**Logs:**")
                    log_content = log_container.code("\n".join(st.session_state.logs[-20:]))
            
            elif st.session_state.complete and st.session_state.output_file:
                status_container.subheader("✅ Đã hoàn thành!")
                
                output_file = st.session_state.output_file
                if Path(output_file).exists():
                    metrics_container.video(output_file)
                    
                    file_stats = Path(output_file).stat()
                    
                    # Hiển thị thông tin video
                    cols = metrics_container.columns(2)
                    cols[0].metric("Kích thước", f"{file_stats.st_size / (1024*1024):.1f} MB")
                    cols[1].metric("Thời gian tạo", datetime.fromtimestamp(file_stats.st_mtime).strftime("%H:%M:%S"))
                    
                    with open(output_file, "rb") as file:
                        cancel_container.download_button(
                            "💾 Tải xuống video",
                            file,
                            file_name=Path(output_file).name,
                            mime="video/mp4",
                            use_container_width=True
                        )
            else:
                status_container.subheader("Trạng thái")
                metrics_container.info("Nhấn nút 'Tạo Video' để bắt đầu xử lý...")
            
            # Preview chỉ hiển thị nếu không đang trong quá trình xử lý hoặc hoàn thành
            if mc_file or (mc_path if 'mc_path' in locals() else None):
                preview_container = log_container.container()
                preview_container.subheader("MC Preview")
                if mc_file:
                    if Path(mc_file.name).suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        preview_container.image(mc_file, use_container_width=True)
                    else:
                        preview_container.video(mc_file)
                elif 'mc_path' in locals() and mc_path:
                    if Path(mc_path).suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        preview_container.image(mc_path, use_container_width=True)
                    else:
                        preview_container.video(mc_path)

    # === Tab 1: Tạo Video Khuôn Mặt AI ===
    with tabs[1]:
        st.subheader("🎭 Tạo Video Khuôn Mặt Nói với AI")
        st.write("Chuyển đổi ảnh hoặc video MC tĩnh thành video với khả năng nói theo audio")
        
        # Chia thành 2 cột
        ai_col1, ai_col2 = st.columns([3, 2])
        
        with ai_col1:
            # Input files section
            st.subheader("Tải lên files đầu vào")
            
            # MC uploader
            ai_mc_file = st.file_uploader("Tải lên Ảnh/Video MC", type=["png", "jpg", "jpeg", "mp4"], key="ai_mc_file")
            if ai_mc_file:
                if Path(ai_mc_file.name).suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    st.image(ai_mc_file, use_container_width=True, caption="MC Preview")
                else:
                    st.video(ai_mc_file)
            else:
                ai_mc_path = st.selectbox(
                    "Hoặc chọn file mẫu:",
                    options=[""] + [str(p) for p in Path("./example").glob("*.[jp][pn]g")] + [str(p) for p in Path("./example").glob("*mc*.mp4")],
                    format_func=lambda x: Path(x).name if x else "Chọn file mẫu...",
                    key="ai_mc_sample"
                )
                if ai_mc_path:
                    if Path(ai_mc_path).suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        st.image(ai_mc_path, use_container_width=True, caption="MC Preview")
                    else:
                        st.video(ai_mc_path)
            
            # Audio source
            ai_audio_source = st.radio("Nguồn audio:", ["Upload file", "Tạo từ văn bản"], horizontal=True, key="ai_audio_source")
            
            if ai_audio_source == "Upload file":
                ai_audio_file = st.file_uploader("Tải lên Audio thoại", type=["wav", "mp3"], key="ai_audio_file")
                if ai_audio_file:
                    st.audio(ai_audio_file)
                else:
                    ai_audio_path = st.selectbox(
                        "Hoặc chọn audio mẫu:",
                        options=[""] + [str(p) for p in Path("./example").glob("*.wav")] + [str(p) for p in Path("./example").glob("*.mp3")],
                        format_func=lambda x: Path(x).name if x else "Chọn audio mẫu...",
                        key="ai_audio_sample"
                    )
                    if ai_audio_path:
                        st.audio(ai_audio_path)
                ai_text_prompt = None
            else:  # Tạo từ văn bản
                ai_audio_file = None
                ai_audio_path = None
                ai_text_prompt = st.text_area("Nhập văn bản thoại:", height=150, key="ai_text_prompt")
            
            # TTS settings
            if ai_audio_source == "Tạo từ văn bản":
                with st.expander("Cài đặt TTS", expanded=True):
                    ai_tts_service = st.selectbox(
                        "Dịch vụ TTS:",
                        options=["Edge TTS", "OpenAI TTS", "GPT-4o-mini-TTS"],
                        index=2,  # Mặc định chọn GPT-4o-mini-TTS
                        key="ai_tts_service"
                    )
                    
                    # Hiển thị các tùy chọn giọng nói dựa trên dịch vụ
                    if ai_tts_service == "Edge TTS":
                        ai_voice_options = ["vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"]
                        
                        ai_tts_voice = st.selectbox(
                            "Giọng đọc:",
                            options=ai_voice_options,
                            index=0,
                            key="ai_tts_voice"
                        )
                        
                        # Hiển thị điều chỉnh tốc độ cho Edge TTS
                        ai_tts_speed = st.slider(
                            "Tốc độ đọc:",
                            min_value=0.8,
                            max_value=1.5,
                            value=1.2,
                            step=0.1,
                            key="ai_tts_speed"
                        )
                        
                        # Đặt giá trị mặc định cho ai_tts_instructions
                        ai_tts_instructions = ""
                        
                    elif ai_tts_service == "OpenAI TTS":
                        ai_voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
                        
                        ai_tts_voice = st.selectbox(
                            "Giọng đọc:",
                            options=ai_voice_options,
                            index=0,
                            key="ai_tts_voice"
                        )
                        
                        # Hiển thị điều chỉnh tốc độ cho OpenAI TTS
                        ai_tts_speed = st.slider(
                            "Tốc độ đọc:",
                            min_value=0.8,
                            max_value=1.5,
                            value=1.2,
                            step=0.1,
                            key="ai_tts_speed"
                        )
                        
                        # Đặt giá trị mặc định cho ai_tts_instructions
                        ai_tts_instructions = ""
                        
                    else:  # GPT-4o-mini-TTS
                        ai_voice_options = ["Ash", "Ballad", "Coral", "Echo", "Fable", "Onyx", "Nova", "Sage", "Shimmer", "Verse"]
                        
                        st.write("**🔊 Chọn giọng nói và nghe thử:**")
                        
                        ai_tts_voice = st.selectbox(
                            "Giọng đọc:",
                            options=ai_voice_options,
                            index=ai_voice_options.index("Shimmer") if "Shimmer" in ai_voice_options else 0,
                            key="ai_tts_voice"
                        )
                        
                        # Hiển thị mô tả ngắn gọn của giọng đọc đã chọn
                        st.caption(f"**{ai_tts_voice}**: {VOICE_DESCRIPTIONS.get(ai_tts_voice, '')}")
                        
                        # Thêm trường hướng dẫn giọng nói
                        ai_tts_instructions = st.text_area(
                            "Hướng dẫn về giọng điệu:",
                            value="""Tone: Tự nhiên, trôi chảy, chuyên nghiệp
Emotion: Nhiệt tình, tự tin
Delivery: Rõ ràng, nhịp độ vừa phải, nhấn mạnh từ khóa quan trọng""",
                            height=100,
                            key="ai_tts_instructions"
                        )
                        
                        # Tạo mẫu văn bản để nghe thử
                        if ai_text_prompt:
                            ai_sample_text = ai_text_prompt[:200] + "..." if len(ai_text_prompt) > 200 else ai_text_prompt
                        else:
                            ai_sample_text = "Xin chào! Đây là mẫu thử giọng nói từ GPT-4o. Bạn có thể điều chỉnh văn bản này để nghe thử trước khi tạo video."
                        
                        ai_preview_text = st.text_area(
                            "Văn bản mẫu để nghe thử:",
                            value=ai_sample_text,
                            height=80,
                            key="ai_preview_text"
                        )
                        
                        ai_preview_message = st.empty()
                        ai_preview_audio = st.empty()
                        
                        # Nút nghe thử
                        if st.button("🔊 Nghe thử giọng nói", use_container_width=True, key="ai_tts_preview"):
                            if not ai_preview_text.strip():
                                ai_preview_message.warning("Vui lòng nhập văn bản mẫu để nghe thử")
                            else:
                                # Sử dụng asyncio
                                audio_bytes = asyncio.run(preview_audio_tts(
                                    ai_preview_text,
                                    ai_tts_instructions,
                                    ai_tts_voice,
                                    ai_preview_message
                                ))
                                
                                if audio_bytes:
                                    ai_preview_message.success("✅ Tạo mẫu giọng nói thành công!")
                                    ai_preview_audio.audio(audio_bytes, format="audio/mp3")
                        
                        # Hiển thị các thông tin tham khảo về giọng nói và hướng dẫn (bên ngoài expander)
                        if ai_tts_service == "GPT-4o-mini-TTS":
                            st.divider()
                            st.subheader("🔊 Tham khảo về giọng nói GPT-4o-mini-TTS")
                            
                            ai_voice_info_col, ai_examples_col = st.columns(2)
                            
                            with ai_voice_info_col:
                                st.markdown("**Đặc điểm của các giọng nói:**")
                                st.markdown("""
                                - **Ash**: Giọng nam trưởng thành, hơi trầm, phù hợp cho phim tài liệu
                                - **Ballad**: Giọng nữ mềm mại, ấm áp, phù hợp cho nội dung tư vấn
                                - **Coral**: Giọng nữ trẻ, rõ ràng, tự tin, phù hợp cho nội dung giáo dục
                                - **Echo**: Giọng nam trẻ, năng động, phù hợp cho quảng cáo
                                - **Fable**: Giọng nam uy tín, phù hợp cho thông báo chính thức
                                - **Onyx**: Giọng nam trầm, sang trọng, phù hợp cho thuyết trình
                                - **Nova**: Giọng nữ chuyên nghiệp, phù hợp cho tin tức
                                - **Sage**: Giọng nữ từng trải, ấm áp, phù hợp cho podcast
                                - **Shimmer**: Giọng nữ tươi sáng, năng động, phù hợp cho giải trí
                                - **Verse**: Giọng nam tự nhiên, cân bằng, phù hợp cho đa dạng nội dung
                                """)
                            
                            with ai_examples_col:
                                st.markdown("**Ví dụ về hướng dẫn giọng nói:**")
                                st.markdown("""
                                **Giọng diễn thuyết:**
                                ```
                                Tone: Đĩnh đạc, trang trọng, đầy tự tin
                                Emotion: Nhiệt huyết, quyết đoán
                                Delivery: Nhịp độ vừa phải với các ngắt quãng, nhấn mạnh từ khóa quan trọng
                                ```
                                
                                **Giọng tư vấn:**
                                ```
                                Tone: Ấm áp, thân thiện, đáng tin cậy
                                Emotion: Thấu hiểu, quan tâm
                                Delivery: Nhẹ nhàng, rõ ràng, tạo cảm giác an tâm
                                ```
                                
                                **Giọng quảng cáo:**
                                ```
                                Tone: Sôi nổi, cuốn hút, năng động
                                Emotion: Phấn khích, hào hứng
                                Delivery: Nhanh, đầy năng lượng, với cường độ tăng dần
                                ```
                                """)
            
            # Cài đặt AI model
            with st.expander("Cài đặt mô hình AI", expanded=False):
                ai_model = st.selectbox(
                    "Mô hình AI:",
                    options=["Mô hình mặc định", "Mô hình tối ưu hóa"],
                    help="Chọn mô hình AI để tạo video khuôn mặt nói",
                    index=0,
                    key="ai_model"
                )
                
                ai_quality = st.select_slider(
                    "Chất lượng video AI:",
                    options=["low", "medium", "high"],
                    value="medium",
                    key="ai_quality"
                )
                
                ai_advanced = st.checkbox("Cài đặt nâng cao", value=False, key="ai_advanced")
                if ai_advanced:
                    ai_inference_steps = st.slider(
                        "Số bước inference:",
                        min_value=5,
                        max_value=20,
                        value=10,
                        step=1,
                        key="ai_inference_steps"
                    )
            
            # Tùy chỉnh khẩu hình AI (riêng biệt, không lồng trong expander khác)
            with st.expander("🗣️ Tùy chỉnh khẩu hình", expanded=False):
                # Điều chỉnh mức độ chuyển động môi
                ai_vad_alpha = st.slider(
                    "Mức độ chuyển động môi:",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                    step=0.05,
                    key="ai_vad_alpha",
                    help="Giá trị thấp hơn sẽ làm giảm chuyển động môi, giá trị cao hơn sẽ tăng chuyển động"
                )
                
                # Tùy chọn nâng cao
                ai_mouth_advanced = st.checkbox("Tùy chọn nâng cao cho khẩu hình", value=False, key="ai_mouth_advanced")
                if ai_mouth_advanced:
                    # Chọn các thành phần biểu cảm
                    ai_exp_components = st.multiselect(
                        "Thành phần biểu cảm:",
                        options=["exp", "pitch", "yaw", "roll", "t"],
                        default=["exp", "pitch", "yaw", "roll", "t"],
                        key="ai_exp_components",
                        help="Chọn các thành phần biểu cảm để sử dụng từ mô hình"
                    )
                    
                    # Điều chỉnh tỷ lệ cho các thành phần
                    ai_exp_scale = st.slider(
                        "Tỷ lệ biểu cảm miệng (exp):",
                        min_value=0.5,
                        max_value=1.5,
                        value=1.0,
                        step=0.1,
                        key="ai_exp_scale",
                        help="Điều chỉnh tỷ lệ biểu cảm miệng"
                    )
                    
                    ai_pose_scale = st.slider(
                        "Tỷ lệ chuyển động đầu (pitch, yaw, roll):",
                        min_value=0.5,
                        max_value=1.5,
                        value=1.0,
                        step=0.1,
                        key="ai_pose_scale",
                        help="Điều chỉnh tỷ lệ chuyển động đầu"
                    )
                    
                    # Điều chỉnh offset biểu cảm môi
                    ai_delta_exp_enabled = st.checkbox("Thêm offset biểu cảm môi", value=False, key="ai_delta_exp_enabled")
                    if ai_delta_exp_enabled:
                        ai_delta_exp_value = st.slider(
                            "Giá trị offset:",
                            min_value=-0.2,
                            max_value=0.2,
                            value=0.0,
                            step=0.01,
                            key="ai_delta_exp_value"
                        )
            
            # Nút "Tạo Video Khuôn Mặt Nói"
            ai_submitted = st.button(
                "🚀 Tạo Video Khuôn Mặt Nói",
                use_container_width=True,
                type="primary",
                key="ai_create_button"
            )
        
        # Cột hiển thị tiến trình và kết quả
        with ai_col2:
            # Tạo các placeholder cho UI trạng thái
            ai_elapsed_time_container = st.empty()
            ai_status_container = st.empty()
            ai_progress_container = st.empty()
            ai_metrics_container = st.container()
            ai_cancel_container = st.empty()
            ai_result_container = st.container()
            
            if st.session_state.processing:
                ai_status_container.subheader("⏳ Đang xử lý...")
                
                # Hiển thị thời gian xử lý
                if st.session_state.process_start_time:
                    elapsed = time.time() - st.session_state.process_start_time
                    ai_elapsed_time_container.caption(f"Thời gian xử lý: {int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
                
                progress = ai_progress_container.progress(0)
                
                # Nút hủy xử lý
                cancel_button = ai_cancel_container.button("🛑 Hủy xử lý", key="ai_cancel_processing", use_container_width=True)
            elif st.session_state.complete and st.session_state.output_file:
                ai_status_container.subheader("✅ Đã hoàn thành!")
                
                output_file = st.session_state.output_file
                if Path(output_file).exists():
                    ai_metrics_container.video(output_file)
                    
                    file_stats = Path(output_file).stat()
                    
                    # Hiển thị thông tin video
                    cols = ai_metrics_container.columns(2)
                    cols[0].metric("Kích thước", f"{file_stats.st_size / (1024*1024):.1f} MB")
                    cols[1].metric("Thời gian tạo", datetime.fromtimestamp(file_stats.st_mtime).strftime("%H:%M:%S"))
                    
                    with open(output_file, "rb") as file:
                        ai_cancel_container.download_button(
                            "💾 Tải xuống video",
                            file,
                            file_name=Path(output_file).name,
                            mime="video/mp4",
                            use_container_width=True,
                            key="ai_download_button"
                        )
            else:
                # Hiển thị trạng thái ban đầu
                ai_status_container.subheader("Trạng thái")
                ai_metrics_container.info("Nhấn nút 'Tạo Video Khuôn Mặt Nói' để bắt đầu xử lý...")
        
        # Xử lý khi nhấn nút tạo video
        if ai_submitted and not st.session_state.processing:
            # Xác định đường dẫn files
            mc_path_final = ai_mc_file if ai_mc_file else ai_mc_path if 'ai_mc_path' in locals() and ai_mc_path else None
            audio_path_final = ai_audio_file if ai_audio_file else ai_audio_path if 'ai_audio_path' in locals() and ai_audio_path else None
            
            if (mc_path_final and (audio_path_final or ai_text_prompt)):
                # Chuẩn bị workflow chỉ cho talking head generation
                ai_workflow_steps = {k: False for k in WORKFLOW_STEPS}
                ai_workflow_steps["prepare_files"] = True
                ai_workflow_steps["tts_generation"] = ai_audio_source == "Tạo từ văn bản"
                ai_workflow_steps["talking_head_generation"] = True
                
                # Cài đặt session state và UI
                st.session_state.processing = True
                st.session_state.process_start_time = time.time()
                st.session_state.logs = ["Bắt đầu quá trình tạo video khuôn mặt nói..."]
                
                # Cập nhật UI trạng thái ban đầu
                ai_status_container.subheader("⏳ Đang xử lý...")
                elapsed = time.time() - st.session_state.process_start_time
                ai_elapsed_time_container.caption(f"Thời gian xử lý: {int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
                progress = ai_progress_container.progress(0)
                cancel_button = ai_cancel_container.button("🛑 Hủy xử lý", key="ai_cancel_processing", use_container_width=True)
                
                # Chuẩn bị tempdir và đường dẫn
                temp_dir = Path(tempfile.mkdtemp())
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Chuẩn bị hàng đợi giao tiếp
                msg_queue = queue.Queue()
                cancel_event = threading.Event()
                
                # Chuẩn bị containers cho handler
                ui_containers = {
                    'status': ai_status_container,
                    'progress': progress,
                    'log_content': None,  # Không hiển thị log trên tab này
                    'metrics': ai_metrics_container
                }
                
                # Chuẩn bị các tham số TTS
                tts_service_val = ai_tts_service if ai_audio_source == "Tạo từ văn bản" else "Edge TTS"
                tts_voice_val = ai_tts_voice if ai_audio_source == "Tạo từ văn bản" else "vi-VN-NamMinhNeural"
                tts_speed_val = ai_tts_speed if ai_audio_source == "Tạo từ văn bản" and ai_tts_service != "GPT-4o-mini-TTS" else 1.2
                tts_instructions_val = ai_tts_instructions if ai_audio_source == "Tạo từ văn bản" and ai_tts_service == "GPT-4o-mini-TTS" else ""
                
                # Khởi chạy thread xử lý với workflow chỉ dành cho AI talking head
                thread = threading.Thread(
                    target=process_video,
                    args=(
                        ai_workflow_steps,
                        mc_path_final,
                        None,
                        audio_path_final,
                        ai_text_prompt,
                        temp_dir,
                        msg_queue,
                        cancel_event,
                        editor,
                        timestamp,
                        tts_service_val,
                        tts_voice_val,
                        tts_speed_val,
                    ),
                    kwargs={
                        'tts_instructions_val': tts_instructions_val,
                        'ai_model_val': ai_model,  # Truyền lựa chọn mô hình AI
                        'quality_val': ai_quality,
                        # Thêm các tham số khẩu hình
                        'vad_alpha': ai_vad_alpha if 'ai_vad_alpha' in locals() else 1.0,
                        'exp_components': ai_exp_components if 'ai_exp_components' in locals() and ai_mouth_advanced else None,
                        'exp_scale': ai_exp_scale if 'ai_exp_scale' in locals() and ai_mouth_advanced else 1.0,
                        'pose_scale': ai_pose_scale if 'ai_pose_scale' in locals() and ai_mouth_advanced else 1.0,
                        'delta_exp_enabled': ai_delta_exp_enabled if 'ai_delta_exp_enabled' in locals() and ai_mouth_advanced else False,
                        'delta_exp_value': ai_delta_exp_value if 'ai_delta_exp_value' in locals() and ai_delta_exp_enabled and ai_mouth_advanced else 0.0,
                    }
                )
                thread.daemon = True
                thread.start()
                
                # UI theo dõi tiến trình (tương tự như tab chính)
                try:
                    while thread.is_alive() or not msg_queue.empty():
                        # Cập nhật thời gian xử lý
                        if st.session_state.process_start_time:
                            elapsed = time.time() - st.session_state.process_start_time
                            ai_elapsed_time_container.caption(
                                f"Thời gian xử lý: {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
                            )
                        
                        # Kiểm tra nếu đã nhấn nút hủy
                        if cancel_button:
                            cancel_event.set()
                            st.session_state.processing = False
                            st.warning("Đã hủy quá trình xử lý")
                            break
                        
                        # Xử lý thông điệp
                        try:
                            msg_type, content = msg_queue.get(timeout=0.1)
                            handle_message(msg_type, content, ui_containers, False)  # Không hiển thị logs
                            msg_queue.task_done()
                            
                            if msg_type == 'complete':
                                time.sleep(0.5)  # Đợi UI cập nhật
                                st.rerun()
                        except queue.Empty:
                            time.sleep(0.1)
                        
                        # Thêm độ trễ để tránh quá tải UI
                        time.sleep(0.05)
                except Exception as e:
                    st.error(f"Lỗi UI: {str(e)}")
                    st.session_state.processing = False
            else:
                st.error("Vui lòng tải lên cả MC và audio (hoặc nhập văn bản)")

    # === Tab 2: Quy Trình ===
    with tabs[2]:
        st.subheader("⚙️ Cấu Hình Quy Trình")
        
        st.write("Chọn các bước xử lý cần thực hiện. Các bước không được chọn sẽ bị bỏ qua trong quá trình xử lý.")
        
        # Chia thành 2 cột để hiển thị checkbox
        left_col, right_col = st.columns(2)
        
        # Đảm bảo workflow_steps có trong session state
        workflow_steps_dict = {}
        
        # Hiển thị các checkbox trong hai cột để giao diện cân đối
        steps = list(WORKFLOW_STEPS.items())
        mid_idx = len(steps) // 2 + len(steps) % 2
        
        # Cột trái
        with left_col:
            for step_id, step_name in steps[:mid_idx]:
                # Lấy giá trị hiện tại từ session state một cách an toàn
                current_value = True
                if 'workflow_steps' in st.session_state:
                    if isinstance(st.session_state.workflow_steps, dict):
                        current_value = st.session_state.workflow_steps.get(step_id, True)
                
                workflow_steps_dict[step_id] = st.checkbox(
                    step_name,
                    value=current_value,
                    key=f"workflow_{step_id}"
                )
        
        # Cột phải
        with right_col:
            for step_id, step_name in steps[mid_idx:]:
                # Lấy giá trị hiện tại từ session state một cách an toàn
                current_value = True
                if 'workflow_steps' in st.session_state:
                    if isinstance(st.session_state.workflow_steps, dict):
                        current_value = st.session_state.workflow_steps.get(step_id, True)
                
                workflow_steps_dict[step_id] = st.checkbox(
                    step_name,
                    value=current_value,
                    key=f"workflow_{step_id}"
                )
        
        # Cập nhật workflow_steps trong session state
        st.session_state.workflow_steps = workflow_steps_dict
        
        # Thêm mô tả chi tiết
        with st.expander("ℹ️ Chi tiết các bước xử lý", expanded=False):
            st.markdown("""
            **Mô tả chi tiết từng bước:**
            
            **Chuẩn bị files**: Chuẩn bị và sao chép các file đầu vào để xử lý
            
            **Tạo âm thanh từ văn bản**: Sử dụng công nghệ TTS để tạo âm thanh từ văn bản nhập vào
            
            **Tạo phụ đề từ audio**: Phân tích audio và tạo file phụ đề (.srt)
            
            **Tạo video khuôn mặt nói**: Sử dụng AI để tạo hiệu ứng nói cho MC dựa trên audio
            
            **Ghép video MC và nền**: Ghép video MC đã tạo vào video nền với vị trí đã chọn
            
            **Thêm phụ đề vào video**: Áp dụng phụ đề vào video cuối cùng với hiệu ứng đã chọn
            """)
        
        # Nút reset tất cả các bước
        if st.button("↩️ Khôi phục tất cả bước", use_container_width=True):
            st.session_state.workflow_steps = {k: True for k in WORKFLOW_STEPS}
            st.rerun()
        
        # Hiển thị thông tin về các điều kiện phụ thuộc
        st.info("**Lưu ý:** Một số bước phụ thuộc vào các bước trước đó. Khi bỏ qua một bước, hệ thống sẽ tự động dùng dữ liệu mẫu hoặc kết quả có sẵn.")

    # === Tab 3: Lịch Sử ===
    with tabs[3]:
        st.subheader("Lịch sử video đã tạo")
        
        # Cập nhật lịch sử từ thư mục
        update_history_from_folder()
        
        if not st.session_state.history:
            st.info("Chưa có video nào được tạo.")
        else:
            # Tùy chọn sắp xếp và tìm kiếm
            col1, col2 = st.columns(2)
            with col1:
                sort_option = st.selectbox(
                    "Sắp xếp theo:",
                    ["Thời gian tạo (mới nhất)", "Thời gian tạo (cũ nhất)", "Kích thước (lớn nhất)", "Kích thước (nhỏ nhất)"],
                    index=0
                )
                
            with col2:
                search_term = st.text_input("Tìm kiếm:", placeholder="Nhập tên file...")
            
            # Sắp xếp và lọc lịch sử
            history = sorted(
                st.session_state.history,
                key=lambda x: x.get('created', datetime.now()) if sort_option.startswith("Thời gian") else x.get('size', 0),
                reverse=sort_option in ["Thời gian tạo (mới nhất)", "Kích thước (lớn nhất)"]
            )
            
            # Lọc theo từ khóa
            if search_term:
                history = [item for item in history if search_term.lower() in Path(item.get('path', '')).name.lower()]
            
            # Hiển thị danh sách
            for i, item in enumerate(history):
                file_path = Path(item['path'])
                if not file_path.exists():
                    continue
                
                with st.expander(f"{file_path.name} ({item['created'].strftime('%Y-%m-%d %H:%M:%S')})", expanded=i==0):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.video(str(file_path))
                    
                    with col2:
                        st.write(f"Kích thước: {item['size']:.1f} MB")
                        st.write(f"Thời gian tạo: {item['created'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    buttons_col1, buttons_col2 = st.columns(2)
                    with buttons_col1:
                        with open(file_path, "rb") as f:
                            st.download_button(
                                "💾 Tải xuống",
                                f,
                                file_name=file_path.name,
                                mime="video/mp4",
                                key=f"download_{file_path.name}",
                                use_container_width=True
                            )
                    
                    with buttons_col2:
                        if st.button("🗑️ Xóa video", key=f"delete_{file_path.name}", use_container_width=True):
                            try:
                                file_path.unlink()
                                st.session_state.history = [h for h in st.session_state.history if h['path'] != str(file_path)]
                                st.success(f"Đã xóa {file_path.name}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Không thể xóa file: {str(e)}")

        # === Tab 4: Hướng Dẫn ===
    with tabs[4]:
        st.subheader("Hướng dẫn sử dụng")
        
        # Các phần hướng dẫn
        with st.expander("🚀 Bắt đầu nhanh", expanded=True):
            st.markdown("""
            **Các bước cơ bản để tạo video MC:**
            
            - 1. **Tải lên hoặc chọn file MC**
              - Hình ảnh hoặc video có nhân vật nói
            - 2. **Tải lên hoặc chọn video nền**
              - Video nền cho MC
            - 3. **Chọn nguồn audio**
              - Tải lên file audio hoặc tạo mới từ văn bản
            - 4. **Điều chỉnh cài đặt MC**
              - Vị trí và kích thước MC trên video
            - 5. **Điều chỉnh cài đặt phụ đề**
              - Chọn kiểu và kích thước phụ đề
            - 6. **Nhấn nút "Tạo Video"**
              - Chờ xử lý và tải xuống kết quả
            """)
        
        with st.expander("⚙️ Các tính năng nâng cao", expanded=False):
            st.markdown("""
            **Tính năng tự động hóa:**
            - Tự động điều chỉnh kích thước MC dựa trên tỷ lệ video nền
            - Tự động điều chỉnh kích thước phụ đề phù hợp với độ phân giải
            
            **Kiểu phụ đề:**
            - Style 01: Hiển thị từng từ một với màu sắc thay đổi
            - Style 02: Phụ đề có hiệu ứng màu gradient và zoom nhẹ
            
            **Cấu hình quy trình:**
            - Chọn các bước cụ thể để tùy chỉnh quy trình xử lý
            - Bỏ qua các bước không cần thiết để tiết kiệm thời gian
            
            **Video Khuôn Mặt AI:**
            - Sử dụng tab chuyên biệt để tạo video khuôn mặt nói mà không cần video nền
            - Chọn giữa mô hình mặc định và mô hình tối ưu hóa để đạt kết quả tốt nhất
            
            **GPT-4o-mini-TTS:**
            - Tạo âm thanh với cảm xúc và giọng điệu phong phú
            - Chọn từ 10 giọng đọc khác nhau (Ash, Ballad, Coral, Echo, Fable, Onyx, Nova, Sage, Shimmer, Verse)
            - Tùy chỉnh hướng dẫn về tông giọng, cảm xúc và cách truyền đạt
            - Nghe thử trước khi tạo video để đảm bảo chất lượng
            
            **Điều chỉnh khẩu hình miệng:**
            - Điều chỉnh mức độ chuyển động môi theo âm thanh
            - Tùy chỉnh thành phần biểu cảm và tỷ lệ áp dụng
            - Thêm offset biểu cảm môi để điều chỉnh hình dáng miệng
            """)
        
        with st.expander("🔍 Xử lý sự cố", expanded=False):
            st.markdown("""
            **Vấn đề thường gặp:**
            1. **Xử lý chậm:** Giảm kích thước/độ phân giải file đầu vào
            2. **Lỗi khi xử lý video:** Kiểm tra định dạng file
            3. **Phụ đề không đồng bộ:** Kiểm tra audio có rõ ràng không
            
            **Nếu gặp lỗi:**
            - Xem logs để biết chi tiết
            - Thử sử dụng file mẫu để xác định vấn đề
            """)

    # === Xử lý khi submit ===
    if submitted and not st.session_state.processing:
        # Xác định đường dẫn các files
        mc_path_final = mc_file if mc_file else mc_path if 'mc_path' in locals() and mc_path else None
        bg_path_final = bg_file if bg_file else bg_path if 'bg_path' in locals() and bg_path else None
        audio_path_final = audio_file if audio_file else audio_path if 'audio_path' in locals() and audio_path else None
        
        # Lấy workflow_steps từ session_state
        if 'workflow_steps' not in st.session_state:
            workflow_steps = {k: True for k in WORKFLOW_STEPS}
        else:
            workflow_steps = st.session_state.workflow_steps.copy() if isinstance(st.session_state.workflow_steps, dict) else {k: True for k in WORKFLOW_STEPS}
        
        if ((mc_path_final and bg_path_final) and (audio_path_final or text_prompt)):
            st.session_state.processing = True
            st.session_state.process_start_time = time.time()
            st.session_state.logs = ["Bắt đầu quá trình xử lý..."]
            
            # Cập nhật UI trạng thái ban đầu
            status_container.subheader("⏳ Đang xử lý...")
            elapsed = time.time() - st.session_state.process_start_time
            elapsed_time_container.caption(f"Thời gian xử lý: {int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
            progress = progress_container.progress(0)
            cancel_button = cancel_container.button("🛑 Hủy xử lý", key="cancel_processing", use_container_width=True)
            if show_logs:
                log_container.markdown("**Logs:**")
                log_content = log_container.code("Đang bắt đầu...")
            
            # Tạo tempdir và đường dẫn
            temp_dir = Path(tempfile.mkdtemp())
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Chuẩn bị hàng đợi giao tiếp
            msg_queue = queue.Queue()
            
            # Sử dụng threading.Event thay vì truy cập st.session_state từ luồng phụ
            cancel_event = threading.Event()
            
            # Chuẩn bị containers cho handler
            ui_containers = {
                'status': status_container,
                'progress': progress,
                'log_content': log_content if show_logs else None,
                'metrics': metrics_container
            }
            
            # Chuẩn bị dữ liệu để truyền cho thread
            # Trích xuất các giá trị cần thiết từ UI để truyền vào thread
            tts_service_val = tts_service if audio_source == "Tạo từ văn bản" else "Edge TTS"
            tts_voice_val = tts_voice if audio_source == "Tạo từ văn bản" else "vi-VN-NamMinhNeural"
            tts_speed_val = tts_speed if audio_source == "Tạo từ văn bản" and tts_service != "GPT-4o-mini-TTS" else 1.2
            tts_instructions_val = tts_instructions if audio_source == "Tạo từ văn bản" and tts_service == "GPT-4o-mini-TTS" else ""
            caption_style_val = caption_style
            fontsize_val = fontsize
            caption_position_val = caption_position
            caption_zoom_val = caption_zoom
            zoom_size_val = zoom_size
            quality_val = quality
            
            # Khởi chạy thread xử lý với tham số là dictionary workflow_steps
            thread = threading.Thread(
                target=process_video,
                args=(
                    workflow_steps,
                    mc_path_final,
                    bg_path_final,
                    audio_path_final,
                    text_prompt,
                    temp_dir,
                    msg_queue,
                    cancel_event,
                    editor,
                    timestamp,
                    tts_service_val,
                    tts_voice_val,
                    tts_speed_val,
                ),
                kwargs={
                    'tts_instructions_val': tts_instructions_val,
                    'position_val': position,
                    'scale_val': scale,
                    'caption_style_val': caption_style_val,
                    'fontsize_val': fontsize_val,
                    'caption_position_val': caption_position_val,
                    'caption_zoom_val': caption_zoom_val,
                    'zoom_size_val': zoom_size_val,
                    'quality_val': quality_val,
                    'ai_model_val': "Mô hình mặc định",  # Mặc định cho tab chính
                    # Thêm các tham số khẩu hình
                    'vad_alpha': vad_alpha if 'vad_alpha' in locals() else 1.0,
                    'exp_components': exp_components if 'exp_components' in locals() and mouth_advanced else None,
                    'exp_scale': exp_scale if 'exp_scale' in locals() and mouth_advanced else 1.0,
                    'pose_scale': pose_scale if 'pose_scale' in locals() and mouth_advanced else 1.0,
                    'delta_exp_enabled': delta_exp_enabled if 'delta_exp_enabled' in locals() and mouth_advanced else False,
                    'delta_exp_value': delta_exp_value if 'delta_exp_value' in locals() and delta_exp_enabled and mouth_advanced else 0.0,
                }
            )
            thread.daemon = True
            thread.start()
            
            # UI theo dõi tiến trình
            try:
                while thread.is_alive() or not msg_queue.empty():
                    # Cập nhật thời gian xử lý
                    if st.session_state.process_start_time:
                        elapsed = time.time() - st.session_state.process_start_time
                        elapsed_time_container.caption(f"Thời gian xử lý: {int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
                    
                    # Kiểm tra nếu đã nhấn nút hủy
                    if cancel_button:
                        cancel_event.set()
                        st.session_state.processing = False
                        st.warning("Đã hủy quá trình xử lý")
                        break
                    
                    # Xử lý thông điệp
                    try:
                        msg_type, content = msg_queue.get(timeout=0.1)
                        handle_message(msg_type, content, ui_containers, show_logs)
                        msg_queue.task_done()
                        
                        if msg_type == 'complete':
                            time.sleep(0.5)  # Đợi UI cập nhật trước khi rerun
                            st.rerun()
                    except queue.Empty:
                        time.sleep(0.1)
                    
                    # Thêm độ trễ để tránh quá tải UI
                    time.sleep(0.05)
                
                # Nếu thread vẫn chạy nhưng vòng lặp kết thúc
                if thread.is_alive():
                    st.warning("Tiến trình xử lý vẫn đang chạy nền...")
                
            except Exception as e:
                error_details = traceback.format_exc()
                st.error(f"Lỗi UI: {str(e)}\n{error_details}")
                st.session_state.processing = False
        else:
            st.error("Vui lòng chọn đầy đủ: MC, video nền, và audio (hoặc nhập văn bản)")

    # Thêm footer với thông tin aiclip.ai
    # st.markdown("---")
    # st.markdown(
    #     "<div style='text-align: center;'>"
    #     "Copyright © 2025 "
    #     "<a href='https://aiclip.ai' target='_blank'>aiclip.ai</a>"
    #     "</div>",
    #     unsafe_allow_html=True
    # )

if __name__ == "__main__":
    main()
