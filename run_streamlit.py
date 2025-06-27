#!/usr/bin/env python3
"""Streamlit UI tối ưu cho Google Colab - AI Video Creator
Bao gồm tất cả tính năng nâng cao trừ subtitle
"""

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

# Tắt chế độ theo dõi file của Streamlit
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

# Import modules
try:
    from video_editor import VideoEditor
except ImportError:
    st.error("❌ Không thể import VideoEditor. Vui lòng kiểm tra file video_editor.py")
    st.stop()

# OpenAI client - sẽ được khởi tạo sau khi có API key
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    st.warning("⚠️ OpenAI client không khả dụng")
    OPENAI_AVAILABLE = False
    
openai_client = None

# === Khởi tạo OpenAI Client ===
def initialize_openai_client():
    """Khởi tạo OpenAI client với API key từ session state"""
    global openai_client
    if OPENAI_AVAILABLE and st.session_state.openai_api_key.strip():
        try:
            openai_client = AsyncOpenAI(api_key=st.session_state.openai_api_key.strip())
            return True
        except Exception as e:
            st.error(f"Lỗi khởi tạo OpenAI client: {str(e)}")
            openai_client = None
            return False
    else:
        openai_client = None
        return False

# === Kiểm tra API Key ===
async def test_openai_api_key(api_key: str) -> Tuple[bool, str]:
    """Kiểm tra tính hợp lệ của OpenAI API key"""
    try:
        test_client = AsyncOpenAI(api_key=api_key.strip())
        # Test với một request đơn giản
        response = await test_client.models.list()
        return True, "API key hợp lệ"
    except Exception as e:
        error_msg = str(e)
        if "invalid_api_key" in error_msg:
            return False, "API key không hợp lệ"
        elif "rate_limit" in error_msg:
            return False, "Đã vượt quá giới hạn rate limit"
        elif "insufficient_quota" in error_msg:
            return False, "Tài khoản không đủ quota"
        else:
            return False, f"Lỗi kết nối: {error_msg}"

# === Cấu hình Google Colab ===
def get_colab_config():
    """Lấy cấu hình từ environment variables cho Google Colab"""
    config = {
        'data_root': os.environ.get('DITTO_DATA_ROOT', './checkpoints/ditto_trt'),
        'gpu_arch': os.environ.get('DITTO_GPU_ARCH', 'pre_ampere'),
        'cfg_pkl': './checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl'
    }
    
    # Kiểm tra files tồn tại
    if not os.path.exists(config['data_root']):
        st.error(f"❌ Thư mục model không tìm thấy: {config['data_root']}")
        st.info("💡 Vui lòng chạy lại cell setup models")
        st.stop()
    
    if not os.path.exists(config['cfg_pkl']):
        st.error(f"❌ File config không tìm thấy: {config['cfg_pkl']}")
        st.info("💡 Vui lòng chạy lại cell tải config")
        st.stop()
    
    return config

# === Định nghĩa các bước quy trình ===
WORKFLOW_STEPS = {
    "prepare_files": "Chuẩn bị files",
    "tts_generation": "Tạo âm thanh từ văn bản",
    "talking_head_generation": "Tạo video khuôn mặt nói",
    "video_overlay": "Ghép video MC và nền",
}

# === Thông tin mô tả giọng nói ===
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

# === Khởi tạo session state ===
def init_session_state():
    """Khởi tạo toàn bộ session state cần thiết"""
    # Lấy API key từ environment variable trước (từ notebook setup)
    env_openai_key = os.environ.get('OPENAI_API_KEY', '').strip()
    
    defaults = {
        'processing': False,
        'complete': False,
        'output_file': None,
        'history': [],
        'process_start_time': None,
        'auto_scale': True,
        'logs': [],
        'workflow_steps': {k: True for k in WORKFLOW_STEPS},
        'cancel_event': None,
        'msg_queue': None,
        'tts_instructions_preset': "Tone: Tự nhiên, trôi chảy, chuyên nghiệp\nEmotion: Nhiệt tình, tự tin\nDelivery: Rõ ràng, nhịp độ vừa phải, nhấn mạnh từ khóa quan trọng",
        'openai_api_key': env_openai_key,  # Ưu tiên environment variable
        'openai_api_status': 'valid' if env_openai_key else 'not_tested',  # Assume valid if from env
        'api_key_source': 'environment' if env_openai_key else 'manual',  # Track source
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Nếu có API key từ environment và chưa khởi tạo client
    if env_openai_key and not st.session_state.openai_api_key:
        st.session_state.openai_api_key = env_openai_key
        st.session_state.openai_api_status = 'valid'
        st.session_state.api_key_source = 'environment'

# === Hàm xác thực tham số khẩu hình ===
def validate_mouth_params(vad_alpha=1.0, exp_components=None, exp_scale=1.0, pose_scale=1.0, delta_exp_enabled=False, delta_exp_value=0.0):
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

# === Hàm tiện ích ===
@lru_cache(maxsize=32)
def get_video_resolution(video_path: str) -> Tuple[int, int]:
    """Lấy độ phân giải của video với cache"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
            cap.release()
            return width, height
        return 1920, 1080
    except Exception:
        return 1920, 1080

def calculate_auto_scale(mc_path: Union[str, Any], bg_width: int, bg_height: int) -> float:
    """Tính toán tỉ lệ scale phù hợp cho MC"""
    try:
        mc_width, mc_height = 0, 0
        
        # Xử lý các loại đầu vào khác nhau
        if hasattr(mc_path, 'getbuffer'):  # UploadedFile
            suffix = Path(mc_path.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                temp.write(mc_path.getbuffer())
                temp_path = temp.name
            
            try:
                if suffix in ['.jpg', '.jpeg', '.png']:
                    img = cv2.imread(temp_path)
                    if img is not None:
                        mc_width, mc_height = img.shape[1], img.shape[0]
                else:  # Video
                    cap = cv2.VideoCapture(temp_path)
                    if cap.isOpened():
                        mc_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        mc_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
            finally:
                os.unlink(temp_path)
        else:  # Path string
            mc_path_str = str(mc_path)
            suffix = Path(mc_path_str).suffix.lower()
            
            if suffix in ['.jpg', '.jpeg', '.png']:
                img = cv2.imread(mc_path_str)
                if img is not None:
                    mc_width, mc_height = img.shape[1], img.shape[0]
            else:  # Video
                cap = cv2.VideoCapture(mc_path_str)
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
    """Cập nhật lịch sử từ thư mục output"""
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

# === Hàm tạo audio bằng GPT-4o-mini-TTS ===
async def generate_gpt4o_tts(text: str, output_path: str, instructions: str, voice: str = "shimmer") -> bool:
    """Tạo audio từ văn bản bằng GPT-4o-mini-TTS với hướng dẫn về giọng điệu"""
    try:
        # Kiểm tra openai_client có sẵn
        if not openai_client:
            raise Exception("OpenAI client chưa được khởi tạo. Vui lòng kiểm tra API key trong tab Cài đặt.")
        
        # Tạo file PCM tạm
        temp_pcm = output_path + ".pcm"
        
        # Tạo audio với streaming
        async with openai_client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice.lower(),
            input=text,
            response_format="pcm",
        ) as response:
            # Lưu nội dung PCM vào file
            with open(temp_pcm, 'wb') as f:
                async for chunk in response.iter_bytes():
                    f.write(chunk)
        
        # Chuyển đổi PCM sang MP3 bằng ffmpeg
        result = subprocess.run([
            "ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", temp_pcm,
            "-acodec", "libmp3lame", "-b:a", "192k", output_path
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
        # Kiểm tra openai_client có sẵn
        if not openai_client:
            raise Exception("OpenAI client chưa được khởi tạo. Vui lòng kiểm tra API key trong tab Cài đặt.")
        
        if message_placeholder:
            message_placeholder.write("⏳ Đang tạo mẫu giọng nói...")
        
        # Tạo tệp tạm thời
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
            temp_path = temp.name
        
        # Tạo audio với streaming
        async with openai_client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice.lower(),
            input=text,
            response_format="pcm",
        ) as response:
            # Lưu nội dung PCM vào file
            with open(temp_path + ".pcm", 'wb') as f:
                async for chunk in response.iter_bytes():
                    f.write(chunk)
        
        # Chuyển đổi PCM sang MP3 bằng ffmpeg
        result = subprocess.run([
            "ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", temp_path + ".pcm",
            "-acodec", "libmp3lame", "-b:a", "192k", temp_path
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
    if msg_type == 'status':
        containers['status'].write(content)
    elif msg_type == 'progress':
        containers['progress'].progress(content)
    elif msg_type == 'log':
        st.session_state.logs.append(content)
        if show_logs and 'log_content' in containers and containers['log_content']:
            containers['log_content'].code("\n".join(st.session_state.logs[-20:]))
    elif msg_type == 'metrics':
        with containers['metrics']:
            cols = st.columns(len(content))
            for i, (key, value) in enumerate(content.items()):
                cols[i].metric(key, value)
    elif msg_type == 'error':
        st.error(content)
        st.session_state.processing = False
    elif msg_type == 'complete':
        st.session_state.processing = False
        st.session_state.complete = True
        st.session_state.output_file = content['output_file']
        
        # Thêm vào lịch sử
        st.session_state.history.append({
            'path': content['output_file'],
            'created': datetime.now(),
            'size': content.get('file_size', 0)
        })

# === Hàm xử lý video chính ===
def process_video(workflow_dict, mc_path_final, bg_path_final, audio_path_final, text_prompt, temp_dir, msg_queue, cancel_event, editor, timestamp, tts_service_val, tts_voice_val, tts_speed_val, tts_instructions_val="", position_val="Góc dưới phải", scale_val=0.25, quality_val="medium", ai_model_val="Mô hình mặc định", vad_alpha=1.0, exp_components=None, exp_scale=1.0, pose_scale=1.0, delta_exp_enabled=False, delta_exp_value=0.0):
    """Xử lý video trong thread riêng biệt với đầy đủ tính năng"""
    try:
        # Xác thực tham số khẩu hình
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
            # Thiết lập các biến cần thiết
            mc_temp_path = temp_dir / f"mc{Path(mc_path_final.name).suffix if hasattr(mc_path_final, 'name') else Path(str(mc_path_final)).suffix}"
            bg_temp_path = temp_dir / f"bg{Path(bg_path_final.name).suffix if hasattr(bg_path_final, 'name') else Path(str(bg_path_final)).suffix}" if bg_path_final else None
            audio_temp_path = temp_dir / "audio.mp3"
            talking_path = temp_dir / "talking.mp4"
            output_file = editor.output_dir / f"video_mc_{timestamp}.mp4"
            final_output = editor.output_dir / f"final_mc_{timestamp}.mp4"
            
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
        
        # Xử lý audio
        if audio_path_final:  # Upload file
            if hasattr(audio_path_final, 'getbuffer'):
                with open(audio_temp_path, "wb") as f:
                    f.write(audio_path_final.getbuffer())
                actual_audio_path = audio_temp_path
            else:
                actual_audio_path = audio_path_final
        else:  # Tạo từ văn bản
            if workflow_dict.get("tts_generation", True):
                msg_queue.put(('status', "🎙️ Đang tạo audio từ văn bản..."))
                msg_queue.put(('log', "Bắt đầu tạo audio từ văn bản..."))
                
                # Xử lý TTS dựa trên service
                if tts_service_val == "GPT-4o-mini-TTS":
                    msg_queue.put(('log', f"Sử dụng GPT-4o-mini-TTS với giọng {tts_voice_val}"))
                    
                    # Sử dụng asyncio để chạy function async
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
                else:
                    # Sử dụng các service khác
                    tts_service = "edge" if tts_service_val == "Edge TTS" else "openai"
                    success, error = editor.generate_audio_from_text(
                        text_prompt,
                        audio_temp_path,
                        service=tts_service,
                        voice=tts_voice_val,
                        speed=tts_speed_val
                    )
                    
                    if not success:
                        msg_queue.put(('error', f"Lỗi tạo audio: {error}"))
                        return
                
                actual_audio_path = audio_temp_path
            else:
                msg_queue.put(('log', "⏩ Bỏ qua bước tạo audio"))
                # Tạo audio mẫu
                actual_audio_path = str(audio_temp_path)
                subprocess.run([
                    "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                    "-t", "5", "-q:a", "0", "-map", "0", str(audio_temp_path)
                ], capture_output=True)
        
        msg_queue.put(('progress', 30))
        
        # Kiểm tra nếu đã hủy quá trình
        if cancel_event.is_set():
            msg_queue.put(('error', "Quá trình đã bị hủy bởi người dùng"))
            return
        
        # Tạo video khuôn mặt nói
        if workflow_dict.get("talking_head_generation", True):
            msg_queue.put(('status', "🎭 Đang tạo video khuôn mặt nói..."))
            msg_queue.put(('log', "Bắt đầu tạo video khuôn mặt..."))
            
            # Lấy cấu hình
            config = get_colab_config()
            
            # Chọn model path dựa trên lựa chọn
            if ai_model_val == "Mô hình tối ưu hóa":
                model_path = config['data_root'].replace('ditto_trt', 'ditto_trt_custom')
                msg_queue.put(('log', "Sử dụng mô hình tối ưu hóa"))
            else:
                model_path = config['data_root']
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
                for i in range(10000):
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
            
            # Lưu more_kwargs vào file pickle
            more_kwargs_path = temp_dir / "more_kwargs.pkl"
            with open(more_kwargs_path, 'wb') as f:
                pickle.dump(more_kwargs, f)
            
            # Ước tính số frame
            audio, sr = librosa.core.load(str(actual_audio_path), sr=16000)
            num_frames = int(len(audio) / 16000 * 25)
            msg_queue.put(('log', f"Ước tính video sẽ có khoảng {num_frames} frames"))
            
            # Sử dụng subprocess để gọi inference.py
            cmd = [
                "python", "inference.py",
                "--data_root", model_path,
                "--cfg_pkl", config['cfg_pkl'],
                "--audio_path", str(actual_audio_path),
                "--source_path", str(actual_mc_path),
                "--output_path", str(talking_path),
                "--more_kwargs", str(more_kwargs_path)
            ]
            
            msg_queue.put(('log', f"Chạy lệnh: {' '.join(cmd)}"))
            
            # Khởi chạy tiến trình inference với theo dõi output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
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
                    clean = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', line.strip())
                    if not clean or "aligned" in clean:
                        continue
                    
                    # Xử lý thông tin audio processing
                    if "dit:" in clean and (m := re.search(r'dit: (\d+)it.*?(\d+\.\d+)?it/s', clean)):
                        step, speed = int(m.group(1)), float(m.group(2) or 0)
                        progress_value = min(30 + step/10*10, 40)
                        msg_queue.put(('progress', int(progress_value)))
                        msg_queue.put(('status', f"Đang xử lý âm thanh → chuyển động ({step}/10)"))
                        msg_queue.put(('metrics', {'Tiến độ âm thanh': f"{step*10}%", 'Tốc độ': f"{speed:.1f}it/s"}))
                        if step % 2 == 0:
                            msg_queue.put(('log', f"➡️ Audio processing: {step}/10 ({speed:.1f}it/s)"))
                    
                    # Xử lý thông tin frame video
                    elif "writer:" in clean and (m := re.search(r'writer: (\d+)it.*?(\d+\.\d+)?it/s', clean)):
                        frame, speed = int(m.group(1)), float(m.group(2) or 0)
                        frame_count, fps = frame, speed
                        progress_value = min(40 + frame/400*20, 60)
                        msg_queue.put(('progress', int(progress_value)))
                        msg_queue.put(('status', f"Đang tạo video (frame {frame})"))
                        msg_queue.put(('metrics', {'Frames': frame, 'FPS': f"{speed:.1f}"}))
                        if frame % 50 == 0 or frame <= 5:
                            msg_queue.put(('log', f"🎬 Video: frame {frame} ({speed:.1f} fps)"))
                else:
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
                    "-c:a", "aac",
                    "-shortest", str(talking_path)
                ]
                # Lọc bỏ các tham số None
                fallback_cmd = [cmd for cmd in fallback_cmd if cmd is not None]
                
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
                    "-i", str(actual_audio_path), "-c:v", "libx264", "-tune", "stillimage",
                    "-c:a", "aac", "-shortest", str(talking_path)
                ]
                subprocess.run(ffmpeg_cmd, capture_output=True)
            
            msg_queue.put(('progress', 60))
        
        # Nếu không có background, sử dụng talking head làm kết quả cuối
        if actual_bg_path is None or not workflow_dict.get("video_overlay", True):
            msg_queue.put(('log', "Dùng video khuôn mặt nói làm kết quả cuối cùng"))
            shutil.copy(str(talking_path), str(final_output))
            
            msg_queue.put(('progress', 100))
            msg_queue.put(('status', "✅ Hoàn thành!"))
            msg_queue.put(('log', "Xử lý video hoàn tất!"))
            
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
            
            # Truyền trực tiếp tên tiếng Việt
            overlay_cmd = [
                "python", "video_overlay.py",
                "-m", str(talking_path),
                "-b", str(actual_bg_path),
                "-o", str(output_file),
                "-p", position_val,
                "-s", str(scale_val),
                "-q", quality_val
            ]
            
            msg_queue.put(('log', f"Chạy lệnh ghép video: {' '.join(overlay_cmd)}"))
            
            try:
                # Chạy lệnh với timeout
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
                        "-filter_complex", f"[1:v]scale=iw*{scale_val}:ih*{scale_val}[overlay];[0:v][overlay]overlay={pos}",
                        "-c:v", "libx264",
                        "-preset", {"low": "ultrafast", "medium": "medium", "high": "slow"}.get(quality_val, "medium"),
                        "-crf", "23",
                        str(output_file)
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
                        fallback_cmd.extend(["-map", "0:a?", "-c:a", "aac"])
                    
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
        
        # Sử dụng output_file làm kết quả cuối cùng (không có subtitle)
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
                msg_queue.put(('log', f"Đã sao chép video làm kết quả cuối cùng"))
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

# === Main App ===
def main():
    # Cấu hình trang
    st.set_page_config(
        page_title="🎭 AI Video Creator",
        page_icon="🎭",
        layout="wide"
    )
    
    # Khởi tạo session state
    init_session_state()
    
    # Lấy cấu hình
    config = get_colab_config()
    
    # Header
    st.title("🎭 AI Video Creator")
    st.caption("Tạo video khuôn mặt nói với AI - Phiên bản Google Colab")
    
    # Hiển thị thông tin cấu hình
    with st.expander("⚙️ Thông tin hệ thống", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📁 **Models**: {config['data_root']}")
            st.info(f"🎮 **GPU**: {config['gpu_arch']}")
        with col2:
            st.info(f"📊 **Config**: {config['cfg_pkl']}")
            st.info(f"🔧 **Trạng thái**: {'✅ Sẵn sàng' if os.path.exists(config['data_root']) else '❌ Chưa sẵn sàng'}")
    
    # Khởi tạo editor
    editor = VideoEditor(output_dir="./output")
    
    # === Sidebar cho cài đặt ===
    with st.sidebar:
        st.title("⚙️ Cài đặt chung")
        
        # Workflow configuration
        st.subheader("🔄 Quy trình xử lý")
        workflow_steps = {}
        for step_id, step_name in WORKFLOW_STEPS.items():
            workflow_steps[step_id] = st.checkbox(
                step_name,
                value=st.session_state.workflow_steps.get(step_id, True),
                key=f"workflow_{step_id}"
            )
        st.session_state.workflow_steps = workflow_steps
        
        # Quality settings
        st.subheader("📊 Chất lượng")
        quality = st.select_slider(
            "Chất lượng video",
            options=["low", "medium", "high"],
            value="medium",
            format_func=lambda x: {"low": "Thấp", "medium": "Trung bình", "high": "Cao"}[x]
        )
        
        # Show logs
        show_logs = st.checkbox("Hiển thị logs chi tiết", value=False)
        
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
    tabs = st.tabs(["🎬 Tạo Video MC", "🎭 Video Khuôn Mặt AI", "🎙️ Text-to-Speech", "📋 Lịch Sử", "⚙️ Cài Đặt", "❓ Hướng Dẫn"])
    
    # === Tab 0: Tạo Video MC ===
    with tabs[0]:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("📁 Tải lên files")
            
            # MC uploader
            mc_file = st.file_uploader(
                "Tải lên Ảnh/Video MC",
                type=["png", "jpg", "jpeg", "mp4"],
                help="Ảnh hoặc video của người MC"
            )
            if mc_file:
                if Path(mc_file.name).suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    st.image(mc_file, use_container_width=True, caption="Xem trước MC")
                else:
                    st.video(mc_file)
            else:
                mc_path = st.selectbox(
                    "Hoặc chọn file mẫu:",
                    options=[""] + [str(p) for p in Path("./example").glob("*.[jp][pn]g")] + [str(p) for p in Path("./example").glob("*mc*.mp4")],
                    format_func=lambda x: Path(x).name if x else "Chọn file mẫu...",
                    key="mc_sample"
                )
                if mc_path:
                    if Path(mc_path).suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        st.image(mc_path, use_container_width=True, caption="Xem trước MC")
                    else:
                        st.video(mc_path)
            
            # BG uploader
            bg_file = st.file_uploader(
                "Tải lên Video Nền",
                type=["mp4"],
                help="Video nền để ghép với MC"
            )
            if bg_file:
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
            audio_source = st.radio(
                "Nguồn audio:",
                ["Upload file", "Tạo từ văn bản"],
                horizontal=True
            )
            
            if audio_source == "Upload file":
                audio_file = st.file_uploader(
                    "Tải lên Audio thoại",
                    type=["wav", "mp3"]
                )
                if audio_file:
                    st.audio(audio_file)
                else:
                    audio_path = st.selectbox(
                        "Hoặc chọn audio mẫu:",
                        options=[""] + [str(p) for p in Path("./example").glob("*.wav")] + [str(p) for p in Path("./example").glob("*.mp3")],
                        format_func=lambda x: Path(x).name if x else "Chọn audio mẫu...",
                        key="audio_sample"
                    )
                    if audio_path:
                        st.audio(audio_path)
                text_prompt = None
            else:
                audio_file = None
                audio_path = None
                text_prompt = st.text_area(
                    "Nhập văn bản thoại:",
                    height=150,
                    placeholder="Nhập nội dung bạn muốn MC nói..."
                )
                
                # TTS settings ngắn gọn
                if text_prompt:
                    with st.expander("🎙️ Cài đặt TTS nhanh", expanded=True):
                        tts_service = st.selectbox(
                            "Dịch vụ TTS:",
                            options=["Edge TTS", "OpenAI TTS", "GPT-4o-mini-TTS"],
                            index=0,
                            key="tts_service_main"
                        )
                        
                        if tts_service == "Edge TTS":
                            tts_voice = st.selectbox(
                                "Giọng đọc:",
                                options=["vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"],
                                key="tts_voice_main"
                            )
                            tts_speed = st.slider("Tốc độ:", 0.8, 1.5, 1.2, 0.1, key="tts_speed_main")
                            tts_instructions = ""
                        elif tts_service == "OpenAI TTS":
                            tts_voice = st.selectbox(
                                "Giọng đọc:",
                                options=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                                key="tts_voice_main"
                            )
                            tts_speed = st.slider("Tốc độ:", 0.8, 1.5, 1.2, 0.1, key="tts_speed_main")
                            tts_instructions = ""
                        else:  # GPT-4o-mini-TTS
                            tts_voice = st.selectbox(
                                "Giọng đọc:",
                                options=["Ash", "Ballad", "Coral", "Echo", "Fable", "Onyx", "Nova", "Sage", "Shimmer", "Verse"],
                                index=8,
                                key="tts_voice_main"
                            )
                            tts_speed = 1.2
                            tts_instructions = st.text_area(
                                "Hướng dẫn giọng điệu:",
                                value="Tone: Tự nhiên, trôi chảy\nEmotion: Nhiệt tình, tự tin\nDelivery: Rõ ràng, nhịp độ vừa phải",
                                height=80,
                                key="tts_instructions_main"
                            )
            
            # MC Settings
            st.subheader("🎛️ Cài đặt MC")
            
            # Position and scale
            col_p, col_s = st.columns(2)
            with col_p:
                position = st.selectbox(
                    "Vị trí MC",
                    ["Góc trên trái", "Góc trên phải", "Góc dưới trái", "Góc dưới phải", "Chính giữa"],
                    index=3
                )
            
            with col_s:
                auto_scale = st.checkbox(
                    "Tự động điều chỉnh kích thước",
                    value=st.session_state.auto_scale
                )
                st.session_state.auto_scale = auto_scale
            
            # Scale calculation
            scale = 0.25
            if auto_scale and bg_file:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(bg_file.name).suffix) as temp:
                        temp.write(bg_file.getbuffer())
                        bg_temp_path = temp.name
                    
                    width, height = get_video_resolution(bg_temp_path)
                    os.unlink(bg_temp_path)
                    
                    scale = calculate_auto_scale(mc_file if mc_file else mc_path, width, height)
                    st.write(f"Kích thước tự động: {scale:.2f}")
                except Exception:
                    scale = 0.25
            elif auto_scale and 'bg_path' in locals() and bg_path:
                try:
                    width, height = get_video_resolution(bg_path)
                    scale = calculate_auto_scale(mc_file if mc_file else mc_path, width, height)
                    st.write(f"Kích thước tự động: {scale:.2f}")
                except Exception:
                    scale = 0.25
            else:
                scale = st.slider("Kích thước", 0.1, 0.5, 0.25, 0.05)
            
            # Advanced mouth controls
            with st.expander("🗣️ Điều khiển khẩu hình nâng cao", expanded=False):
                vad_alpha = st.slider(
                    "Mức độ chuyển động môi:",
                    min_value=0.0, max_value=1.0, value=1.0, step=0.05,
                    help="Giá trị thấp hơn sẽ làm giảm chuyển động môi"
                )
                
                mouth_advanced = st.checkbox("Tùy chọn nâng cao", value=False)
                if mouth_advanced:
                    exp_components = st.multiselect(
                        "Thành phần biểu cảm:",
                        options=["exp", "pitch", "yaw", "roll", "t"],
                        default=["exp", "pitch", "yaw", "roll", "t"],
                        help="Chọn các thành phần biểu cảm để sử dụng"
                    )
                    
                    exp_scale = st.slider(
                        "Tỷ lệ biểu cảm miệng:",
                        min_value=0.5, max_value=1.5, value=1.0, step=0.1
                    )
                    
                    pose_scale = st.slider(
                        "Tỷ lệ chuyển động đầu:",
                        min_value=0.5, max_value=1.5, value=1.0, step=0.1
                    )
                    
                    delta_exp_enabled = st.checkbox("Thêm offset biểu cảm môi", value=False)
                    delta_exp_value = 0.0
                    if delta_exp_enabled:
                        delta_exp_value = st.slider(
                            "Giá trị offset:",
                            min_value=-0.2, max_value=0.2, value=0.0, step=0.01
                        )
                else:
                    exp_components = None
                    exp_scale = 1.0
                    pose_scale = 1.0
                    delta_exp_enabled = False
                    delta_exp_value = 0.0
            
            # AI Model selection
            ai_model = st.selectbox(
                "Mô hình AI:",
                options=["Mô hình mặc định", "Mô hình tối ưu hóa"],
                help="Chọn mô hình AI để tạo video",
                index=0
            )
            
            # Submit button
            submitted = st.button(
                "🚀 Tạo Video MC",
                use_container_width=True,
                type="primary",
                disabled=st.session_state.processing
            )
        
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
                metrics_container.info("Nhấn nút 'Tạo Video MC' để bắt đầu xử lý...")
                
                # Preview chỉ hiển thị nếu không đang trong quá trình xử lý hoặc hoàn thành
                if mc_file or ('mc_path' in locals() and mc_path):
                    preview_container = log_container.container()
                    preview_container.subheader("Xem trước MC")
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
        
        # Xử lý khi submit
        if submitted and not st.session_state.processing:
            # Xác định đường dẫn các files
            mc_path_final = mc_file if mc_file else (mc_path if 'mc_path' in locals() and mc_path else None)
            bg_path_final = bg_file if bg_file else (bg_path if 'bg_path' in locals() and bg_path else None)
            audio_path_final = audio_file if audio_file else (audio_path if 'audio_path' in locals() and audio_path else None)
            
            if ((mc_path_final and bg_path_final) and (audio_path_final or text_prompt)):
                st.session_state.processing = True
                st.session_state.process_start_time = time.time()
                st.session_state.logs = ["Bắt đầu quá trình xử lý video MC..."]
                
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
                cancel_event = threading.Event()
                
                # Chuẩn bị containers cho handler
                ui_containers = {
                    'status': status_container,
                    'progress': progress,
                    'log_content': log_content if show_logs else None,
                    'metrics': metrics_container
                }
                
                # Chuẩn bị dữ liệu để truyền cho thread
                tts_service_val = tts_service if audio_source == "Tạo từ văn bản" else "Edge TTS"
                tts_voice_val = tts_voice if audio_source == "Tạo từ văn bản" else "vi-VN-NamMinhNeural"
                tts_speed_val = tts_speed if audio_source == "Tạo từ văn bản" and tts_service != "GPT-4o-mini-TTS" else 1.2
                tts_instructions_val = tts_instructions if audio_source == "Tạo từ văn bản" and tts_service == "GPT-4o-mini-TTS" else ""
                
                # Khởi chạy thread xử lý
                thread = threading.Thread(
                    target=process_video,
                    args=(
                        st.session_state.workflow_steps,
                        mc_path_final, bg_path_final, audio_path_final, text_prompt,
                        temp_dir, msg_queue, cancel_event, editor, timestamp,
                        tts_service_val, tts_voice_val, tts_speed_val,
                    ),
                    kwargs={
                        'tts_instructions_val': tts_instructions_val,
                        'position_val': position,
                        'scale_val': scale,
                        'quality_val': quality,
                        'ai_model_val': ai_model,
                        'vad_alpha': vad_alpha,
                        'exp_components': exp_components,
                        'exp_scale': exp_scale,
                        'pose_scale': pose_scale,
                        'delta_exp_enabled': delta_exp_enabled,
                        'delta_exp_value': delta_exp_value,
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
                                time.sleep(0.5)
                                st.rerun()
                        except queue.Empty:
                            time.sleep(0.1)
                        
                        time.sleep(0.05)
                except Exception as e:
                    st.error(f"Lỗi UI: {str(e)}")
                    st.session_state.processing = False
            else:
                st.error("Vui lòng chọn đầy đủ: MC, video nền, và audio (hoặc nhập văn bản)")
    
    # === Tab 1: Tạo Video Khuôn Mặt AI ===
    with tabs[1]:
        st.subheader("🎭 Tạo Video Khuôn Mặt Nói với AI")
        st.write("Chuyển đổi ảnh hoặc video MC tĩnh thành video với khả năng nói theo audio")
        
        ai_col1, ai_col2 = st.columns([3, 2])
        
        with ai_col1:
            st.subheader("📁 Tải lên files đầu vào")
            
            # MC uploader
            ai_mc_file = st.file_uploader(
                "Tải lên Ảnh/Video MC",
                type=["png", "jpg", "jpeg", "mp4"],
                key="ai_mc_file"
            )
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
            ai_audio_source = st.radio(
                "Nguồn audio:",
                ["Upload file", "Tạo từ văn bản"],
                horizontal=True,
                key="ai_audio_source"
            )
            
            if ai_audio_source == "Upload file":
                ai_audio_file = st.file_uploader(
                    "Tải lên Audio thoại",
                    type=["wav", "mp3"],
                    key="ai_audio_file"
                )
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
            else:
                ai_audio_file = None
                ai_audio_path = None
                ai_text_prompt = st.text_area(
                    "Nhập văn bản thoại:",
                    height=150,
                    key="ai_text_prompt",
                    placeholder="Nhập nội dung bạn muốn MC nói..."
                )
                
                # TTS settings cho AI mode
                if ai_text_prompt:
                    with st.expander("🎙️ Cài đặt TTS", expanded=True):
                        ai_tts_service = st.selectbox(
                            "Dịch vụ TTS:",
                            options=["Edge TTS", "OpenAI TTS", "GPT-4o-mini-TTS"],
                            index=2,
                            key="ai_tts_service"
                        )
                        
                        if ai_tts_service == "Edge TTS":
                            ai_tts_voice = st.selectbox(
                                "Giọng đọc:",
                                options=["vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"],
                                key="ai_tts_voice"
                            )
                            ai_tts_speed = st.slider("Tốc độ:", 0.8, 1.5, 1.2, 0.1, key="ai_tts_speed")
                            ai_tts_instructions = ""
                        elif ai_tts_service == "OpenAI TTS":
                            ai_tts_voice = st.selectbox(
                                "Giọng đọc:",
                                options=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                                key="ai_tts_voice"
                            )
                            ai_tts_speed = st.slider("Tốc độ:", 0.8, 1.5, 1.2, 0.1, key="ai_tts_speed")
                            ai_tts_instructions = ""
                        else:
                            ai_tts_voice = st.selectbox(
                                "Giọng đọc:",
                                options=["Ash", "Ballad", "Coral", "Echo", "Fable", "Onyx", "Nova", "Sage", "Shimmer", "Verse"],
                                index=8,
                                key="ai_tts_voice"
                            )
                            ai_tts_speed = 1.2
                            ai_tts_instructions = st.text_area(
                                "Hướng dẫn giọng điệu:",
                                value="Tone: Tự nhiên, trôi chảy\nEmotion: Nhiệt tình, tự tin\nDelivery: Rõ ràng, nhịp độ vừa phải",
                                height=80,
                                key="ai_tts_instructions"
                            )
            
            # AI Model settings
            with st.expander("🤖 Cài đặt mô hình AI", expanded=False):
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
                    key="ai_quality",
                    format_func=lambda x: {"low": "Thấp", "medium": "Trung bình", "high": "Cao"}[x]
                )
            
            # Tùy chỉnh khẩu hình AI
            with st.expander("🗣️ Tùy chỉnh khẩu hình", expanded=False):
                ai_vad_alpha = st.slider(
                    "Mức độ chuyển động môi:",
                    min_value=0.0, max_value=1.0, value=1.0, step=0.05,
                    key="ai_vad_alpha",
                    help="Giá trị thấp hơn sẽ làm giảm chuyển động môi"
                )
                
                ai_mouth_advanced = st.checkbox("Tùy chọn nâng cao", value=False, key="ai_mouth_advanced")
                if ai_mouth_advanced:
                    ai_exp_components = st.multiselect(
                        "Thành phần biểu cảm:",
                        options=["exp", "pitch", "yaw", "roll", "t"],
                        default=["exp", "pitch", "yaw", "roll", "t"],
                        key="ai_exp_components"
                    )
                    
                    ai_exp_scale = st.slider(
                        "Tỷ lệ biểu cảm miệng:",
                        min_value=0.5, max_value=1.5, value=1.0, step=0.1,
                        key="ai_exp_scale"
                    )
                    
                    ai_pose_scale = st.slider(
                        "Tỷ lệ chuyển động đầu:",
                        min_value=0.5, max_value=1.5, value=1.0, step=0.1,
                        key="ai_pose_scale"
                    )
                    
                    ai_delta_exp_enabled = st.checkbox("Thêm offset biểu cảm môi", value=False, key="ai_delta_exp_enabled")
                    ai_delta_exp_value = 0.0
                    if ai_delta_exp_enabled:
                        ai_delta_exp_value = st.slider(
                            "Giá trị offset:",
                            min_value=-0.2, max_value=0.2, value=0.0, step=0.01,
                            key="ai_delta_exp_value"
                        )
                else:
                    ai_exp_components = None
                    ai_exp_scale = 1.0
                    ai_pose_scale = 1.0
                    ai_delta_exp_enabled = False
                    ai_delta_exp_value = 0.0
            
            # Submit button
            ai_submitted = st.button(
                "🚀 Tạo Video Khuôn Mặt Nói",
                use_container_width=True,
                type="primary",
                key="ai_create_button"
            )
        
        with ai_col2:
            # Cột hiển thị tiến trình và kết quả
            ai_elapsed_time_container = st.empty()
            ai_status_container = st.empty()
            ai_progress_container = st.empty()
            ai_metrics_container = st.container()
            ai_cancel_container = st.empty()
            ai_result_container = st.container()
            
            if st.session_state.processing:
                ai_status_container.subheader("⏳ Đang xử lý...")
                
                if st.session_state.process_start_time:
                    elapsed = time.time() - st.session_state.process_start_time
                    ai_elapsed_time_container.caption(f"Thời gian xử lý: {int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
                
                progress = ai_progress_container.progress(0)
                cancel_button = ai_cancel_container.button("🛑 Hủy xử lý", key="ai_cancel_processing", use_container_width=True)
                
            elif st.session_state.complete and st.session_state.output_file:
                ai_status_container.subheader("✅ Đã hoàn thành!")
                
                output_file = st.session_state.output_file
                if Path(output_file).exists():
                    ai_metrics_container.video(output_file)
                    
                    file_stats = Path(output_file).stat()
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
                ai_status_container.subheader("Trạng thái")
                ai_metrics_container.info("Nhấn nút 'Tạo Video Khuôn Mặt Nói' để bắt đầu xử lý...")
        
        # Xử lý khi nhấn nút tạo video AI
        if ai_submitted and not st.session_state.processing:
            # Xác định đường dẫn files
            mc_path_final = ai_mc_file if ai_mc_file else (ai_mc_path if 'ai_mc_path' in locals() and ai_mc_path else None)
            audio_path_final = ai_audio_file if ai_audio_file else (ai_audio_path if 'ai_audio_path' in locals() and ai_audio_path else None)
            
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
                        mc_path_final, None, audio_path_final, ai_text_prompt,
                        temp_dir, msg_queue, cancel_event, editor, timestamp,
                        tts_service_val, tts_voice_val, tts_speed_val,
                    ),
                    kwargs={
                        'tts_instructions_val': tts_instructions_val,
                        'ai_model_val': ai_model,
                        'quality_val': ai_quality,
                        'vad_alpha': ai_vad_alpha,
                        'exp_components': ai_exp_components,
                        'exp_scale': ai_exp_scale,
                        'pose_scale': ai_pose_scale,
                        'delta_exp_enabled': ai_delta_exp_enabled,
                        'delta_exp_value': ai_delta_exp_value,
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
                        
                        time.sleep(0.05)
                except Exception as e:
                    st.error(f"Lỗi UI: {str(e)}")
                    st.session_state.processing = False
            else:
                st.error("Vui lòng tải lên cả MC và audio (hoặc nhập văn bản)")
    
    # === Tab 2: Text-to-Speech ===
    with tabs[2]:
        st.subheader("🎙️ Text-to-Speech - Tạo giọng nói từ văn bản")
        st.write("Tạo và nghe thử các giọng nói khác nhau từ văn bản")
        
        tts_col1, tts_col2 = st.columns([2, 1])
        
        with tts_col1:
            # Text input
            tts_text = st.text_area(
                "📝 Nhập văn bản:",
                height=150,
                placeholder="Nhập nội dung bạn muốn chuyển thành giọng nói...",
                key="tts_standalone_text"
            )
            
            # Service selection
            tts_service_standalone = st.selectbox(
                "🔧 Dịch vụ TTS:",
                options=["Edge TTS", "OpenAI TTS", "GPT-4o-mini-TTS"],
                index=2,
                key="tts_service_standalone"
            )
            
            # Voice and settings based on service
            if tts_service_standalone == "Edge TTS":
                st.markdown("### 🎤 Cài đặt Edge TTS")
                tts_voice_standalone = st.selectbox(
                    "Giọng đọc:",
                    options=["vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"],
                    key="tts_voice_standalone"
                )
                tts_speed_standalone = st.slider(
                    "Tốc độ đọc:",
                    min_value=0.8, max_value=1.5, value=1.2, step=0.1,
                    key="tts_speed_standalone"
                )
                tts_instructions_standalone = ""
                
                # Voice descriptions
                voice_info = {
                    "vi-VN-NamMinhNeural": "Giọng nam trẻ, rõ ràng, phù hợp cho nội dung chính thức",
                    "vi-VN-HoaiMyNeural": "Giọng nữ ấm áp, thân thiện, phù hợp cho nội dung giáo dục"
                }
                st.caption(f"ℹ️ {voice_info[tts_voice_standalone]}")
                
            elif tts_service_standalone == "OpenAI TTS":
                st.markdown("### 🎤 Cài đặt OpenAI TTS")
                tts_voice_standalone = st.selectbox(
                    "Giọng đọc:",
                    options=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                    key="tts_voice_standalone"
                )
                tts_speed_standalone = st.slider(
                    "Tốc độ đọc:",
                    min_value=0.8, max_value=1.5, value=1.2, step=0.1,
                    key="tts_speed_standalone"
                )
                tts_instructions_standalone = ""
                
                # Voice descriptions for OpenAI
                openai_voice_info = {
                    "alloy": "Giọng trung tính, cân bằng",
                    "echo": "Giọng nam trẻ",
                    "fable": "Giọng nam trưởng thành",
                    "onyx": "Giọng nam trầm",
                    "nova": "Giọng nữ trẻ",
                    "shimmer": "Giọng nữ tươi sáng"
                }
                st.caption(f"ℹ️ {openai_voice_info[tts_voice_standalone]}")
                
            else:  # GPT-4o-mini-TTS
                st.markdown("### 🎤 Cài đặt GPT-4o-mini-TTS")
                tts_voice_standalone = st.selectbox(
                    "Giọng đọc:",
                    options=["Ash", "Ballad", "Coral", "Echo", "Fable", "Onyx", "Nova", "Sage", "Shimmer", "Verse"],
                    index=8,  # Shimmer
                    key="tts_voice_standalone"
                )
                
                # Hiển thị mô tả giọng nói
                st.caption(f"ℹ️ **{tts_voice_standalone}**: {VOICE_DESCRIPTIONS.get(tts_voice_standalone, '')}")
                
                tts_speed_standalone = 1.2  # Fixed for GPT-4o
                
                # Khởi tạo giá trị preset nếu chưa có
                if 'tts_instructions_preset' not in st.session_state:
                    st.session_state.tts_instructions_preset = "Tone: Tự nhiên, trôi chảy, chuyên nghiệp\nEmotion: Nhiệt tình, tự tin\nDelivery: Rõ ràng, nhịp độ vừa phải, nhấn mạnh từ khóa quan trọng"
                
                tts_instructions_standalone = st.text_area(
                    "🎭 Hướng dẫn về giọng điệu:",
                    value=st.session_state.tts_instructions_preset,
                    height=120,
                    key="tts_instructions_standalone",
                    help="Mô tả tông giọng, cảm xúc và cách truyền đạt mong muốn"
                )
                
                # Cập nhật preset khi người dùng thay đổi
                if tts_instructions_standalone != st.session_state.tts_instructions_preset:
                    st.session_state.tts_instructions_preset = tts_instructions_standalone
                
                # Instruction templates
                with st.expander("📋 Mẫu hướng dẫn giọng điệu", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🎤 Giọng diễn thuyết:**")
                        if st.button("Sử dụng", key="preset_speech"):
                            st.session_state.tts_instructions_preset = "Tone: Đĩnh đạc, trang trọng, đầy tự tin\nEmotion: Nhiệt huyết, quyết đoán\nDelivery: Nhịp độ vừa phải với các ngắt quãng, nhấn mạnh từ khóa quan trọng"
                            st.rerun()
                        
                        st.markdown("**💼 Giọng thuyết trình:**")
                        if st.button("Sử dụng", key="preset_presentation"):
                            st.session_state.tts_instructions_preset = "Tone: Chuyên nghiệp, rõ ràng, tự tin\nEmotion: Tập trung, nghiêm túc\nDelivery: Nhịp độ đều đặn, phát âm rõ ràng từng từ"
                            st.rerun()
                    
                    with col2:
                        st.markdown("**🤝 Giọng tư vấn:**")
                        if st.button("Sử dụng", key="preset_consulting"):
                            st.session_state.tts_instructions_preset = "Tone: Ấm áp, thân thiện, đáng tin cậy\nEmotion: Thấu hiểu, quan tâm\nDelivery: Nhẹ nhàng, rõ ràng, tạo cảm giác an tâm"
                            st.rerun()
                        
                        st.markdown("**📺 Giọng quảng cáo:**")
                        if st.button("Sử dụng", key="preset_ads"):
                            st.session_state.tts_instructions_preset = "Tone: Sôi nổi, cuốn hút, năng động\nEmotion: Phấn khích, hào hứng\nDelivery: Nhanh, đầy năng lượng, với cường độ tăng dần"
                            st.rerun()
            
            # Action buttons
            col_preview, col_download = st.columns(2)
            
            with col_preview:
                preview_button = st.button(
                    "🔊 Nghe thử giọng nói",
                    use_container_width=True,
                    disabled=not tts_text.strip()
                )
            
            with col_download:
                generate_button = st.button(
                    "💾 Tạo và tải xuống",
                    use_container_width=True,
                    type="primary",
                    disabled=not tts_text.strip()
                )
        
        with tts_col2:
            st.subheader("🎵 Kết quả")
            
            # Preview area
            preview_message = st.empty()
            preview_audio = st.empty()
            
            # Handle preview button
            if preview_button and tts_text.strip():
                preview_message.info("⏳ Đang tạo mẫu giọng nói...")
                
                if tts_service_standalone == "GPT-4o-mini-TTS" and openai_client:
                    try:
                        # Use asyncio for GPT-4o preview
                        audio_bytes = asyncio.run(preview_audio_tts(
                            tts_text[:200],  # Limit preview length
                            tts_instructions_standalone,
                            tts_voice_standalone,
                            preview_message
                        ))
                        
                        if audio_bytes:
                            preview_message.success("✅ Tạo mẫu giọng nói thành công!")
                            preview_audio.audio(audio_bytes, format="audio/mp3")
                    except Exception as e:
                        preview_message.error(f"Lỗi: {str(e)}")
                        
                else:
                    try:
                        # Use VideoEditor for other services
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
                            temp_path = temp.name
                        
                        tts_service = "edge" if tts_service_standalone == "Edge TTS" else "openai"
                        success, error = editor.generate_audio_from_text(
                            tts_text[:200],  # Limit preview length
                            temp_path,
                            service=tts_service,
                            voice=tts_voice_standalone,
                            speed=tts_speed_standalone
                        )
                        
                        if success and os.path.exists(temp_path):
                            preview_message.success("✅ Tạo mẫu giọng nói thành công!")
                            with open(temp_path, "rb") as f:
                                audio_bytes = f.read()
                            preview_audio.audio(audio_bytes, format="audio/mp3")
                            os.unlink(temp_path)
                        else:
                            preview_message.error(f"Lỗi: {error}")
                            
                    except Exception as e:
                        preview_message.error(f"Lỗi: {str(e)}")
            
            # Handle generate and download
            if generate_button and tts_text.strip():
                with st.spinner("⏳ Đang tạo file audio..."):
                    try:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        
                        if tts_service_standalone == "GPT-4o-mini-TTS" and openai_client:
                            # Generate with GPT-4o-mini-TTS
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
                                temp_path = temp.name
                            
                            success = asyncio.run(generate_gpt4o_tts(
                                tts_text,
                                temp_path,
                                tts_instructions_standalone,
                                tts_voice_standalone
                            ))
                            
                            if success and os.path.exists(temp_path):
                                with open(temp_path, "rb") as f:
                                    audio_bytes = f.read()
                                
                                st.download_button(
                                    "💾 Tải xuống audio",
                                    audio_bytes,
                                    file_name=f"tts_gpt4o_{timestamp}.mp3",
                                    mime="audio/mp3",
                                    use_container_width=True
                                )
                                
                                st.audio(audio_bytes, format="audio/mp3")
                                os.unlink(temp_path)
                            else:
                                st.error("Lỗi khi tạo audio với GPT-4o-mini-TTS")
                                
                        else:
                            # Generate with other services
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
                                temp_path = temp.name
                            
                            tts_service = "edge" if tts_service_standalone == "Edge TTS" else "openai"
                            success, error = editor.generate_audio_from_text(
                                tts_text,
                                temp_path,
                                service=tts_service,
                                voice=tts_voice_standalone,
                                speed=tts_speed_standalone
                            )
                            
                            if success and os.path.exists(temp_path):
                                with open(temp_path, "rb") as f:
                                    audio_bytes = f.read()
                                
                                st.download_button(
                                    "💾 Tải xuống audio",
                                    audio_bytes,
                                    file_name=f"tts_{tts_service}_{timestamp}.mp3",
                                    mime="audio/mp3",
                                    use_container_width=True
                                )
                                
                                st.audio(audio_bytes, format="audio/mp3")
                                os.unlink(temp_path)
                            else:
                                st.error(f"Lỗi: {error}")
                                
                    except Exception as e:
                        st.error(f"Lỗi khi tạo audio: {str(e)}")
            
            # TTS Info
            st.markdown("---")
            st.markdown("### 📊 Thông tin TTS")
            if tts_service_standalone == "GPT-4o-mini-TTS":
                st.info("🎭 **GPT-4o-mini-TTS**: Giọng nói biểu cảm với AI")
                st.markdown("""
                **Tính năng:**
                - ✅ 10 giọng nói đa dạng
                - ✅ Tùy chỉnh giọng điệu
                - ✅ Biểu cảm tự nhiên
                - ✅ Chất lượng cao
                """)
            elif tts_service_standalone == "Edge TTS":
                st.info("🎤 **Edge TTS**: Miễn phí, chất lượng tốt")
                st.markdown("""
                **Tính năng:**
                - ✅ Miễn phí hoàn toàn
                - ✅ Hỗ trợ tiếng Việt
                - ✅ Điều chỉnh tốc độ
                - ✅ Chất lượng ổn định
                """)
            else:
                st.info("🤖 **OpenAI TTS**: Chất lượng cao, đa ngôn ngữ")
                st.markdown("""
                **Tính năng:**
                - ✅ Chất lượng premium
                - ✅ 6 giọng nói khác nhau
                - ✅ Hỗ trợ nhiều ngôn ngữ
                - ✅ Điều chỉnh tốc độ
                """)
    
    # === Tab 3: Lịch sử ===
    with tabs[3]:
        st.subheader("📋 Lịch sử video đã tạo")
        
        # Cập nhật lịch sử từ thư mục
        update_history_from_folder()
        
        if not st.session_state.history:
            st.info("📝 Chưa có video nào được tạo.")
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
                        st.write(f"**Kích thước:** {item['size']:.1f} MB")
                        st.write(f"**Thời gian tạo:** {item['created'].strftime('%Y-%m-%d %H:%M:%S')}")
                        
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
    
    # === Tab 4: Cài đặt ===
    with tabs[4]:
        st.subheader("⚙️ Cài đặt API Keys")
        st.write("Cấu hình các API keys cần thiết cho các tính năng nâng cao")
        
        # Khởi tạo OpenAI client nếu chưa có
        initialize_openai_client()
        
        # OpenAI API Key Section
        with st.expander("🤖 OpenAI API Key", expanded=True):
            st.markdown("""
            **OpenAI API Key** được sử dụng cho:
            - 🎭 GPT-4o-mini-TTS (Text-to-Speech chất lượng cao)
            - 🎤 OpenAI TTS (Text-to-Speech chuyên nghiệp)
            
            💡 **Cách lấy API Key:**
            1. Truy cập [OpenAI Platform](https://platform.openai.com/account/api-keys)
            2. Đăng nhập và tạo API key mới
            3. Copy và paste vào ô bên dưới
            """)
            
            # Hiển thị thông tin nguồn API key
            if st.session_state.get('api_key_source') == 'environment':
                st.info("ℹ️ **API Key được tự động tải từ cấu hình Notebook** - Không cần nhập lại!")
            
            # API Key input - disable nếu đã có từ environment
            api_key_input = st.text_input(
                "OpenAI API Key:",
                value=st.session_state.openai_api_key if st.session_state.get('api_key_source') != 'environment' else "••••••••••••••••••••••••••••••••••••••••••••••••••••",
                type="password",
                placeholder="sk-proj-..." if st.session_state.get('api_key_source') != 'environment' else "Đã cấu hình từ Notebook",
                help="API key được tự động tải từ cell cấu hình trong Notebook" if st.session_state.get('api_key_source') == 'environment' else "Nhập OpenAI API key để sử dụng GPT-4o-mini-TTS và OpenAI TTS",
                disabled=st.session_state.get('api_key_source') == 'environment'
            )
            
            # Buttons row - chỉ hiển thị khi không phải từ environment
            if st.session_state.get('api_key_source') != 'environment':
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    save_button = st.button("💾 Lưu", use_container_width=True)
                
                with col2:
                    test_button = st.button("🧪 Kiểm tra", use_container_width=True, disabled=not (api_key_input and api_key_input.strip()))
                
                with col3:
                    clear_button = st.button("🗑️ Xóa", use_container_width=True)
                
                # Handle buttons
                if save_button and api_key_input and api_key_input.strip():
                    st.session_state.openai_api_key = api_key_input.strip()
                    st.session_state.openai_api_status = 'not_tested'
                    st.session_state.api_key_source = 'manual'
                    initialize_openai_client()
                    st.success("✅ Đã lưu OpenAI API key!")
                    st.rerun()
                
                if test_button and api_key_input and api_key_input.strip():
                    st.session_state.openai_api_status = 'testing'
                    with st.spinner("🧪 Đang kiểm tra API key..."):
                        try:
                            is_valid, message = asyncio.run(test_openai_api_key(api_key_input.strip()))
                            if is_valid:
                                st.session_state.openai_api_status = 'valid'
                                st.success(f"✅ {message}")
                                st.session_state.openai_api_key = api_key_input.strip()
                                st.session_state.api_key_source = 'manual'
                                initialize_openai_client()
                            else:
                                st.session_state.openai_api_status = 'invalid'
                                st.error(f"❌ {message}")
                        except Exception as e:
                            st.session_state.openai_api_status = 'invalid'
                            st.error(f"❌ Lỗi kiểm tra: {str(e)}")
                    st.rerun()
                
                if clear_button:
                    st.session_state.openai_api_key = ''
                    st.session_state.openai_api_status = 'not_tested'
                    st.session_state.api_key_source = 'manual'
                    initialize_openai_client()
                    st.info("🗑️ Đã xóa API key")
                    st.rerun()
            else:
                # Chỉ hiển thị nút test connection cho API key từ environment
                if st.button("🔗 Test kết nối với API key từ Notebook", use_container_width=True):
                    with st.spinner("🧪 Đang kiểm tra API key từ Notebook..."):
                        try:
                            is_valid, message = asyncio.run(test_openai_api_key(st.session_state.openai_api_key))
                            if is_valid:
                                st.session_state.openai_api_status = 'valid'
                                st.success(f"✅ {message}")
                                initialize_openai_client()
                            else:
                                st.session_state.openai_api_status = 'invalid'
                                st.error(f"❌ {message}")
                        except Exception as e:
                            st.session_state.openai_api_status = 'invalid'
                            st.error(f"❌ Lỗi kiểm tra: {str(e)}")
                    st.rerun()
            
            # Status indicator
            if st.session_state.openai_api_key:
                status = st.session_state.openai_api_status
                api_source = st.session_state.get('api_key_source', 'manual')
                
                if status == 'valid':
                    source_text = "từ Notebook" if api_source == 'environment' else "thủ công"
                    st.success(f"✅ API key đã được xác thực và đang hoạt động (nguồn: {source_text})")
                elif status == 'invalid':
                    st.error("❌ API key không hợp lệ hoặc có lỗi")
                elif status == 'testing':
                    st.info("🧪 Đang kiểm tra API key...")
                else:
                    if api_source == 'environment':
                        st.info("ℹ️ API key từ Notebook - Nhấn 'Test kết nối' để xác thực.")
                    else:
                        st.warning("⚠️ API key chưa được kiểm tra. Nhấn 'Kiểm tra' để xác thực.")
            else:
                st.info("ℹ️ Chưa có API key. Một số tính năng sẽ không khả dụng.")
        
        # Thông tin API Usage
        if openai_client:
            with st.expander("📊 Thông tin sử dụng API", expanded=False):
                st.markdown("""
                **Lưu ý về chi phí:**
                - GPT-4o-mini-TTS: ~$0.150 / 1M ký tự
                - OpenAI TTS: ~$15.00 / 1M ký tự
                - Một đoạn văn 1000 từ ≈ 5000 ký tự
                
                **Khuyến nghị:**
                - Sử dụng Edge TTS (miễn phí) cho mục đích thử nghiệm
                - GPT-4o-mini-TTS cho chất lượng cao với chi phí hợp lý
                - OpenAI TTS cho chất lượng premium
                """)
                
                # Test connection button
                if st.button("🔗 Test kết nối", key="test_connection"):
                    with st.spinner("Đang test kết nối..."):
                        try:
                            is_valid, message = asyncio.run(test_openai_api_key(st.session_state.openai_api_key))
                            if is_valid:
                                st.success(f"✅ {message}")
                            else:
                                st.error(f"❌ {message}")
                        except Exception as e:
                            st.error(f"❌ Lỗi: {str(e)}")
        
        # API Keys Security
        with st.expander("🔒 Bảo mật API Keys", expanded=False):
            st.markdown("""
            **⚠️ Lưu ý bảo mật quan trọng:**
            
            1. **Không chia sẻ API keys** với người khác
            2. **Xóa API keys** khi không sử dụng nữa
            3. **Giám sát usage** thường xuyên trên OpenAI Platform
            4. **Đặt limits** cho API usage để tránh chi phí bất ngờ
            
            **🛡️ API keys được lưu trong:**
            - Session memory của Streamlit (tạm thời)
            - Không được lưu vào file hoặc database
            - Sẽ bị xóa khi tắt ứng dụng
            """)
    
    # === Tab 5: Hướng dẫn ===
    with tabs[5]:
        st.subheader("❓ Hướng dẫn sử dụng")
        
        with st.expander("🚀 Bắt đầu nhanh", expanded=True):
            st.markdown("""
            ### 🎬 Tạo Video MC:
            1. **📁 Tải lên file MC**: Ảnh hoặc video có nhân vật nói
            2. **🎞️ Tải lên video nền**: Video nền cho MC
            3. **🎵 Chọn nguồn audio**: Tải lên file audio hoặc tạo từ văn bản
            4. **⚙️ Điều chỉnh cài đặt**: Vị trí và kích thước MC
            5. **🚀 Nhấn "Tạo Video MC"**: Chờ xử lý và tải xuống kết quả
            
            ### 🎭 Tạo Video Khuôn Mặt AI:
            1. **📁 Tải lên file MC**: Ảnh hoặc video khuôn mặt
            2. **🎵 Chọn nguồn audio**: Audio hoặc văn bản
            3. **🤖 Chọn mô hình AI**: Mặc định hoặc tối ưu hóa
            4. **🗣️ Tùy chỉnh khẩu hình**: Điều chỉnh chuyển động môi
            5. **🚀 Nhấn "Tạo Video Khuôn Mặt Nói"**: Chờ kết quả
            """)
        
        with st.expander("🎙️ Text-to-Speech", expanded=False):
            st.markdown("""
            ### Dịch vụ TTS có sẵn:
            
            **🤖 GPT-4o-mini-TTS** (Khuyến nghị):
            - 10 giọng nói đa dạng
            - Tùy chỉnh giọng điệu chi tiết
            - Biểu cảm tự nhiên
            
            **🎤 Edge TTS** (Miễn phí):
            - Hỗ trợ tiếng Việt tốt
            - Không cần API key
            - Chất lượng ổn định
            
            **🔊 OpenAI TTS** (Premium):
            - Chất lượng cao
            - 6 giọng nói khác nhau
            - Cần API key
            """)
        
        with st.expander("⚙️ Tính năng nâng cao", expanded=False):
            st.markdown("""
            ### 🗣️ Điều khiển khẩu hình:
            - **Mức độ chuyển động môi**: Điều chỉnh độ mạnh của chuyển động môi
            - **Thành phần biểu cảm**: Chọn các phần của khuôn mặt để animate
            - **Tỷ lệ biểu cảm**: Điều chỉnh cường độ biểu cảm
            - **Offset biểu cảm**: Thay đổi hình dáng môi mặc định
            
            ### 🎛️ Cài đặt video:
            - **Vị trí MC**: 5 vị trí khác nhau trên video
            - **Tự động scale**: Tính toán kích thước phù hợp
            - **Chất lượng**: Low/Medium/High
            - **Mô hình AI**: Mặc định hoặc tối ưu hóa
            
            ### 🔄 Quy trình xử lý:
            - Bật/tắt từng bước xử lý
            - Tùy chỉnh workflow theo nhu cầu
            - Theo dõi tiến trình real-time
            """)
        
        with st.expander("🔧 Xử lý sự cố", expanded=False):
            st.markdown("""
            ### ❌ Vấn đề thường gặp:
            
            **🐌 Xử lý chậm:**
            - Giảm chất lượng xuống "Low"
            - Sử dụng file input nhỏ hơn
            - Kiểm tra GPU memory
            
            **❌ Lỗi khi tạo video:**
            - Kiểm tra định dạng file đúng
            - Đảm bảo files không bị lỗi
            - Thử restart runtime
            
            **❌ Lỗi TTS:**
            - Kiểm tra API keys (nếu dùng OpenAI)
            - Thử dịch vụ Edge TTS
            - Kiểm tra kết nối internet
            
            **🔄 App bị đơ:**
            - Nhấn nút "Hủy xử lý"
            - Restart Streamlit
            - Kiểm tra logs để debug
            """)
        
        with st.expander("💡 Mẹo tối ưu", expanded=False):
            st.markdown("""
            ### 📸 Chuẩn bị file MC:
            - Khuôn mặt rõ ràng, nhìn thẳng
            - Nền đồng màu hoặc trong suốt
            - Ánh sáng đều, không bị tối
            - Độ phân giải ít nhất 512x512
            
            ### 🎞️ Video nền:
            - Định dạng MP4, tỷ lệ 16:9
            - Độ phân giải HD (1280x720) trở lên
            - Thời lượng phù hợp với audio
            - Nội dung phù hợp với chủ đề
            
            ### 🎵 Audio chất lượng:
            - File WAV hoặc MP3
            - Không có tiếng ồn nền
            - Giọng nói rõ ràng
            - Tốc độ nói vừa phải
            
            ### ⚡ Tối ưu hiệu suất:
            - Sử dụng GPU T4 hoặc cao hơn
            - Đóng các tab không cần thiết
            - Kiểm tra RAM còn trống
            - Sử dụng chất lượng Medium cho cân bằng
            """)

if __name__ == "__main__":
    main()
