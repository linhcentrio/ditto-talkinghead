import librosa
import math
import os
import numpy as np
import random
import torch
import pickle
import sys
from stream_pipeline_offline import StreamSDK

def seed_everything(seed):
    """Set random seeds for reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_pkl(pkl_path):
    """Load pickle file safely with error handling"""
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file {pkl_path}: {e}")
        return {}

def run(SDK: StreamSDK, audio_path: str, source_path: str, output_path: str, more_kwargs: str | dict = {}):
    """Run inference with support for more_kwargs parameter"""
    # Handle more_kwargs
    if isinstance(more_kwargs, str):
        more_kwargs = load_pkl(more_kwargs)
    
    # Extract setup and run kwargs
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})
    
    # Setup SDK with setup_kwargs
    SDK.setup(source_path, output_path, **setup_kwargs)
    
    # Load audio
    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)
    
    # Get parameters from run_kwargs
    fade_in = run_kwargs.get("fade_in", 1)
    fade_out = run_kwargs.get("fade_out", 1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    
    # Setup number of frames and control info
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)
    
    # Process audio based on mode
    online_mode = SDK.online_mode
    if online_mode:
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480
        for i in range(0, len(audio), chunksize[1] * 640):
            audio_chunk = audio[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
    else:
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
        SDK.close()
    
    # Combine audio and video
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    print(cmd)
    os.system(cmd)
    
    print(f"Output saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Import argparse within the main block to avoid potential issues
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run inference for audio-driven face animation")
    parser.add_argument("--data_root", type=str, default="./checkpoints/ditto_trt_Ampere_Plus", 
                        help="Path to trt data_root")
    parser.add_argument("--cfg_pkl", type=str, default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl", 
                        help="Path to cfg_pkl")
    parser.add_argument("--audio_path", type=str, required=True, 
                        help="Path to input audio file (.wav/.mp3)")
    parser.add_argument("--source_path", type=str, required=True, 
                        help="Path to input image/video file")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Path to output video file (.mp4)")
    parser.add_argument("--more_kwargs", type=str, 
                        help="Path to more_kwargs pickle file")
    parser.add_argument("--seed", type=int, default=1024, 
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        seed_everything(args.seed)
    
    # Initialize SDK within a try-except block to handle errors gracefully
    try:
        # Initialize SDK
        data_root = args.data_root   # model dir
        cfg_pkl = args.cfg_pkl     # cfg pkl
        SDK = StreamSDK(cfg_pkl, data_root)
        
        # Input args
        audio_path = args.audio_path    # .wav
        source_path = args.source_path   # video|image
        output_path = args.output_path   # .mp4
        
        # Process more_kwargs if provided
        more_kwargs = {}
        if args.more_kwargs:
            if os.path.exists(args.more_kwargs):
                more_kwargs = load_pkl(args.more_kwargs)
                print(f"Loaded more_kwargs from {args.more_kwargs}: {more_kwargs.keys()}")
            else:
                print(f"Warning: more_kwargs file not found: {args.more_kwargs}")
        
        # Run inference
        run(SDK, audio_path, source_path, output_path, more_kwargs)
        
    except Exception as e:
        # Handle exceptions gracefully
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
