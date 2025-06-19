#!/usr/bin/env python
import os
import sys
import time
import uuid
import json
import logging
import shutil
import glob
import subprocess
from datetime import datetime
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query, Path, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union, Literal
import secrets
from celery import Celery
from celery.result import AsyncResult

# Basic configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("image_talking")

# Directories and paths
INPUT_DIR = "/home/image_talking/inputs"
OUTPUT_DIR = "/home/image_talking/outputs"
LOG_DIR = "/home/image_talking/logs"
AUDIO_DIR = os.path.join(INPUT_DIR, "audio")
SOURCE_DIR = os.path.join(INPUT_DIR, "source")

# Get the currently running Python interpreter path
PYTHON_PATH = "/home/image_talking/venv/bin/python"
INFERENCE_SCRIPT = "/home/image_talking/inference.py"

# Redis and Celery settings
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
API_KEY = os.getenv("API_KEY", secrets.token_hex(16))

# Default model paths
DEFAULT_DATA_ROOT = "/home/image_talking/checkpoints/trt_custom/"
DEFAULT_CFG_PKL = "/home/image_talking/checkpoints/cfg/v0.4_hubert_cfg_trt.pkl"

# Create necessary directories
for directory in [INPUT_DIR, OUTPUT_DIR, LOG_DIR, AUDIO_DIR, SOURCE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize FastAPI
app = FastAPI(title="Image Talking API")
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize Celery
redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
celery_app = Celery('image_talking', broker=redis_url, backend=redis_url)
celery_app.conf.worker_concurrency = 2

# Models
class InferenceRequest(BaseModel):
    audio_path: str = Field(..., description="Path to input audio file (.wav/.mp3)")
    source_path: str = Field(..., description="Path to input image/video file")
    output_path: Optional[str] = Field(None, description="Path to output video file (.mp4)")
    data_root: Optional[str] = Field(DEFAULT_DATA_ROOT, description="Path to trt data_root")
    cfg_pkl: Optional[str] = Field(DEFAULT_CFG_PKL, description="Path to cfg_pkl")
    more_kwargs: Optional[str] = Field(None, description="Path to more_kwargs pickle file")
    seed: Optional[int] = Field(1024, description="Random seed for reproducibility")

class TaskResponse(BaseModel):
    task_id: str = Field(..., description="Task ID")
    status: str = Field("pending", description="Task status")

# API security
async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True

# Celery task for processing
@celery_app.task(name="tasks.run_inference", bind=True)
def run_inference(self, audio_path, source_path, output_path=None, data_root=DEFAULT_DATA_ROOT, 
                 cfg_pkl=DEFAULT_CFG_PKL, more_kwargs=None, seed=1024):
    """Task for processing audio-driven face animation"""
    task_id = self.request.id
    log_file = os.path.join(LOG_DIR, f"task_{task_id}.log")
    
    # Configure logging for the task
    task_logger = logging.getLogger(f"task_{task_id}")
    task_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    task_logger.addHandler(file_handler)
    
    try:
        task_logger.info(f"Starting processing: audio={audio_path}, source={source_path}")
        
        # Check input files
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        # Setup output path if not provided
        if not output_path:
            audio_filename = os.path.basename(audio_path)
            source_filename = os.path.basename(source_path)
            audio_name = os.path.splitext(audio_filename)[0]
            source_name = os.path.splitext(source_filename)[0]
            output_path = os.path.join(OUTPUT_DIR, f"{source_name}_{audio_name}.mp4")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'Processing', 'audio_path': audio_path, 'source_path': source_path})
        
        # Prepare command
        cmd = [
            PYTHON_PATH, 
            INFERENCE_SCRIPT,
            "--data_root", data_root,
            "--cfg_pkl", cfg_pkl,
            "--audio_path", audio_path,
            "--source_path", source_path,
            "--output_path", output_path,
            "--seed", str(seed)
        ]
        
        # Add more_kwargs if provided
        if more_kwargs:
            cmd.extend(["--more_kwargs", more_kwargs])
        
        task_logger.info(f"Command: {' '.join(cmd)}")
        
        # Create marker for tracking
        temp_marker = os.path.join(OUTPUT_DIR, f"task_{task_id}_{int(time.time())}.marker")
        with open(temp_marker, 'w') as f:
            f.write(f"Audio: {audio_path}\nSource: {source_path}\nOutput: {output_path}")
        
        # Execute and track
        start_time = time.time()
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration = time.time() - start_time
        
        # Log results
        task_logger.info(f"Result: returncode={process.returncode}")
        task_logger.info(f"STDOUT: {process.stdout}")
        task_logger.info(f"STDERR: {process.stderr}")
        
        # Remove marker
        if os.path.exists(temp_marker):
            os.remove(temp_marker)
        
        # Handle errors
        if process.returncode != 0:
            task_logger.error(f"Error: {process.stderr}")
            return {'status': 'error', 'error': process.stderr, 'audio_path': audio_path, 'source_path': source_path}
        
        # Check if output exists
        if not os.path.exists(output_path):
            task_logger.warning(f"Output file does not exist: {output_path}")
            
            # Look for recent files
            cutoff_time = time.time() - 30
            recent_files = [
                f for f in glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")) 
                if os.path.isfile(f) and os.path.getctime(f) > cutoff_time
            ]
            
            if recent_files:
                newest_file = max(recent_files, key=os.path.getctime)
                shutil.move(newest_file, output_path)
                task_logger.info(f"Moved newly created file to {output_path}")
            else:
                return {'status': 'error', 'error': 'Output file not found'}
        
        # Final check
        if not os.path.exists(output_path):
            return {'status': 'error', 'error': 'Failed to create output file'}
        
        # Success
        file_size = os.path.getsize(output_path)
        human_size = f"{file_size / (1024*1024):.2f} MB"
        
        return {
            'status': 'success',
            'audio_path': audio_path,
            'source_path': source_path,
            'output_path': output_path,
            'static_url': f"/static/{os.path.basename(output_path)}",
            'output_size': human_size,
            'duration': f"{duration:.2f}s"
        }
        
    except Exception as e:
        import traceback
        task_logger.exception(f"Exception: {str(e)}")
        return {'status': 'error', 'error': str(e), 'stack_trace': traceback.format_exc()}
    finally:
        # Cleanup
        for handler in task_logger.handlers:
            handler.close()
        task_logger.removeHandler(handler)

# API Endpoints
@app.post("/inference", response_model=TaskResponse)
async def create_inference_task(request: InferenceRequest, api_key: bool = Depends(verify_api_key)):
    """Create a task for audio-driven face animation from paths"""
    if not os.path.exists(request.audio_path):
        raise HTTPException(status_code=404, detail=f"Audio file not found: {request.audio_path}")
    if not os.path.exists(request.source_path):
        raise HTTPException(status_code=404, detail=f"Source file not found: {request.source_path}")
    
    task = run_inference.delay(
        request.audio_path, 
        request.source_path, 
        request.output_path,
        request.data_root,
        request.cfg_pkl,
        request.more_kwargs,
        request.seed
    )
    
    logger.info(f"Created task {task.id} for audio {request.audio_path} and source {request.source_path}")
    return TaskResponse(task_id=task.id)

@app.post("/upload", response_model=TaskResponse)
async def upload_and_process(
    audio_file: UploadFile = File(...),
    source_file: UploadFile = File(...),
    data_root: Optional[str] = Form(DEFAULT_DATA_ROOT),
    cfg_pkl: Optional[str] = Form(DEFAULT_CFG_PKL),
    seed: Optional[int] = Form(1024),
    output_name: Optional[str] = Form(None),
    api_key: bool = Depends(verify_api_key)
):
    """Upload audio and source files and create processing task"""
    # Check audio file format
    audio_ext = os.path.splitext(audio_file.filename)[1].lower()
    allowed_audio_ext = ['.wav', '.mp3']
    if audio_ext not in allowed_audio_ext:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {audio_ext}")
    
    # Check source file format
    source_ext = os.path.splitext(source_file.filename)[1].lower()
    allowed_source_ext = ['.jpg', '.jpeg', '.png', '.mp4', '.mov', '.avi']
    if source_ext not in allowed_source_ext:
        raise HTTPException(status_code=400, detail=f"Unsupported source format: {source_ext}")
    
    # Generate unique IDs
    audio_id = str(uuid.uuid4())
    source_id = str(uuid.uuid4())
    
    # Save files
    audio_path = os.path.join(AUDIO_DIR, f"{audio_id}{audio_ext}")
    source_path = os.path.join(SOURCE_DIR, f"{source_id}{source_ext}")
    
    try:
        # Save audio file
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Save source file
        with open(source_path, "wb") as buffer:
            shutil.copyfileobj(source_file.file, buffer)
        
        # Determine output path
        if output_name:
            output_path = os.path.join(OUTPUT_DIR, output_name)
        else:
            output_path = os.path.join(OUTPUT_DIR, f"{source_id}_{audio_id}.mp4")
        
        # Create task
        task = run_inference.delay(
            audio_path, 
            source_path, 
            output_path,
            data_root,
            cfg_pkl,
            None,  # more_kwargs
            seed
        )
        
        logger.info(f"Uploaded {audio_file.filename} and {source_file.filename}, created task {task.id}")
        return TaskResponse(task_id=task.id)
    
    except Exception as e:
        # Cleanup if error
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists(source_path):
            os.remove(source_path)
        logger.exception(f"Error uploading files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str, api_key: bool = Depends(verify_api_key)):
    """Check task status"""
    try:
        task_result = AsyncResult(task_id, app=celery_app)
        
        response = {
            "task_id": task_id,
            "status": task_result.status,
        }
        
        # Add progress info
        if task_result.state == 'PROGRESS' and task_result.info:
            response.update(task_result.info)
        
        # Add result or error
        if task_result.successful():
            response['result'] = task_result.result
            # Check if file exists
            if 'output_path' in task_result.result:
                response['file_exists'] = os.path.exists(task_result.result['output_path'])
        elif task_result.failed():
            response['error'] = str(task_result.result)
        
        # Add log info
        log_file = os.path.join(LOG_DIR, f"task_{task_id}.log")
        if os.path.exists(log_file):
            response['log_file'] = log_file
        
        return response
    
    except Exception as e:
        logger.exception(f"Error checking task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/{task_id}/log")
async def get_task_log(task_id: str, lines: int = Query(100, ge=1, le=1000), api_key: bool = Depends(verify_api_key)):
    """Get task log"""
    log_file = os.path.join(LOG_DIR, f"task_{task_id}.log")
    
    if not os.path.exists(log_file):
        raise HTTPException(status_code=404, detail=f"Log not found: {task_id}")
    
    try:
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {"task_id": task_id, "log_content": "".join(last_lines)}
    
    except Exception as e:
        logger.exception(f"Error reading log {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/tasks/{task_id}")
async def revoke_task(task_id: str, terminate: bool = Query(False), api_key: bool = Depends(verify_api_key)):
    """Cancel pending or running task"""
    try:
        celery_app.control.revoke(task_id, terminate=terminate)
        return {"message": f"Cancelled task {task_id}", "terminate": terminate}
    except Exception as e:
        logger.exception(f"Error cancelling task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{file_type}")
async def list_files(file_type: Literal["audio", "source", "output"], pattern: Optional[str] = None, api_key: bool = Depends(verify_api_key)):
    """List files in the specified directory"""
    try:
        if file_type == "audio":
            dir_path = AUDIO_DIR
        elif file_type == "source":
            dir_path = SOURCE_DIR
        elif file_type == "output":
            dir_path = OUTPUT_DIR
        else:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        files = []
        for filename in os.listdir(dir_path):
            if pattern and pattern not in filename:
                continue
            
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path):
                file_stats = os.stat(file_path)
                file_info = {
                    "name": filename,
                    "path": file_path,
                    "size": file_stats.st_size,
                    "size_human": f"{file_stats.st_size / (1024*1024):.2f} MB",
                    "created": datetime.fromtimestamp(file_stats.st_ctime).isoformat()
                }
                
                if file_type == "output":
                    file_info["url"] = f"/static/{filename}"
                
                files.append(file_info)
        
        return {"files": files, "count": len(files)}
    
    except Exception as e:
        logger.exception(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{file_type}/{filename}")
async def download_file(file_type: Literal["audio", "source", "output"], filename: str = Path(...), api_key: bool = Depends(verify_api_key)):
    """Download file from the specified directory"""
    if file_type == "audio":
        dir_path = AUDIO_DIR
    elif file_type == "source":
        dir_path = SOURCE_DIR
    elif file_type == "output":
        dir_path = OUTPUT_DIR
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    file_path = os.path.join(dir_path, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    return FileResponse(path=file_path, filename=filename)

@app.delete("/files/{file_type}/{filename}")
async def delete_file(file_type: Literal["audio", "source", "output"], filename: str = Path(...), api_key: bool = Depends(verify_api_key)):
    """Delete file from the specified directory"""
    if file_type == "audio":
        dir_path = AUDIO_DIR
    elif file_type == "source":
        dir_path = SOURCE_DIR
    elif file_type == "output":
        dir_path = OUTPUT_DIR
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    file_path = os.path.join(dir_path, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    try:
        os.remove(file_path)
        return {"message": f"Deleted file {filename}"}
    except Exception as e:
        logger.exception(f"Error deleting file {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Start application
if __name__ == "__main__":
    # Setup argparse
    import argparse
    parser = argparse.ArgumentParser(description="Image Talking API Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run API")
    parser.add_argument("--port", type=int, default=8000, help="Port to run API")
    parser.add_argument("--worker", action="store_true", help="Run Celery worker")
    args = parser.parse_args()
    
    if args.worker:
        # Run Celery worker using the current Python interpreter
        os.system(f"{PYTHON_PATH} -m celery -A {os.path.basename(__file__).split('.')[0]}.celery_app worker --loglevel=info --concurrency=2")
    else:
        # Run FastAPI
        logger.info(f"Starting API at http://{args.host}:{args.port}")
        logger.info(f"API key: {API_KEY}")
        uvicorn.run(f"{os.path.basename(__file__).split('.')[0]}:app", host=args.host, port=args.port, log_level="info")
