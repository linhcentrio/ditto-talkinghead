import time

def get_stream_sdk(online_mode=False):
    """Chọn lớp StreamSDK phù hợp dựa trên chế độ online."""
    if online_mode:
        print(f"[{time.strftime('%H:%M:%S')}] Sử dụng chế độ xử lý ONLINE")
        from stream_pipeline_online import StreamSDK
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Sử dụng chế độ xử lý OFFLINE")
        from stream_pipeline_offline import StreamSDK
    return StreamSDK
