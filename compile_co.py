# setup.py
import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# --- Cấu hình ---
# Thư mục chứa các tệp nguồn của bạn
# Để trống ('') nếu tệp setup.py nằm ở thư mục gốc
ROOT_DIR = ''

# Các tệp bạn muốn biên dịch (đường dẫn tương đối từ ROOT_DIR)
FILES_TO_COMPILE = [
    # Các tệp ở thư mục gốc (đã có từ trước)
    # 'inference.pyx',
    # 'stream_pipeline_offline.pyx',
    # 'stream_pipeline_online.pyx', # Giữ lại nếu bạn vẫn muốn biên dịch nó, dù inference.py không dùng trực tiếp
    # 'video_editor.pyx',
    # 'video_overlay.pyx',
    # 'run_streamlit_core.pyx',

    # --- CÁC TỆP MỚI ĐƯỢC BỔ SUNG TỪ YÊU CẦU ---
    # Tệp trong thư mục con core/atomic_components
    #'core/atomic_components/audio2motion.pyx',
    'core/atomic_components/avatar_registrar.pyx',
    'core/atomic_components/cfg.pyx',
    'core/atomic_components/condition_handler.pyx',
    'core/atomic_components/decode_f3d.pyx',
    'core/atomic_components/loader.pyx',
    'core/atomic_components/motion_stitch.pyx',
    'core/atomic_components/putback.pyx',
    'core/atomic_components/source2info.pyx',
    'core/atomic_components/warp_f3d.pyx',
    'core/atomic_components/wav2feat.pyx',
    'core/atomic_components/writer.pyx',

    # Tệp trong thư mục con core/aux_models
    'core/aux_models/blaze_face.pyx',
    'core/aux_models/face_mesh.pyx',
    'core/aux_models/hubert_stream.pyx',
    'core/aux_models/insightface_det.pyx',
    'core/aux_models/insightface_landmark106.pyx',
    'core/aux_models/landmark203.pyx',
    'core/aux_models/mediapipe_landmark478.pyx',

    # Tệp trong thư mục con core/models
    'core/models/appearance_extractor.pyx',
    'core/models/decoder.pyx',
    'core/models/lmdm.pyx',
    #'core/models/motion_extractor.pyx', # Đã có từ trước
    'core/models/stitch_network.pyx',
    'core/models/warp_network.pyx',

    # Tệp trong thư mục con core/utils
    'core/utils/crop.pyx',
    'core/utils/eye_info.pyx',
    'core/utils/get_mask.pyx',
    'core/utils/load_model.pyx',
    'core/utils/tensorrt_utils.pyx',
    # Lưu ý: core/utils/blend/ đã là Cython, không cần thêm vào đây trừ khi bạn muốn thay đổi cách nó được build.
    # Nếu core/utils/blend/__init__.py chỉ import .so từ build Cython riêng của nó thì không cần thêm.
    # Nếu bạn muốn setup.py này quản lý luôn việc build của blend, bạn cần cấu hình Extension cho nó phức tạp hơn một chút.
    # Hiện tại, giả định blend được build riêng hoặc không cần build lại qua setup.py này.
]

# --- Tự động tạo các Extension ---

# Biên dịch các tệp trong danh sách FILES_TO_COMPILE
extensions = []
for f_path in FILES_TO_COMPILE:
    module_name_parts = os.path.splitext(f_path)[0].split(os.path.sep)
    module_name = '.'.join(module_name_parts)
    
    extensions.append(
        Extension(
            name=module_name,
            sources=[os.path.join(ROOT_DIR, f_path)],
            include_dirs=[numpy.get_include()]
        )
    )

# Thiết lập trình biên dịch
setup(
    name="DittoTalkingHead_CompiledModules", # Tên có thể tùy chỉnh
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",      # Sử dụng cú pháp Python 3
            'embedsignature': True,     # Nhúng chữ ký hàm vào docstrings (hữu ích khi debug)
            # 'profile': True,          # Bật profiling (nếu cần tối ưu hiệu năng)
            # 'linetrace': True,        # Bật line tracing (nếu cần debug với cProfile)
        },
        quiet=False, # Đặt thành False để xem chi tiết quá trình biên dịch, True để giảm output
        # force=True, # Buộc biên dịch lại ngay cả khi tệp .c chưa thay đổi (hữu ích khi debug setup.py)
        # nthreads=os.cpu_count() or 1 # Sử dụng nhiều CPU core để biên dịch nhanh hơn
    ),
    # Thêm các tùy chọn khác nếu cần, ví dụ:
    # include_dirs=[numpy.get_include(), 'đường/dẫn/tới/thư_viện_C_khác'],
    # library_dirs=['đường/dẫn/tới/thư_viện_C_khác'],
    # libraries=['tên_thư_viện_C_khác_không_có_lib_và_đuôi'],
)

print(f"\nĐã hoàn tất quá trình biên dịch cho {len(extensions)} tệp.")
print("Các tệp mã máy (.so hoặc .pyd) đã được tạo.")
print("Hãy kiểm tra kỹ lưỡng và xóa các tệp .pyx và .c không cần thiết trước khi phân phối.")

