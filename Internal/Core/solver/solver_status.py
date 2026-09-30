import platform


def detect_gpu_available():
    if platform.system() == "Darwin":
        return False

    try:
        from numba import cuda

        return len(list(cuda.gpus)) > 0
    except Exception:
        return False


def detect_opencl_available():
    """Return whether PyOpenCL can see at least one usable compute device."""
    try:
        import pyopencl as cl

        return any(platform.get_devices() for platform in cl.get_platforms())
    except Exception:
        return False


gpu_available = detect_gpu_available()
opencl_available = detect_opencl_available()
bake_running = False
bake_available = False
active_bake_operator = None
last_output_directory = None
progress = 0.0
progress_current_frames = 0
progress_total_frames = 0
progress_text = "0%"
