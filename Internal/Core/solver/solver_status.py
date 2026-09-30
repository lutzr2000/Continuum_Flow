def detect_gpu_available():
    """Return whether PyOpenCL can see at least one GPU device."""
    try:
        import pyopencl as cl

        return any(
            platform.get_devices(device_type=cl.device_type.GPU)
            for platform in cl.get_platforms()
        )
    except Exception:
        return False


gpu_available = detect_gpu_available()
bake_running = False
bake_available = False
active_bake_operator = None
last_output_directory = None
progress = 0.0
progress_current_frames = 0
progress_total_frames = 0
progress_text = "0%"
