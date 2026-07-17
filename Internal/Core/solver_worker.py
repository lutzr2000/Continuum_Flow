import threading
import traceback

from .solver_manager import solver_manager


def start_worker_in_background(preload_backend=None):
    if solver_manager.is_ready():
        if preload_backend:
            solver_manager.request_preload(preload_backend)
        return

    threading.Thread(
        target=_background_start,
        args=(preload_backend,),
        daemon=True,
    ).start()


def _background_start(preload_backend=None):
    try:
        ensure_worker_running(wait=True, timeout=120.0, preload_backend=preload_backend)
    except Exception:
        print("Failed to start solver worker in background:")
        traceback.print_exc()


def ensure_worker_running(wait=True, timeout=120.0, preload_backend=None):
    ready = solver_manager.start(wait=wait, timeout=timeout)
    if ready and preload_backend:
        solver_manager.request_preload(preload_backend)
    return ready


def request_preload(backend, config=None):
    solver_manager.request_preload(backend, config=config)


def start_job(config):
    return solver_manager.start_job(config)


def get_job_result(job_id):
    return solver_manager.get_job_result(job_id)


def shutdown_worker(restart=True):
    solver_manager.shutdown()
    if restart:
        start_worker_in_background()
