import threading
import traceback

from .solver_manager import solver_manager


def start_worker_in_background():
    if solver_manager.is_ready():
        return

    threading.Thread(
        target=_background_start,
        daemon=True,
    ).start()


def _background_start():
    try:
        ensure_worker_running(wait=True, timeout=120.0)
    except Exception:
        print("Failed to start solver worker in background:")
        traceback.print_exc()


def ensure_worker_running(wait=True, timeout=120.0):
    return solver_manager.start(
        wait=wait,
        timeout=timeout,
    )


def start_job(config):
    return solver_manager.start_job(config)


def get_job_result(job_id):
    return solver_manager.get_job_result(job_id)


def shutdown_worker(restart=True):
    solver_manager.shutdown()

    if restart:
        start_worker_in_background()