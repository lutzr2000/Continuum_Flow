import json
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path


_PROCESS_LOCK = threading.Lock()
_PROCESS_CONDITION = threading.Condition(_PROCESS_LOCK)
_PROCESS = None
_READY = False
_STARTING = False
_READER_THREAD = None
_WRITER_LOCK = threading.Lock()
_NEXT_JOB_ID = 1
_ACTIVE_JOB_ID = None
_JOB_RESULTS = {}
_PRELOADED_BACKENDS = set()
_PRELOAD_IN_FLIGHT = set()
_LAST_ERROR = None


def _addon_root():
    return Path(__file__).resolve().parents[2]


def _creation_flags():
    if sys.platform.startswith("win") and hasattr(subprocess, "CREATE_NO_WINDOW"):
        return subprocess.CREATE_NO_WINDOW
    return 0


def _reset_runtime_state_locked():
    global _PROCESS, _READY, _STARTING, _READER_THREAD, _ACTIVE_JOB_ID, _LAST_ERROR
    _PROCESS = None
    _READY = False
    _STARTING = False
    _READER_THREAD = None
    _ACTIVE_JOB_ID = None
    _LAST_ERROR = None
    _PRELOAD_IN_FLIGHT.clear()


def _mark_active_job_failed_locked(message):
    global _ACTIVE_JOB_ID
    if _ACTIVE_JOB_ID is None:
        return
    _JOB_RESULTS[_ACTIVE_JOB_ID] = {
        "type": "job_finished",
        "job_id": _ACTIVE_JOB_ID,
        "success": False,
        "message": message,
    }
    _ACTIVE_JOB_ID = None


def _send_message(message):
    with _WRITER_LOCK:
        process = _PROCESS
        if process is None or process.stdin is None or process.poll() is not None:
            raise RuntimeError("Solver worker is not running")
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()


def _handle_worker_message(message):
    global _READY, _STARTING, _ACTIVE_JOB_ID, _LAST_ERROR

    message_type = str(message.get("type") or "").strip().lower()

    with _PROCESS_CONDITION:
        if message_type == "ready":
            _READY = True
            _STARTING = False
            _LAST_ERROR = None
        elif message_type == "preload_complete":
            backend = str(message.get("backend") or "").strip().upper()
            _PRELOAD_IN_FLIGHT.discard(backend)
            if message.get("success", True) and backend:
                _PRELOADED_BACKENDS.add(backend)
        elif message_type == "job_started":
            _ACTIVE_JOB_ID = int(message.get("job_id", 0) or 0) or None
        elif message_type == "job_finished":
            job_id = int(message.get("job_id", 0) or 0)
            _JOB_RESULTS[job_id] = message
            if _ACTIVE_JOB_ID == job_id:
                _ACTIVE_JOB_ID = None
        elif message_type == "error":
            _LAST_ERROR = message.get("message") or "Unknown solver worker error"
            job_id = int(message.get("job_id", 0) or 0)
            if job_id:
                _JOB_RESULTS[job_id] = {
                    "type": "job_finished",
                    "job_id": job_id,
                    "success": False,
                    "message": _LAST_ERROR,
                    "traceback": message.get("traceback"),
                }
                if _ACTIVE_JOB_ID == job_id:
                    _ACTIVE_JOB_ID = None
            else:
                _READY = False
                _STARTING = False
        elif message_type == "log":
            log_message = message.get("message")
            if log_message:
                print("[Solver]", log_message)

        _PROCESS_CONDITION.notify_all()


def _reader_loop(process):
    try:
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                print("[Solver]", line)
                continue
            _handle_worker_message(message)
    finally:
        return_code = process.poll()
        with _PROCESS_CONDITION:
            _mark_active_job_failed_locked(
                f"Solver worker exited unexpectedly with code {return_code}"
            )
            _reset_runtime_state_locked()
            _PROCESS_CONDITION.notify_all()


def _launch_worker_locked():
    global _PROCESS, _STARTING, _READER_THREAD, _LAST_ERROR

    if _PROCESS is not None and _PROCESS.poll() is None:
        return

    _LAST_ERROR = None
    _STARTING = True
    _PRELOADED_BACKENDS.clear()
    _JOB_RESULTS.clear()

    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "Solver.General.main",
            "--worker",
        ],
        cwd=str(_addon_root()),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        creationflags=_creation_flags(),
    )
    _PROCESS = process
    _READER_THREAD = threading.Thread(
        target=_reader_loop,
        args=(process,),
        daemon=True,
    )
    _READER_THREAD.start()


def ensure_worker_running(wait=True, timeout=120.0, preload_backend=None):
    with _PROCESS_CONDITION:
        process_running = _PROCESS is not None and _PROCESS.poll() is None
        if not process_running and not _STARTING:
            _launch_worker_locked()

        if wait:
            deadline = None if timeout is None else (time.monotonic() + float(timeout))
            while True:
                if _PROCESS is not None and _PROCESS.poll() is not None:
                    raise RuntimeError("Solver worker exited during startup")
                if _READY and _PROCESS is not None and _PROCESS.poll() is None:
                    break
                if _LAST_ERROR:
                    raise RuntimeError(str(_LAST_ERROR))
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if remaining is not None and remaining <= 0:
                    raise RuntimeError("Timed out while starting solver worker")
                _PROCESS_CONDITION.wait(timeout=remaining)

        ready = _READY and _PROCESS is not None and _PROCESS.poll() is None

    if ready and preload_backend:
        request_preload(preload_backend)

    return ready


def _background_start(preload_backend=None):
    try:
        ensure_worker_running(wait=True, timeout=120.0, preload_backend=preload_backend)
    except Exception:
        print("Failed to start solver worker in background:")
        traceback.print_exc()


def start_worker_in_background(preload_backend=None):
    with _PROCESS_CONDITION:
        process_running = _PROCESS is not None and _PROCESS.poll() is None
        ready = _READY
        starting = _STARTING

    if process_running or starting:
        if preload_backend and ready:
            request_preload(preload_backend)
        return

    threading.Thread(
        target=_background_start,
        args=(preload_backend,),
        daemon=True,
    ).start()


def request_preload(backend):
    backend = str(backend or "").strip().upper()
    if backend not in {"CPU", "GPU"}:
        return False

    with _PROCESS_CONDITION:
        if not _READY or _PROCESS is None or _PROCESS.poll() is not None:
            return False
        if backend in _PRELOADED_BACKENDS or backend in _PRELOAD_IN_FLIGHT:
            return True
        _PRELOAD_IN_FLIGHT.add(backend)

    try:
        _send_message({"command": "preload", "backend": backend})
        return True
    except Exception:
        with _PROCESS_CONDITION:
            _PRELOAD_IN_FLIGHT.discard(backend)
        raise


def start_job(config_dict):
    global _NEXT_JOB_ID, _ACTIVE_JOB_ID

    ensure_worker_running(wait=True, timeout=120.0)

    backend = str(
        (((config_dict.get("simulations") or [{}])[0].get("settings") or {}).get("solver_backend") or "GPU")
    ).strip().upper()

    with _PROCESS_CONDITION:
        if _ACTIVE_JOB_ID is not None:
            raise RuntimeError("Solver worker is already running a bake job")
        job_id = _NEXT_JOB_ID
        _NEXT_JOB_ID += 1
        _JOB_RESULTS.pop(job_id, None)
        _ACTIVE_JOB_ID = job_id

    try:
        _send_message(
            {
                "command": "run_job",
                "job_id": job_id,
                "config": config_dict,
            }
        )
    except Exception:
        with _PROCESS_CONDITION:
            _mark_active_job_failed_locked("Failed to submit bake job to solver worker")
            _PROCESS_CONDITION.notify_all()
        raise

    request_preload(backend)
    return job_id


def get_job_result(job_id):
    with _PROCESS_LOCK:
        return _JOB_RESULTS.get(int(job_id or 0))


def wait_for_job(job_id, timeout=None):
    job_id = int(job_id or 0)
    with _PROCESS_CONDITION:
        deadline = None if timeout is None else (time.monotonic() + float(timeout))
        while job_id not in _JOB_RESULTS:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining is not None and remaining <= 0:
                return None
            _PROCESS_CONDITION.wait(timeout=remaining)
        return _JOB_RESULTS.get(job_id)


def shutdown_worker(restart=False):
    process = None

    with _PROCESS_CONDITION:
        process = _PROCESS

    if process is not None and process.poll() is None:
        try:
            _send_message({"command": "shutdown"})
            process.wait(timeout=5)
        except Exception:
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass

    with _PROCESS_CONDITION:
        _mark_active_job_failed_locked("Solver worker stopped")
        _reset_runtime_state_locked()
        _PROCESS_CONDITION.notify_all()

    if restart:
        start_worker_in_background()


def is_worker_ready():
    with _PROCESS_LOCK:
        return _READY and _PROCESS is not None and _PROCESS.poll() is None
