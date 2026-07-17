import json
import subprocess
import sys
import threading
import time
from pathlib import Path


class SolverManager:
    def __init__(self):
        self._condition = threading.Condition()
        self._write_lock = threading.Lock()

        self._process = None
        self._ready = False
        self._starting = False
        self._reader_thread = None
        self._active_job_id = None
        self._job_results = {}
        self._next_job_id = 1
        self._preloaded_backends = set()
        self._preload_in_flight = set()
        self._last_error = None

    def start(self, wait=True, timeout=120.0):
        with self._condition:
            process_running = self._process is not None and self._process.poll() is None
            if not process_running and not self._starting:
                self._launch_worker_locked()

            if wait:
                deadline = None if timeout is None else (time.monotonic() + float(timeout))
                while True:
                    if self._process is not None and self._process.poll() is not None:
                        raise RuntimeError("Solver worker exited during startup")
                    if self._ready and self._process is not None and self._process.poll() is None:
                        break
                    if self._last_error:
                        raise RuntimeError(str(self._last_error))
                    remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                    if remaining is not None and remaining <= 0:
                        raise RuntimeError("Timed out while starting solver worker")
                    self._condition.wait(timeout=remaining)

            return self._ready and self._process is not None and self._process.poll() is None

    def request_preload(self, backend, config=None):
        backend = str(backend or "").strip().upper()
        if not backend:
            return

        self.start(wait=True, timeout=120.0)

        with self._condition:
            if backend in self._preloaded_backends or backend in self._preload_in_flight:
                return
            self._preload_in_flight.add(backend)

        self._send(
            {
                "command": "preload",
                "backend": backend,
                "config": config or {},
            }
        )

    def start_job(self, config):
        self.start(wait=True, timeout=120.0)

        with self._condition:
            if self._active_job_id is not None:
                raise RuntimeError("Solver is already busy")

            job_id = self._next_job_id
            self._next_job_id += 1
            self._active_job_id = job_id

        self._send(
            {
                "command": "run_job",
                "job_id": job_id,
                "config": config,
            }
        )

        return job_id

    def get_job_result(self, job_id):
        with self._condition:
            return self._job_results.pop(int(job_id), None)

    def wait_for_job(self, job_id, timeout=None):
        with self._condition:
            finished = self._condition.wait_for(
                lambda: int(job_id) in self._job_results,
                timeout=timeout,
            )
            if not finished:
                return None
            return self._job_results.pop(int(job_id), None)

    def shutdown(self):
        with self._condition:
            process = self._process

        if process is None:
            return

        if process.poll() is None:
            try:
                self._send({"command": "shutdown"})
                process.wait(timeout=5)
            except Exception:
                process.kill()
                process.wait(timeout=5)

        with self._condition:
            self._reset_runtime_state_locked()
            self._condition.notify_all()

    def is_ready(self):
        with self._condition:
            return self._ready and self._process is not None and self._process.poll() is None

    def _send(self, message):
        with self._write_lock:
            process = self._process
            if process is None or process.stdin is None or process.poll() is not None:
                raise RuntimeError("Solver worker is not running")

            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()

    def _handle_message(self, message):
        message_type = str(message.get("type") or "").strip().lower()

        with self._condition:
            if message_type == "ready":
                self._ready = True
                self._starting = False
                self._last_error = None
            elif message_type == "preload_complete":
                backend = str(message.get("backend") or "").strip().upper()
                self._preload_in_flight.discard(backend)
                if message.get("success", True) and backend:
                    self._preloaded_backends.add(backend)
                elif message.get("message"):
                    self._last_error = message.get("message")
            elif message_type == "job_started":
                self._active_job_id = int(message.get("job_id", 0) or 0) or None
            elif message_type == "job_finished":
                job_id = int(message.get("job_id", 0) or 0)
                self._job_results[job_id] = message
                if self._active_job_id == job_id:
                    self._active_job_id = None
            elif message_type == "error":
                self._last_error = message.get("message") or "Unknown solver worker error"
                job_id = int(message.get("job_id", 0) or 0)
                if job_id:
                    self._job_results[job_id] = {
                        "type": "job_finished",
                        "job_id": job_id,
                        "success": False,
                        "message": self._last_error,
                        "traceback": message.get("traceback"),
                    }
                    if self._active_job_id == job_id:
                        self._active_job_id = None
                else:
                    self._ready = False
                    self._starting = False
            elif message_type == "log":
                log_message = message.get("message")
                if log_message:
                    print("[Solver]", log_message)

            self._condition.notify_all()

    def _read_messages(self, process):
        try:
            if process.stdout is None:
                return

            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    print("[Solver]", line)
                    continue

                self._handle_message(message)
        finally:
            return_code = process.poll()
            with self._condition:
                self._mark_active_job_failed_locked(
                    f"Solver worker exited unexpectedly with code {return_code}"
                )
                self._reset_runtime_state_locked()
                self._condition.notify_all()

    def _launch_worker_locked(self):
        if self._process is not None and self._process.poll() is None:
            return

        self._last_error = None
        self._starting = True
        self._ready = False
        self._preloaded_backends.clear()
        self._preload_in_flight.clear()
        self._job_results.clear()

        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "Solver.General.dummy_main",
                "--worker",
            ],
            cwd=str(self._addon_root()),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=self._creation_flags(),
        )
        self._process = process
        self._reader_thread = threading.Thread(
            target=self._read_messages,
            args=(process,),
            daemon=True,
        )
        self._reader_thread.start()

    def _reset_runtime_state_locked(self):
        self._process = None
        self._ready = False
        self._starting = False
        self._reader_thread = None
        self._active_job_id = None
        self._last_error = None
        self._preload_in_flight.clear()

    def _mark_active_job_failed_locked(self, message):
        if self._active_job_id is None:
            return
        self._job_results[self._active_job_id] = {
            "type": "job_finished",
            "job_id": self._active_job_id,
            "success": False,
            "message": message,
        }
        self._active_job_id = None

    @staticmethod
    def _addon_root():
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def _creation_flags():
        if sys.platform.startswith("win") and hasattr(subprocess, "CREATE_NO_WINDOW"):
            return subprocess.CREATE_NO_WINDOW
        return 0


solver_manager = SolverManager()
