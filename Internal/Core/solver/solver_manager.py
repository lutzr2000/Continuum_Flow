import base64
from multiprocessing.connection import Listener
import secrets
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
        self._listener = None
        self._connection = None
        self._ready = False
        self._starting = False
        self._reader_thread = None
        self._active_job_id = None
        self._job_results = {}
        self._next_job_id = 1
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
            connection = self._connection
            if process is None or connection is None or process.poll() is not None:
                raise RuntimeError("Solver worker is not running")

            connection.send(message)

    def _handle_message(self, message):
        message_type = str(message.get("type") or "").strip().lower()

        with self._condition:
            if message_type == "ready":
                self._ready = True
                self._starting = False
                self._last_error = None
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

    def _read_messages(self, process, listener):
        connection = None
        try:
            connection = listener.accept()
            with self._condition:
                self._connection = connection
                self._condition.notify_all()

            while True:
                message = connection.recv()
                self._handle_message(message)
        except (EOFError, OSError):
            pass
        finally:
            if connection is not None:
                connection.close()
            listener.close()
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
        self._job_results.clear()

        authkey = secrets.token_bytes(32)
        listener = Listener(("127.0.0.1", 0), authkey=authkey)
        host, port = listener.address

        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "Solver.General.main",
                    "--worker",
                    "--connection-host",
                    str(host),
                    "--connection-port",
                    str(port),
                    "--connection-authkey",
                    base64.b64encode(authkey).decode("ascii"),
                ],
                cwd=str(self._addon_root()),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=self._creation_flags(),
            )
        except Exception:
            listener.close()
            raise

        self._process = process
        self._listener = listener
        self._reader_thread = threading.Thread(
            target=self._read_messages,
            args=(process, listener),
            daemon=True,
        )
        self._reader_thread.start()

    def _reset_runtime_state_locked(self):
        connection = self._connection
        listener = self._listener
        self._connection = None
        self._listener = None

        if connection is not None:
            connection.close()
        if listener is not None:
            listener.close()

        self._process = None
        self._ready = False
        self._starting = False
        self._reader_thread = None
        self._active_job_id = None
        self._last_error = None

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
        return Path(__file__).resolve().parents[3]

    @staticmethod
    def _creation_flags():
        if sys.platform.startswith("win") and hasattr(subprocess, "CREATE_NO_WINDOW"):
            return subprocess.CREATE_NO_WINDOW
        return 0


solver_manager = SolverManager()
