import sys
import threading
import traceback
from pathlib import Path

ADDON_ROOT = Path(__file__).resolve().parents[3]

if str(ADDON_ROOT) not in sys.path:
    sys.path.insert(0, str(ADDON_ROOT))


class SolverManager:
    def __init__(self):
        self._lock = threading.Lock()

        self._solver_thread = None
        self._active_job_id = None
        self._job_results = {}
        self._next_job_id = 1

        self._stats = {}
        self._compiling_backend = None
        self._compiled_backends = set()

    def start_job(self, config):
        from .. import volume_renderer

        volume_renderer.ensure_update_live_preview()

        with self._lock:
            if self._active_job_id is not None:
                raise RuntimeError("Solver is already busy")

            self._stats = {}

            job_id = self._next_job_id
            self._next_job_id += 1

            self._active_job_id = job_id

            thread = threading.Thread(
                target=self._run_job,
                args=(job_id, config),
                name=f"ContinuumFlowSolver-{job_id}",
                daemon=True,
            )

            self._solver_thread = thread

        thread.start()

        return job_id

    def _handle_message(self, message):
        if not isinstance(message, dict):
            return

        message_type = message.get("type")

        if message_type == "stats":
            with self._lock:
                self._stats = dict(message)
            return

        if message_type == "log":
            text = message.get("message")

            if text:
                print(f"[Solver] {text}")

            return

    def _run_job(self, job_id, config):
        clear_callback = None

        try:
            from Solver.General.main import (
                main,
                set_message_callback,
                clear_message_callback,
            )

            clear_callback = clear_message_callback
            set_message_callback(self._handle_message)

            main(config)

            result = {
                "type": "job_finished",
                "job_id": job_id,
                "success": True,
            }

        except Exception as exc:
            traceback_text = traceback.format_exc()

            print(traceback_text)

            result = {
                "type": "job_finished",
                "job_id": job_id,
                "success": False,
                "message": str(exc) or "Solver job failed",
                "traceback": traceback_text,
            }

        finally:
            if clear_callback is not None:
                clear_callback()

            from Solver.Kernel import preview

            preview.close()

            with self._lock:
                self._job_results[job_id] = result

                if self._active_job_id == job_id:
                    self._active_job_id = None

                self._stats = {}

                self._solver_thread = None

    def get_job_result(self, job_id):
        with self._lock:
            return self._job_results.pop(int(job_id), None)

    def is_compiling(self):
        return False

    def get_stats(self):
        with self._lock:
            return dict(self._stats)

    def request_preload(self, backend, config=None):
        return

    def shutdown(self):
        from Solver.Kernel import preview

        preview.close()
        with self._lock:
            thread = self._solver_thread

        if thread is not None and thread.is_alive():
            print("[Solver] Solver thread is still running during shutdown.")


solver_manager = SolverManager()
