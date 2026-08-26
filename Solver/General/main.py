from pathlib import Path
from time import perf_counter
import contextlib
import json
import sys
import traceback

voxelise_mesh_module = None
gpu_kernel_module = None
cpu_kernel_module = None
np = None


class _JsonLogStream:
    def __init__(self):
        self._buffer = ""

    def write(self, text):
        text = str(text or "")
        if not text:
            return 0

        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                _emit_message({"type": "log", "message": line})
        return len(text)

    def flush(self):
        remaining = self._buffer.strip()
        if remaining:
            _emit_message({"type": "log", "message": remaining})
        self._buffer = ""


def _emit_message(message):
    sys.__stdout__.write(json.dumps(message) + "\n")
    sys.__stdout__.flush()


def _extend_sys_path(config):
    extra_paths = (config.get("meta") or {}).get("parent_sys_path") or ()
    for path in reversed(extra_paths):
        if path and path not in sys.path:
            sys.path.insert(0, path)


def _prepare_cuda_libraries():
    import os
    import shutil
    import platform

    system = platform.system()

    for entry in map(Path, sys.path):
        nvidia = entry / "nvidia"
        if not nvidia.exists():
            continue

        lib_dirs = set()

        if system == "Windows":
            for dll in nvidia.rglob("*.dll"):
                lib_dirs.add(dll.parent)

                name = dll.name.lower()

                if name.startswith("cudart64_"):
                    alias = dll.parent / "cudart.dll"
                    if not alias.exists():
                        shutil.copy2(dll, alias)

                elif name.startswith("nvvm64_"):
                    alias = dll.parent / "nvvm.dll"
                    if not alias.exists():
                        shutil.copy2(dll, alias)

            for lib_dir in lib_dirs:
                os.add_dll_directory(str(lib_dir))

        elif system == "Linux":
            for so in nvidia.rglob("*.so*"):
                lib_dirs.add(so.parent)

            if lib_dirs:
                old = os.environ.get("LD_LIBRARY_PATH", "")
                new = os.pathsep.join(str(p) for p in sorted(lib_dirs))
                os.environ["LD_LIBRARY_PATH"] = new + (os.pathsep + old if old else "")

        elif system == "Darwin":
            return


def _cancel_requested(config):
    cancel_flag_path = (
        (config.get("meta") or {}).get("cancel_flag_path") or ""
    ).strip()
    return bool(cancel_flag_path) and Path(cancel_flag_path).exists()


def _ensure_common_modules(config):
    global np

    _extend_sys_path(config)
    _prepare_cuda_libraries()

    if np is None:
        import numpy as _np

        np = _np


def _get_solver_kernel_module():
    global gpu_kernel_module, cpu_kernel_module

    if gpu_kernel_module is None:
        import Solver.Kernel_GPU.kernel as _gpu_kernel_module

        gpu_kernel_module = _gpu_kernel_module
    return gpu_kernel_module


def preload_backend(backend, config=None):
    config = config or {}
    backend = str(backend or "GPU").strip().upper()
    _ensure_common_modules(config)
    _get_solver_kernel_module(backend)
    return backend


def main(config=None):
    global np, voxelise_mesh_module, gpu_kernel_module, cpu_kernel_module

    total_start_time = perf_counter()
    try:
        config = config or {}
        _ensure_common_modules(config)

        solver_kernel_module = _get_solver_kernel_module()

        if _cancel_requested(config):
            print("Bake cancelled during preprocessing.")
            return

        return solver_kernel_module.solver(
            config,
        )
    finally:
        total_runtime = perf_counter() - total_start_time
        print(f"Bake runtime: {total_runtime:.3f} s")
        print("################################################################")


def _run_with_worker_logging(callback, *args, **kwargs):
    logger = _JsonLogStream()
    with contextlib.redirect_stdout(logger), contextlib.redirect_stderr(logger):
        try:
            return callback(*args, **kwargs)
        finally:
            logger.flush()


def _run_worker_loop():
    _emit_message({"type": "ready"})

    for raw_line in sys.stdin:
        payload = raw_line.strip()
        if not payload:
            continue

        try:
            message = json.loads(payload)
        except json.JSONDecodeError:
            _emit_message(
                {
                    "type": "error",
                    "message": "Received invalid JSON command in solver worker",
                }
            )
            continue

        command = str(message.get("command") or "").strip().lower()

        if command == "shutdown":
            break

        if command == "preload":
            backend = str(message.get("backend") or "GPU").strip().upper()
            config = message.get("config") or {}
            try:
                _run_with_worker_logging(preload_backend, backend, config)
                _emit_message(
                    {
                        "type": "preload_complete",
                        "backend": backend,
                        "success": True,
                    }
                )
            except Exception:
                _emit_message(
                    {
                        "type": "preload_complete",
                        "backend": backend,
                        "success": False,
                        "message": traceback.format_exc(),
                    }
                )
            continue

        if command == "run_job":
            job_id = int(message.get("job_id", 0) or 0)
            config = message.get("config") or {}
            _emit_message({"type": "job_started", "job_id": job_id})
            try:
                _run_with_worker_logging(main, config)
                _emit_message(
                    {
                        "type": "job_finished",
                        "job_id": job_id,
                        "success": True,
                    }
                )
            except Exception:
                _emit_message(
                    {
                        "type": "job_finished",
                        "job_id": job_id,
                        "success": False,
                        "message": "Solver job failed",
                        "traceback": traceback.format_exc(),
                    }
                )
            continue

        _emit_message(
            {
                "type": "error",
                "message": f"Unknown solver worker command: {command}",
            }
        )


if __name__ == "__main__":
    try:
        if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
            _run_worker_loop()
        else:
            if len(sys.argv) < 2:
                raise ValueError("Expected bake directory path as first argument.")

            bake_directory = Path(sys.argv[1]).resolve()
            config = json.load(sys.stdin)
            config["bake_directory"] = str(bake_directory)
            main(config)

    except Exception:
        traceback.print_exc()
        sys.exit(1)
