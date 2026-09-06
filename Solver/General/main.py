from pathlib import Path
from time import perf_counter
import contextlib
import json
import sys
import traceback
import os
import shutil
import platform


def emit_message(message):
    sys.__stdout__.write(json.dumps(message) + "\n")
    sys.__stdout__.flush()


def main(config):
    total_start_time = perf_counter()

    try:
        extra_paths = (config.get("meta") or {}).get("parent_sys_path") or ()
        for path in reversed(extra_paths):
            if path and path not in sys.path:
                sys.path.insert(0, path)

        prepare_cuda_libraries()

        simulation_cfg = config.get("simulation") or {}
        solver_backend = (
            str((simulation_cfg.get("settings") or {}).get("solver_backend", "GPU"))
            .strip()
            .upper()
        )

        if solver_backend == "CPU":
            raise NotImplementedError("CPU solver is not implemented.")

        elif solver_backend == "GPU":
            import Solver.Kernel_GPU.kernel as solver_kernel_module

            return solver_kernel_module.solver(config)

    finally:
        total_runtime = perf_counter() - total_start_time
        print(f"Bake runtime: {total_runtime:.3f} s")
        print("################################################################")


def prepare_cuda_libraries():
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


def run_worker():
    emit_message({"type": "ready"})

    for raw_line in sys.stdin:
        payload = raw_line.strip()
        if not payload:
            continue

        message = json.loads(payload)
        command = str(message.get("command") or "").strip().lower()

        if command == "shutdown":
            break

        if command == "run_job":
            job_id = int(message.get("job_id", 0) or 0)
            config = message.get("config") or {}

            emit_message(
                {
                    "type": "job_started",
                    "job_id": job_id,
                }
            )

            logger = _JsonLogStream()

            try:
                with contextlib.redirect_stdout(logger), contextlib.redirect_stderr(
                    logger
                ):
                    try:
                        main(config)
                    finally:
                        logger.flush()

                emit_message(
                    {
                        "type": "stats",
                        "frame": 0,
                        "active_tiles": 0,
                        "total_tiles": 0,
                        "active_cells": 0,
                        "total_cells": 0,
                        "vram_used_mb": 0.0,
                        "vram_total_mb": 0.0,
                    }
                )

                emit_message(
                    {
                        "type": "job_finished",
                        "job_id": job_id,
                        "success": True,
                    }
                )

            except Exception:
                emit_message(
                    {
                        "type": "stats",
                        "frame": 0,
                        "active_tiles": 0,
                        "total_tiles": 0,
                        "active_cells": 0,
                        "total_cells": 0,
                        "vram_used_mb": 0.0,
                        "vram_total_mb": 0.0,
                    }
                )

                emit_message(
                    {
                        "type": "job_finished",
                        "job_id": job_id,
                        "success": False,
                        "message": "Solver job failed",
                        "traceback": traceback.format_exc(),
                    }
                )

            continue

        emit_message(
            {
                "type": "error",
                "message": f"Unknown solver worker command: {command}",
            }
        )


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
                emit_message(
                    {
                        "type": "log",
                        "message": line,
                    }
                )

        return len(text)

    def flush(self):
        remaining = self._buffer.strip()

        if remaining:
            emit_message(
                {
                    "type": "log",
                    "message": remaining,
                }
            )

        self._buffer = ""


if __name__ == "__main__":
    try:
        if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
            run_worker()

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
