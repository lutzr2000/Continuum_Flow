import base64
import contextlib
from multiprocessing.connection import Client
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback

_dll_directory_handles = []
_registered_cuda_library_dirs = set()


def get_sys_path(config):
    meta = config.get("meta") or {}
    parent_sys_path = meta.get("parent_sys_path") or []
    if not parent_sys_path:
        return

    restored_paths = []
    seen = set()

    for entry in parent_sys_path:
        normalized = str(entry or "").strip()
        if not normalized or normalized in seen:
            continue
        restored_paths.append(normalized)
        seen.add(normalized)

    current_paths = [path for path in sys.path if path not in seen]
    sys.path[:] = restored_paths + current_paths


def find_cuda_libs():
    """Make CUDA libraries bundled in NVIDIA Python wheels discoverable."""
    system = platform.system()
    if system not in ("Windows", "Linux"):
        return

    dirs = set()
    pattern = "*.dll" if system == "Windows" else "*.so*"

    for entry in sys.path:
        root = Path(entry) / "nvidia"
        if not root.is_dir():
            continue

        for lib in root.rglob(pattern):
            dirs.add(lib.parent)

            if system == "Windows":
                name = lib.name.lower()
                alias = (
                    "cudart.dll"
                    if name.startswith("cudart64_")
                    else "nvvm.dll" if name.startswith("nvvm64_") else None
                )
                if alias and not (dst := lib.parent / alias).exists():
                    try:
                        shutil.copy2(lib, dst)
                    except OSError as e:
                        print(f"Could not create CUDA DLL alias {dst}: {e}")

    paths = [
        str(p) for p in sorted(dirs) if str(p) not in _registered_cuda_library_dirs
    ]
    _registered_cuda_library_dirs.update(paths)

    if system == "Windows":
        for path in paths:
            try:
                _dll_directory_handles.append(os.add_dll_directory(path))
            except OSError as e:
                print(f"Could not register CUDA DLL directory {path}: {e}")
    elif paths:
        old = os.environ.get("LD_LIBRARY_PATH")
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(paths + ([old] if old else []))


def main(config=None):
    config = config or {}
    get_sys_path(config)
    find_cuda_libs()
    simulation = config.get("simulation") or {}
    settings = simulation.get("settings") or {}
    backend = str(settings.get("solver_backend", "GPU")).strip().upper()

    if backend == "CPU":
        print("CPU solver not implemented yet.")
        return

    if backend == "GPU":
        from Solver.Kernel_GPU.kernel import solver

        return solver(config)


def run_worker_loop(host, port, authkey):
    connection = Client((host, port), authkey=authkey)
    connection.send({"type": "ready"})

    class ConnectionLogStream:
        def write(self, text):
            message = str(text).strip()
            if message:
                connection.send({"type": "log", "message": message})
            return len(str(text))

        def flush(self):
            pass

    try:
        while True:
            try:
                message = connection.recv()
            except (EOFError, OSError):
                break

            command = str(message.get("command") or "").strip().lower()

            if command == "shutdown":
                break

            if command == "run_job":
                job_id = int(message.get("job_id", 0) or 0)
                connection.send({"type": "job_started", "job_id": job_id})
                try:
                    with contextlib.redirect_stdout(ConnectionLogStream()):
                        main(message.get("config") or {})
                except Exception:
                    connection.send(
                        {
                            "type": "job_finished",
                            "job_id": job_id,
                            "success": False,
                            "message": "Solver job failed",
                            "traceback": traceback.format_exc(),
                        }
                    )
                else:
                    connection.send(
                        {
                            "type": "job_finished",
                            "job_id": job_id,
                            "success": True,
                        }
                    )
                continue

    finally:
        connection.close()


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        arguments = sys.argv[2:]
        connection_host = arguments[arguments.index("--connection-host") + 1]
        connection_port = int(arguments[arguments.index("--connection-port") + 1])
        connection_authkey = base64.b64decode(
            arguments[arguments.index("--connection-authkey") + 1]
        )
        run_worker_loop(
            connection_host,
            connection_port,
            connection_authkey,
        )
