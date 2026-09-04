import base64
import contextlib
from multiprocessing.connection import Client
import sys
import traceback


def _restore_parent_sys_path(config):
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


def main(config=None):
    config = config or {}
    _restore_parent_sys_path(config)
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
                    connection.send({
                        "type": "job_finished",
                        "job_id": job_id,
                        "success": False,
                        "message": "Solver job failed",
                        "traceback": traceback.format_exc(),
                    })
                else:
                    connection.send({
                        "type": "job_finished",
                        "job_id": job_id,
                        "success": True,
                    })
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
