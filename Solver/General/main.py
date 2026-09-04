import base64
import contextlib
from multiprocessing.connection import Client
import sys
import traceback


def main(config=None):
    print("Started")


def run_with_worker_logging(connection, callback, *args):
    class ConnectionLogStream:
        def write(self, text):
            message = str(text).strip()
            if message:
                connection.send({"type": "log", "message": message})
            return len(str(text))

        def flush(self):
            pass

    with contextlib.redirect_stdout(ConnectionLogStream()):
        callback(*args)


def run_worker_loop(host, port, authkey):
    connection = Client((host, port), authkey=authkey)
    connection.send({"type": "ready"})

    try:
        while True:
            try:
                message = connection.recv()
            except (EOFError, OSError):
                break

            command = str(message.get("command") or "").strip().lower()

            if command == "shutdown":
                break

            if command == "preload":
                connection.send({
                    "type": "preload_complete",
                    "backend": str(message.get("backend") or "").strip().upper(),
                    "success": True,
                })
                continue

            if command == "run_job":
                job_id = int(message.get("job_id", 0) or 0)
                connection.send({"type": "job_started", "job_id": job_id})
                try:
                    run_with_worker_logging(connection, main, message.get("config") or {})
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

            connection.send({
                "type": "error",
                "message": f"Unknown solver worker command: {command}",
            })
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
    else:
        main()
