import json
import queue
import socketserver
import subprocess
import sys
import threading
from pathlib import Path

DEFAULT_VDB_WRITER_PROCESS_COUNT = 4
WRITER_CLOSE_TIMEOUT_SECONDS = 10
WRITER_TERMINATE_TIMEOUT_SECONDS = 3


def _bake_directory():
    """
    Return the local UI/Bake directory path.
    """
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return (Path.cwd() / "UI" / "Bake").resolve()


class _VDBWriterProcess:
    """
    One persistent UI-side VDB writer process.
    """

    def __init__(self, writer_script, writer_config=None):
        command = [sys.executable, str(writer_script)]
        if writer_config is not None:
            command.append(json.dumps(writer_config))
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def write(self, payload):
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("VDB writer process is not connected.")
        if self._process.poll() is not None:
            stderr = self._process.stderr.read().strip() if self._process.stderr else ""
            raise RuntimeError(f"VDB writer process exited early. {stderr}")

        self._process.stdin.write(json.dumps(payload) + "\n")
        self._process.stdin.flush()

        response_line = self._process.stdout.readline()
        if not response_line:
            stderr = self._process.stderr.read().strip() if self._process.stderr else ""
            raise RuntimeError(f"VDB writer process closed the response pipe. {stderr}")

        response = json.loads(response_line)
        if response.get("status") != "ok":
            raise RuntimeError(response.get("message", "unknown VDB writer error"))

    def close(self):
        if self._process.poll() is None and self._process.stdin is not None:
            try:
                self._process.stdin.write("__QUIT__\n")
                self._process.stdin.flush()
                self._process.stdin.close()
            except OSError:
                pass

        try:
            self._process.wait(timeout=WRITER_CLOSE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=WRITER_TERMINATE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()


class _VDBWriterProcessPool:
    """
    Round-robin pool of persistent UI-side VDB writer processes.
    """

    def __init__(self, process_count=DEFAULT_VDB_WRITER_PROCESS_COUNT, writer_config=None):
        writer_script = _bake_directory() / "writer_worker.py"
        if not writer_script.exists():
            raise FileNotFoundError(f"VDB writer script not found: {writer_script}")

        self._processes = []
        try:
            for _ in range(process_count):
                self._processes.append(
                    _VDBWriterProcess(
                        writer_script,
                        writer_config=writer_config,
                    )
                )
        except Exception:
            for writer_process in self._processes:
                try:
                    writer_process.close()
                except Exception:
                    pass
            raise

        self._available_processes = queue.Queue(maxsize=process_count)
        for writer_process in self._processes:
            self._available_processes.put(writer_process)

    def write(self, payload):
        writer_process = self._available_processes.get()
        try:
            writer_process.write(payload)
        finally:
            self._available_processes.put(writer_process)

    def close(self):
        for writer_process in self._processes:
            writer_process.close()


class _ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = False


class _VDBWriteRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        """
        Handle one persistent kernel writer connection.
        """
        try:
            for raw_line in self.rfile:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                if line == "__QUIT__":
                    break

                try:
                    payload = json.loads(line)
                    self.server.write_vdb(payload)
                    response = {"status": "ok"}
                except Exception as exc:
                    response = {"status": "error", "message": str(exc)}

                self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return


class HostVDBWriterServer:
    """
    Host endpoint that dispatches VDB write jobs to a process pool.
    """

    def __init__(
        self,
        host="127.0.0.1",
        writer_process_count=DEFAULT_VDB_WRITER_PROCESS_COUNT,
        writer_config=None,
    ):
        self.host = host
        self._writer_process_count = int(writer_process_count)
        self._writer_pool = _VDBWriterProcessPool(
            process_count=self._writer_process_count,
            writer_config=writer_config,
        )
        self._server = _ThreadingTCPServer((host, 0), _VDBWriteRequestHandler)
        self._server.write_vdb = self._writer_pool.write
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self):
        return int(self._server.server_address[1])

    def endpoint(self):
        return {
            "host": self.host,
            "port": self.port,
            "process_count": self._writer_process_count,
        }

    def start(self):
        self._thread.start()

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()
        self._writer_pool.close()
