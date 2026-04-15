import json
import socketserver
import threading
from multiprocessing import shared_memory

import numpy as np
import openvdb


def _as_vdb_input_array(array):
    if array.dtype == np.float32 and array.flags["C_CONTIGUOUS"]:
        return array
    return np.asarray(array, dtype=np.float32, order="C")


def write_vdb(payload):
    output_vdb_path = payload["output_path"]
    grids = []
    open_shared_memory = []

    try:
        for variable_name, field_info in payload["fields"].items():
            shm = shared_memory.SharedMemory(name=field_info["shm_name"])
            open_shared_memory.append(shm)

            array = np.ndarray(
                tuple(field_info["shape"]),
                dtype=np.dtype(field_info["dtype"]),
                buffer=shm.buf,
            )

            grid = openvdb.FloatGrid()
            grid.name = variable_name
            grid.copyFromArray(_as_vdb_input_array(array))
            grids.append(grid)

        openvdb.write(output_vdb_path, grids=grids)

    finally:
        for shm in open_shared_memory:
            shm.close()



class _ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _VDBWriteRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        """Handle one persistent kernel writer connection."""
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


class HostVDBWriterServer:
    """Small in-process VDB writer endpoint for Blender-started bakes."""

    def __init__(self, host="127.0.0.1"):
        self.host = host
        self._server = _ThreadingTCPServer((host, 0), _VDBWriteRequestHandler)
        self._server.write_vdb = write_vdb
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self):
        return int(self._server.server_address[1])

    def endpoint(self):
        return {"host": self.host, "port": self.port}

    def start(self):
        self._thread.start()

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()
