import json
import subprocess
import sys

import numpy as np

from multiprocessing import shared_memory
from pathlib import Path


def start_writer():
    writer_script = Path(__file__).resolve().parent / "writer_worker.py"

    return subprocess.Popen(
        [sys.executable, str(writer_script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def write(writer, output_data, output_path, tile_size, delta, nx, ny, precision):
    shared_buffers = []

    def share_array(array):
        array = np.ascontiguousarray(array)

        shm = shared_memory.SharedMemory(
            create=True,
            size=array.nbytes,
        )

        shared_buffers.append(shm)

        shared_array = np.ndarray(
            array.shape,
            dtype=array.dtype,
            buffer=shm.buf,
        )

        np.copyto(shared_array, array)

        return {
            "shm_name": shm.name,
            "shape": list(array.shape),
            "dtype": array.dtype.str,
        }

    try:
        payload = {
            "output_path": str(output_path),
            "tile_size": int(tile_size),
            "delta": float(delta),
            "nx": int(nx),
            "ny": int(ny),
            "precision": str(precision),
            "fields": {},
        }

        for name, array in output_data["fields"].items():
            payload["fields"][name] = share_array(array)

        payload["tile_map"] = share_array(output_data["tile_map"])

        writer.stdin.write(json.dumps(payload) + "\n")
        writer.stdin.flush()

        response_line = writer.stdout.readline()

        if not response_line:
            raise RuntimeError("VDB writer process exited.")

        response = json.loads(response_line)

        if response["status"] != "ok":
            raise RuntimeError(response["message"])

    finally:
        for shm in shared_buffers:
            shm.close()
            shm.unlink()


def stop_writer(writer):
    if writer.poll() is None:
        writer.stdin.write("__QUIT__\n")
        writer.stdin.flush()
        writer.stdin.close()

    writer.wait()
