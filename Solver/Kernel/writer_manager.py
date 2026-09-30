import json
import subprocess
import sys

import numpy as np

from multiprocessing import shared_memory
from pathlib import Path

WRITER_COUNT = 4


def start_writer():
    writer_script = Path(__file__).resolve().parent / "writer_worker.py"

    slots = []
    for _ in range(WRITER_COUNT):
        process = subprocess.Popen(
            [sys.executable, str(writer_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        slots.append(
            {
                "process": process,
                "shared_buffers": [],
                "busy": False,
            }
        )

    return {
        "slots": slots,
        "next_slot": 0,
    }


def _finish_slot(slot):
    if not slot["busy"]:
        return

    process = slot["process"]
    try:
        response_line = process.stdout.readline()
        if not response_line:
            raise RuntimeError("VDB writer process exited.")

        response = json.loads(response_line)
        if response["status"] != "ok":
            raise RuntimeError(response["message"])
    finally:
        for shm in slot["shared_buffers"]:
            shm.close()
            shm.unlink()
        slot["shared_buffers"].clear()
        slot["busy"] = False


def write(writer, output_data, output_path, tile_size, delta, nx, ny, precision):
    slots = writer["slots"]
    slot_index = writer["next_slot"]
    slot = slots[slot_index]
    writer["next_slot"] = (slot_index + 1) % len(slots)

    _finish_slot(slot)

    process = slot["process"]
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

        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()
        slot["shared_buffers"] = shared_buffers
        slot["busy"] = True
    except Exception:
        for shm in shared_buffers:
            shm.close()
            shm.unlink()
        raise


def stop_writer(writer):
    first_error = None

    for slot in writer["slots"]:
        try:
            _finish_slot(slot)
        except Exception as exc:
            if first_error is None:
                first_error = exc

    for slot in writer["slots"]:
        process = slot["process"]
        if process.poll() is None:
            process.stdin.write("__QUIT__\n")
            process.stdin.flush()
            process.stdin.close()
        process.wait()

    if first_error is not None:
        raise first_error
