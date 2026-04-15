import json
import sys
from multiprocessing import shared_memory

import numpy as np
import openvdb


def _as_vdb_input_array(array):
    """Return an array layout that OpenVDB can consume without avoidable copies."""
    if array.dtype == np.float32 and array.flags["C_CONTIGUOUS"]:
        return array
    return np.asarray(array, dtype=np.float32, order="C")


def write_vdb(payload):
    """Create one VDB file from field data stored in shared memory."""
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


def main():
    """Run a persistent JSON-lines VDB writer process."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "__QUIT__":
            break

        try:
            write_vdb(json.loads(line))
            sys.stdout.write('{"status": "ok"}\n')
        except Exception as exc:
            sys.stdout.write(json.dumps({"status": "error", "message": str(exc)}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
