import json
import sys
from multiprocessing import shared_memory

import numpy as np
import openvdb


def _as_vdb_input_array(array, dtype):
    """Return an array layout that OpenVDB can consume without avoidable copies."""
    if array.dtype == dtype and array.flags["C_CONTIGUOUS"]:
        return array
    return np.asarray(array, dtype=dtype, order="C")


def _create_scalar_grid(dtype):
    """Create a scalar OpenVDB grid and configure its file storage precision."""
    dtype = np.dtype(dtype)
    grid = openvdb.FloatGrid()
    grid.saveFloatAsHalf = bool(dtype == np.float16)
    return grid, np.float32


def _grid_transform_from_payload(payload, shape):
    """Build the OpenVDB transform that maps grid indices into Blender world space."""
    transform_info = payload.get("transform", {})
    voxel_size = float(transform_info.get("voxel_size", payload.get("delta", 1.0)))
    origin = transform_info.get("origin")
    if origin is None:
        nx, ny, _nz = shape
        origin = (
            -0.5 * float(nx) * voxel_size,
            -0.5 * float(ny) * voxel_size,
            0.0,
        )

    transform = openvdb.Transform()
    transform.postScale(voxel_size)
    transform.postTranslate(tuple(float(component) for component in origin[:3]))
    return transform


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

            grid, grid_dtype = _create_scalar_grid(array.dtype)
            grid.name = variable_name
            grid.copyFromArray(_as_vdb_input_array(array, grid_dtype))
            grid.transform = _grid_transform_from_payload(payload, array.shape)
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
