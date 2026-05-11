"""Worker process that receives JSON payloads and writes OpenVDB output files."""

import json
import os
import sys
from multiprocessing import shared_memory

import numpy as np
import openvdb


def _as_vdb_input_array(array, dtype):
    """Return an array layout that OpenVDB can consume without avoidable copies."""
    if array.dtype == dtype and array.flags["C_CONTIGUOUS"]:
        return array
    return np.asarray(array, dtype=dtype, order="C")


def _create_scalar_grid(storage_dtype):
    """Create a scalar OpenVDB grid and configure its file storage precision."""
    storage_dtype = np.dtype(storage_dtype)
    grid = openvdb.FloatGrid()
    if hasattr(grid, "saveFloatAsHalf"):
        grid.saveFloatAsHalf = bool(storage_dtype == np.float16)
    return grid, np.float32


def _create_vector_grid(storage_dtype):
    """Create a vector OpenVDB grid and configure its file storage precision."""
    storage_dtype = np.dtype(storage_dtype)
    for grid_class_name in ("Vec3SGrid", "VectorGrid", "Vec3fGrid"):
        grid_class = getattr(openvdb, grid_class_name, None)
        if grid_class is None:
            continue
        grid = grid_class()
        if hasattr(grid, "saveFloatAsHalf"):
            grid.saveFloatAsHalf = bool(storage_dtype == np.float16)
        return grid, np.float32
    raise AttributeError("No supported OpenVDB vector grid class is available.")


def _nonzero_mask(array):
    """Return a boolean active mask for one scalar field."""
    return np.abs(array) > 1.0e-12


def _load_shared_array(field_info):
    """Open one shared-memory array view described by the writer payload."""
    shm = shared_memory.SharedMemory(name=field_info["shm_name"])
    array = np.ndarray(
        tuple(field_info["shape"]),
        dtype=np.dtype(field_info["dtype"]),
        buffer=shm.buf,
    )
    return shm, array


def _sparse_export_mask(payload):
    """Build the shared sparse mask from smoke and flame fields when requested."""
    sparse_mask_fields = payload.get("sparse_mask_fields", {})
    if not sparse_mask_fields:
        return None, []

    open_shared_memory = []
    sparse_mask = None
    for field_name in ("smoke", "flame"):
        field_info = sparse_mask_fields.get(field_name)
        if field_info is None:
            continue
        shm, array = _load_shared_array(field_info)
        open_shared_memory.append(shm)
        field_mask = _nonzero_mask(array)
        sparse_mask = field_mask if sparse_mask is None else np.logical_or(sparse_mask, field_mask)
    return sparse_mask, open_shared_memory


def _array_for_export(array, dtype, sparse_mask=None, sparse=False):
    """Prepare one field array for VDB export, applying sparse masking when requested."""
    if sparse and sparse_mask is not None:
        masked_array = np.zeros(array.shape, dtype=dtype)
        copy_mask = sparse_mask
        while copy_mask.ndim < array.ndim:
            copy_mask = np.expand_dims(copy_mask, axis=-1)
        np.copyto(masked_array, array, where=copy_mask, casting="same_kind")
        return _as_vdb_input_array(masked_array, dtype)
    return _as_vdb_input_array(array, dtype)


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


def _open_grid_field_arrays(grid_info):
    """Open all shared-memory arrays needed to build one VDB grid."""
    field_arrays = {}
    open_shared_memory = []
    for field_name, field_info in grid_info.get("fields", {}).items():
        shm, array = _load_shared_array(field_info)
        field_arrays[field_name] = array
        open_shared_memory.append(shm)
    return field_arrays, open_shared_memory


def _grid_array_for_export(grid_info, field_arrays, grid_dtype, sparse_mask):
    """Assemble one scalar or vector array in the layout expected by OpenVDB."""
    field_names = list(grid_info.get("fields", {}).keys())
    if grid_info.get("grid_type") == "vector":
        array = np.stack([field_arrays[field_name] for field_name in field_names], axis=-1)
    else:
        array = field_arrays[field_names[0]]

    return _array_for_export(
        array,
        grid_dtype,
        sparse_mask=sparse_mask,
        sparse=bool(grid_info.get("sparse", False)),
    )


def write_vdb(payload):
    """Create one VDB file from field data stored in shared memory."""
    output_vdb_path = payload["output_path"]
    temporary_output_path = f"{output_vdb_path}.tmp"
    grids = []
    open_shared_memory = []

    try:
        sparse_mask, sparse_mask_handles = _sparse_export_mask(payload)
        open_shared_memory.extend(sparse_mask_handles)

        for grid_info in payload.get("grids", ()):
            field_arrays, field_handles = _open_grid_field_arrays(grid_info)
            open_shared_memory.extend(field_handles)

            storage_dtype = np.dtype(grid_info.get("storage_dtype", grid_info["dtype"]))
            if grid_info.get("grid_type") == "vector":
                grid, grid_dtype = _create_vector_grid(storage_dtype)
            else:
                grid, grid_dtype = _create_scalar_grid(storage_dtype)

            grid_array = _grid_array_for_export(grid_info, field_arrays, grid_dtype, sparse_mask)
            grid.name = str(grid_info["name"]).lower()
            grid.copyFromArray(grid_array)
            grid.transform = _grid_transform_from_payload(payload, tuple(grid_info["shape"]))
            grids.append(grid)

        openvdb.write(temporary_output_path, grids=grids)
        os.replace(temporary_output_path, output_vdb_path)

    finally:
        try:
            if os.path.exists(temporary_output_path):
                os.remove(temporary_output_path)
        except OSError:
            pass
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
