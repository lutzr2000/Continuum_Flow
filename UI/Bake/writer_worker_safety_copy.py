import json
import os
import sys
from multiprocessing import shared_memory
from pathlib import Path
from time import perf_counter

import numpy as np
import openvdb


WRITER_TIMING_STATS = {}
WRITER_TIMING_FRAME_COUNT = 0
WRITER_TIMING_OUTPUT_DIR = None


def _record_timing(name, elapsed):
    WRITER_TIMING_STATS[name] = WRITER_TIMING_STATS.get(name, 0.0) + float(elapsed)


def _set_timing_output_dir_from_payload(payload):
    global WRITER_TIMING_OUTPUT_DIR

    output_path = payload.get("output_path")
    if not output_path:
        return

    output_dir = Path(output_path).resolve().parent
    if WRITER_TIMING_OUTPUT_DIR is None:
        WRITER_TIMING_OUTPUT_DIR = output_dir


def _write_timing_summary():
    if WRITER_TIMING_OUTPUT_DIR is None or not WRITER_TIMING_STATS:
        return

    summary_path = WRITER_TIMING_OUTPUT_DIR / f"writer_timing_summary_{os.getpid()}.txt"
    total_runtime = WRITER_TIMING_STATS.get("write_vdb.total", 0.0)

    lines = [
        "Writer timing summary:",
        f"  pid: {os.getpid()}",
        f"  frames: {WRITER_TIMING_FRAME_COUNT}",
    ]

    for name, elapsed in sorted(WRITER_TIMING_STATS.items(), key=lambda item: item[1], reverse=True):
        share = (elapsed / total_runtime * 100.0) if total_runtime > 0.0 else 0.0
        lines.append(f"  {name}: {elapsed:.3f} s ({share:.1f}%)")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _as_vdb_input_array(array, dtype):
    """
    Return an array layout that OpenVDB can consume without avoidable copies.
    """
    if array.dtype == dtype and array.flags["C_CONTIGUOUS"]:
        return array
    return np.asarray(array, dtype=dtype, order="C")


def _create_scalar_grid(storage_dtype):
    """
    Create a scalar OpenVDB grid and configure its file storage precision.
    """
    storage_dtype = np.dtype(storage_dtype)
    grid = openvdb.FloatGrid()
    if hasattr(grid, "saveFloatAsHalf"):
        grid.saveFloatAsHalf = bool(storage_dtype == np.float16)
    return grid, np.float32


def _create_vector_grid(storage_dtype):
    """
    Create a vector OpenVDB grid and configure its file storage precision.
    """
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


def _nonzero_mask(array, threshold):
    """
    Return a boolean active mask for one scalar field.
    """
    return np.abs(array) > float(threshold)


def _load_shared_array(field_info):
    """
    Open one shared-memory array view described by the writer payload.
    """
    shm = shared_memory.SharedMemory(name=field_info["shm_name"])
    array = np.ndarray(
        tuple(field_info["shape"]),
        dtype=np.dtype(field_info["dtype"]),
        buffer=shm.buf,
    )
    return shm, array


def _sparse_export_mask(payload):
    """
    Build the shared sparse mask from smoke and flame fields when requested.
    """
    sparse_mask_fields = payload.get("sparse_mask_fields", {})
    if not sparse_mask_fields:
        return None, []
    sparse_threshold = float(payload.get("sparse_threshold", 0.0))

    open_shared_memory = []
    sparse_mask = None
    for field_name in ("smoke", "flame"):
        field_info = sparse_mask_fields.get(field_name)
        if field_info is None:
            continue
        shm, array = _load_shared_array(field_info)
        open_shared_memory.append(shm)
        field_mask = _nonzero_mask(array, sparse_threshold)
        sparse_mask = (
            field_mask
            if sparse_mask is None
            else np.logical_or(sparse_mask, field_mask)
        )
    return sparse_mask, open_shared_memory


def _array_for_export(array, dtype, sparse_mask=None, sparse=False):
    """
    Prepare one field array for VDB export, applying sparse masking when requested.
    """
    if sparse and sparse_mask is not None:
        masked_array = np.zeros(array.shape, dtype=dtype)
        copy_mask = sparse_mask
        while copy_mask.ndim < array.ndim:
            copy_mask = np.expand_dims(copy_mask, axis=-1)
        np.copyto(masked_array, array, where=copy_mask, casting="same_kind")
        return _as_vdb_input_array(masked_array, dtype)
    return _as_vdb_input_array(array, dtype)


def _grid_transform_from_payload(payload, shape):
    """
    Build the OpenVDB transform that maps grid indices into Blender world space.
    """
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
    """
    Open all shared-memory arrays needed to build one VDB grid.
    """
    field_arrays = {}
    open_shared_memory = []
    for field_name, field_info in grid_info.get("fields", {}).items():
        shm, array = _load_shared_array(field_info)
        field_arrays[field_name] = array
        open_shared_memory.append(shm)
    return field_arrays, open_shared_memory


def _grid_array_for_export(grid_info, field_arrays, grid_dtype, sparse_mask):
    """
    Assemble one scalar or vector array in the layout expected by OpenVDB.
    """
    field_names = list(grid_info.get("fields", {}).keys())
    if grid_info.get("grid_type") == "vector":
        array = np.stack(
            [field_arrays[field_name] for field_name in field_names], axis=-1
        )
    else:
        array = field_arrays[field_names[0]]

    return _array_for_export(
        array,
        grid_dtype,
        sparse_mask=sparse_mask,
        sparse=bool(grid_info.get("sparse", False)),
    )


def _combine_velocity_component_grids(grid_infos):
    """
    Merge scalar u/v/w component grids into one velocity vector grid when possible.
    """
    combined_grids = []
    component_grids = {}

    for grid_info in grid_infos:
        grid_name = str(grid_info.get("name", "")).lower()
        if grid_name in {"u", "v", "w"} and grid_info.get("grid_type") == "scalar":
            component_grids[grid_name] = grid_info
            continue
        combined_grids.append(grid_info)

    if len(component_grids) != 3:
        for component_name in ("u", "v", "w"):
            grid_info = component_grids.get(component_name)
            if grid_info is not None:
                combined_grids.append(grid_info)
        return combined_grids

    reference_grid = component_grids["u"]
    reference_shape = tuple(reference_grid.get("shape", ()))
    reference_dtype = str(reference_grid.get("dtype"))
    reference_storage_dtype = str(
        reference_grid.get("storage_dtype", reference_dtype)
    )
    reference_sparse = bool(reference_grid.get("sparse", False))

    for component_name in ("v", "w"):
        grid_info = component_grids[component_name]
        if tuple(grid_info.get("shape", ())) != reference_shape:
            for name in ("u", "v", "w"):
                combined_grids.append(component_grids[name])
            return combined_grids
        if str(grid_info.get("dtype")) != reference_dtype:
            for name in ("u", "v", "w"):
                combined_grids.append(component_grids[name])
            return combined_grids
        if (
            str(grid_info.get("storage_dtype", grid_info.get("dtype")))
            != reference_storage_dtype
        ):
            for name in ("u", "v", "w"):
                combined_grids.append(component_grids[name])
            return combined_grids

    combined_grids.append(
        {
            "name": "velocity",
            "grid_type": "vector",
            "shape": reference_shape,
            "dtype": reference_dtype,
            "storage_dtype": reference_storage_dtype,
            "sparse": reference_sparse,
            "fields": {
                "u": dict(reference_grid["fields"]["u"]),
                "v": dict(component_grids["v"]["fields"]["v"]),
                "w": dict(component_grids["w"]["fields"]["w"]),
            },
        }
    )
    return combined_grids


def write_vdb(payload):
    """
    Create one VDB file from field data stored in shared memory.
    """
    global WRITER_TIMING_FRAME_COUNT

    _set_timing_output_dir_from_payload(payload)

    total_start = perf_counter()
    output_vdb_path = payload["output_path"]
    temporary_output_path = f"{output_vdb_path}.tmp"
    grids = []
    open_shared_memory = []

    try:
        section_start = perf_counter()
        sparse_mask, sparse_mask_handles = _sparse_export_mask(payload)
        open_shared_memory.extend(sparse_mask_handles)
        _record_timing("write_vdb.sparse_mask", perf_counter() - section_start)

        section_start = perf_counter()
        grid_infos = _combine_velocity_component_grids(payload.get("grids", ()))
        _record_timing("write_vdb.combine_velocity_grids", perf_counter() - section_start)

        for grid_info in grid_infos:
            section_start = perf_counter()
            field_arrays, field_handles = _open_grid_field_arrays(grid_info)
            open_shared_memory.extend(field_handles)
            _record_timing("write_vdb.open_grid_field_arrays", perf_counter() - section_start)

            section_start = perf_counter()
            storage_dtype = np.dtype(grid_info.get("storage_dtype", grid_info["dtype"]))
            if grid_info.get("grid_type") == "vector":
                grid, grid_dtype = _create_vector_grid(storage_dtype)
            else:
                grid, grid_dtype = _create_scalar_grid(storage_dtype)
            grid_array = _grid_array_for_export(
                grid_info, field_arrays, grid_dtype, sparse_mask
            )
            grid.name = str(grid_info["name"]).lower()
            grid.transform = _grid_transform_from_payload(
                payload, tuple(grid_info["shape"])
            )
            _record_timing("write_vdb.prepare_grid", perf_counter() - section_start)

            section_start = perf_counter()
            grid.copyFromArray(grid_array)
            grids.append(grid)
            _record_timing("write_vdb.copy_from_array", perf_counter() - section_start)

        section_start = perf_counter()
        openvdb.write(temporary_output_path, grids=grids)
        _record_timing("write_vdb.openvdb_write", perf_counter() - section_start)

        section_start = perf_counter()
        os.replace(temporary_output_path, output_vdb_path)
        _record_timing("write_vdb.replace_output", perf_counter() - section_start)

        WRITER_TIMING_FRAME_COUNT += 1

    finally:
        section_start = perf_counter()
        try:
            if os.path.exists(temporary_output_path):
                os.remove(temporary_output_path)
        except OSError:
            pass
        for shm in open_shared_memory:
            shm.close()
        _record_timing("write_vdb.cleanup", perf_counter() - section_start)
        _record_timing("write_vdb.total", perf_counter() - total_start)


def main():
    """
    Run a persistent JSON-lines VDB writer process.
    """
    try:
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
                sys.stdout.write(
                    json.dumps({"status": "error", "message": str(exc)}) + "\n"
                )
            sys.stdout.flush()
    finally:
        _write_timing_summary()


if __name__ == "__main__":
    main()
