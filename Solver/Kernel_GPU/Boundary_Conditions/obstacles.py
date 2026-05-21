import math

import numpy as np
from numba import cuda

import Solver.General.obstacles as general_obstacles
from Solver.Kernel_GPU.kernel_config import THREADS_PER_BLOCK_3D, volume_blocks_per_grid


mesh = general_obstacles.mesh
build_dynamic_runtime = general_obstacles.build_dynamic_runtime
update_dynamic_mask = general_obstacles.update_dynamic_mask
update_dynamic_obstacle_data = general_obstacles.update_dynamic_obstacle_data
_resolve_dynamic_object_state = general_obstacles._resolve_dynamic_object_state


@cuda.jit(cache=True)
def _clear_obstacle_fields_cuda(mask, velocity_x, velocity_y, velocity_z):
    """
    Update the full obstacle mask and wall-velocity fields by clearing all cells.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = mask.shape
    if i < nx and j < ny and k < nz:
        mask[i, j, k] = False
        velocity_x[i, j, k] = 0.0
        velocity_y[i, j, k] = 0.0
        velocity_z[i, j, k] = 0.0


@cuda.jit(cache=True)
def _clear_obstacle_fields_box_cuda(mask, velocity_x, velocity_y, velocity_z, ix0, iy0, iz0, sx, sy, sz):
    """
    Update one obstacle subregion by clearing the mask and wall-velocity fields.
    """
    di, dj, dk = cuda.grid(3)
    if di < sx and dj < sy and dk < sz:
        i = ix0 + di
        j = iy0 + dj
        k = iz0 + dk
        mask[i, j, k] = False
        velocity_x[i, j, k] = 0.0
        velocity_y[i, j, k] = 0.0
        velocity_z[i, j, k] = 0.0


@cuda.jit(cache=True)
def _sample_obstacle_data_backwards_object_cuda(
    out_mask,
    out_velocity_x,
    out_velocity_y,
    out_velocity_z,
    local_masks_flat,
    local_mask_offsets,
    local_mask_shapes,
    local_origins,
    object_index,
    ix0, iy0, iz0,
    sx, sy, sz,
    m00, m01, m02, m03,
    m10, m11, m12, m13,
    m20, m21, m22, m23,
    r00, r01, r02, r03,
    r10, r11, r12, r13,
    r20, r21, r22, r23,
    delta,
    ox, oy, oz,
):
    """
    Update one obstacle region by backward-sampling a moving object on the GPU.
    """
    di, dj, dk = cuda.grid(3)
    if di >= sx or dj >= sy or dk >= sz:
        return

    i = ix0 + di
    j = iy0 + dj
    k = iz0 + dk

    x = ox + i * delta
    y = oy + j * delta
    z = oz + k * delta

    bx = m00 * x + m01 * y + m02 * z + m03
    by = m10 * x + m11 * y + m12 * z + m13
    bz = m20 * x + m21 * y + m22 * z + m23

    base_ox = local_origins[object_index, 0]
    base_oy = local_origins[object_index, 1]
    base_oz = local_origins[object_index, 2]
    bi = int(math.floor((bx - base_ox) / delta + 0.5))
    bj = int(math.floor((by - base_oy) / delta + 0.5))
    bk = int(math.floor((bz - base_oz) / delta + 0.5))

    bn_x = local_mask_shapes[object_index, 0]
    bn_y = local_mask_shapes[object_index, 1]
    bn_z = local_mask_shapes[object_index, 2]
    if 0 <= bi < bn_x and 0 <= bj < bn_y and 0 <= bk < bn_z:
        flat_index = local_mask_offsets[object_index] + (bi * bn_y + bj) * bn_z + bk
        if local_masks_flat[flat_index]:
            out_mask[i, j, k] = True
            out_velocity_x[i, j, k] = r00 * bx + r01 * by + r02 * bz + r03
            out_velocity_y[i, j, k] = r10 * bx + r11 * by + r12 * bz + r13
            out_velocity_z[i, j, k] = r20 * bx + r21 * by + r22 * bz + r23


def prepare_dynamic_runtime_for_gpu(runtime_data):
    """
    Upload static per-object voxel masks so dynamic sampling can run on the GPU.
    """
    if runtime_data.get("gpu_ready"):
        return runtime_data

    objects = runtime_data.get("objects", ())
    mask_shapes = []
    mask_offsets = [0]
    mask_origins = []
    flat_masks = []

    for obj in objects:
        local_mask = obj.get("local_mask")
        if local_mask is None:
            continue
        local_mask = np.ascontiguousarray(local_mask)
        obj["local_mask_device"] = cuda.to_device(local_mask)
        flat_masks.append(local_mask.reshape(-1))
        mask_shapes.append(local_mask.shape)
        mask_origins.append(tuple(np.asarray(obj["local_origin"], dtype=np.float32)))
        mask_offsets.append(mask_offsets[-1] + local_mask.size)

    if flat_masks:
        local_masks_flat = np.concatenate(flat_masks).astype(np.bool_, copy=False)
    else:
        local_masks_flat = np.empty(0, dtype=np.bool_)

    runtime_data["local_masks_flat_device"] = cuda.to_device(local_masks_flat)
    runtime_data["local_mask_offsets_device"] = cuda.to_device(np.asarray(mask_offsets, dtype=np.int32))
    runtime_data["local_mask_shapes_device"] = cuda.to_device(np.asarray(mask_shapes, dtype=np.int32))
    runtime_data["local_origins_device"] = cuda.to_device(np.asarray(mask_origins, dtype=np.float32))
    runtime_data["gpu_mask_initialized"] = False

    for obj in objects:
        obj["last_gpu_index_bounds"] = None

    runtime_data["gpu_ready"] = True
    return runtime_data


def update_dynamic_obstacle_data_gpu(runtime_data, time_value, out_mask, out_velocity_x, out_velocity_y, out_velocity_z):
    """
    Update one moving obstacle mask and its wall velocity fields directly on the GPU.
    """

    prepare_dynamic_runtime_for_gpu(runtime_data)

    shape = runtime_data["shape"]
    origin = np.asarray(runtime_data["origin"], dtype=np.float32)
    delta = np.float32(runtime_data["delta"])
    full_volume = int(shape[0] * shape[1] * shape[2])
    dirty_regions = []
    dirty_volume = 0
    active_objects = []

    for object_index, obj in enumerate(runtime_data["objects"]):
        state = general_obstacles._resolve_dynamic_object_state(obj, time_value, delta, origin, shape)
        previous_bounds = obj.get("last_gpu_index_bounds")
        current_bounds = state["index_bounds"] if state["active"] else None
        dirty_bounds = general_obstacles._merge_index_bounds(previous_bounds, current_bounds)
        if dirty_bounds is not None:
            dirty_regions.append(dirty_bounds)
            sx, sy, sz = general_obstacles._region_shape(dirty_bounds)
            dirty_volume += sx * sy * sz
        if state["active"]:
            active_objects.append((object_index, obj, state))

    if not runtime_data.get("gpu_mask_initialized", False) or dirty_volume >= full_volume:
        _clear_obstacle_fields_cuda[
            volume_blocks_per_grid(shape, THREADS_PER_BLOCK_3D),
            THREADS_PER_BLOCK_3D,
        ](out_mask, out_velocity_x, out_velocity_y, out_velocity_z)
    else:
        for dirty_bounds in dirty_regions:
            ix0, _, iy0, _, iz0, _ = dirty_bounds
            sx, sy, sz = general_obstacles._region_shape(dirty_bounds)
            _clear_obstacle_fields_box_cuda[
                volume_blocks_per_grid((sx, sy, sz), THREADS_PER_BLOCK_3D),
                THREADS_PER_BLOCK_3D,
            ](out_mask, out_velocity_x, out_velocity_y, out_velocity_z, int(ix0), int(iy0), int(iz0), sx, sy, sz)

    for object_index, obj, state in active_objects:
        ix0, _, iy0, _, iz0, _ = state["index_bounds"]
        sx, sy, sz = general_obstacles._region_shape(state["index_bounds"])
        inv = state["inv"]
        rate = state["matrix_rate"]
        _sample_obstacle_data_backwards_object_cuda[
            volume_blocks_per_grid((sx, sy, sz), THREADS_PER_BLOCK_3D),
            THREADS_PER_BLOCK_3D,
        ](
            out_mask,
            out_velocity_x,
            out_velocity_y,
            out_velocity_z,
            runtime_data["local_masks_flat_device"],
            runtime_data["local_mask_offsets_device"],
            runtime_data["local_mask_shapes_device"],
            runtime_data["local_origins_device"],
            int(object_index),
            int(ix0), int(iy0), int(iz0),
            sx, sy, sz,
            np.float32(inv[0, 0]), np.float32(inv[0, 1]), np.float32(inv[0, 2]), np.float32(inv[0, 3]),
            np.float32(inv[1, 0]), np.float32(inv[1, 1]), np.float32(inv[1, 2]), np.float32(inv[1, 3]),
            np.float32(inv[2, 0]), np.float32(inv[2, 1]), np.float32(inv[2, 2]), np.float32(inv[2, 3]),
            np.float32(rate[0, 0]), np.float32(rate[0, 1]), np.float32(rate[0, 2]), np.float32(rate[0, 3]),
            np.float32(rate[1, 0]), np.float32(rate[1, 1]), np.float32(rate[1, 2]), np.float32(rate[1, 3]),
            np.float32(rate[2, 0]), np.float32(rate[2, 1]), np.float32(rate[2, 2]), np.float32(rate[2, 3]),
            delta,
            np.float32(origin[0]), np.float32(origin[1]), np.float32(origin[2]),
        )
        obj["last_gpu_index_bounds"] = state["index_bounds"]

    for obj in runtime_data["objects"]:
        if obj.get("dynamic_state") is None or not obj["dynamic_state"].get("active", False):
            obj["last_gpu_index_bounds"] = None

    runtime_data["gpu_mask_initialized"] = True
    runtime_data["last_has_obstacle"] = bool(len(active_objects) > 0)
    return out_mask, out_velocity_x, out_velocity_y, out_velocity_z
