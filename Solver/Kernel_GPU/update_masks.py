import math
from numba import cuda
import numpy as np
import Solver.General.update_masks as helper_update_masks

_MASK_THREADS_PER_BLOCK = (8, 8, 8)
_PACKED_MASK_CACHE = {}


@cuda.jit(cache=True)
def rebuild_mask_stack_from_objects(
    mask_stack,
    aggregate_mask,
    out_vx,
    out_vy,
    out_vz,
    local_masks_flat,
    local_mask_offsets,
    local_mask_shapes,
    local_origins,
    target_indices,
    bounds,
    inv_mats,
    rates,
    active_flags,
    delta,
    ox,
    oy,
    oz,
    compute_velocity_flag,
):
    i, j, k = cuda.grid(3)
    mask_count, nx, ny, nz = mask_stack.shape

    if i >= nx or j >= ny or k >= nz:
        return

    for mask_idx in range(mask_count):
        mask_stack[mask_idx, i, j, k] = False

    if compute_velocity_flag:
        out_vx[i, j, k] = 0.0
        out_vy[i, j, k] = 0.0
        out_vz[i, j, k] = 0.0

    aggregate_active = False
    obj_count = target_indices.shape[0]
    x = np.float32(ox + i * delta)
    y = np.float32(oy + j * delta)
    z = np.float32(oz + k * delta)

    for obj_idx in range(obj_count):
        if not active_flags[obj_idx]:
            continue

        if (
            i < bounds[obj_idx, 0]
            or i > bounds[obj_idx, 1]
            or j < bounds[obj_idx, 2]
            or j > bounds[obj_idx, 3]
            or k < bounds[obj_idx, 4]
            or k > bounds[obj_idx, 5]
        ):
            continue

        bx = (
            inv_mats[obj_idx, 0] * x
            + inv_mats[obj_idx, 1] * y
            + inv_mats[obj_idx, 2] * z
            + inv_mats[obj_idx, 3]
        )
        by = (
            inv_mats[obj_idx, 4] * x
            + inv_mats[obj_idx, 5] * y
            + inv_mats[obj_idx, 6] * z
            + inv_mats[obj_idx, 7]
        )
        bz = (
            inv_mats[obj_idx, 8] * x
            + inv_mats[obj_idx, 9] * y
            + inv_mats[obj_idx, 10] * z
            + inv_mats[obj_idx, 11]
        )

        base_ox = local_origins[obj_idx, 0]
        base_oy = local_origins[obj_idx, 1]
        base_oz = local_origins[obj_idx, 2]
        bi = int(math.floor((bx - base_ox) / delta + 0.5))
        bj = int(math.floor((by - base_oy) / delta + 0.5))
        bk = int(math.floor((bz - base_oz) / delta + 0.5))

        bn_x = local_mask_shapes[obj_idx, 0]
        bn_y = local_mask_shapes[obj_idx, 1]
        bn_z = local_mask_shapes[obj_idx, 2]
        if 0 <= bi < bn_x and 0 <= bj < bn_y and 0 <= bk < bn_z:
            flat_index = local_mask_offsets[obj_idx] + (bi * bn_y + bj) * bn_z + bk
            if local_masks_flat[flat_index]:
                target_idx = target_indices[obj_idx]
                mask_stack[target_idx, i, j, k] = True
                aggregate_active = True
                if compute_velocity_flag:
                    out_vx[i, j, k] = (
                        rates[obj_idx, 0] * bx
                        + rates[obj_idx, 1] * by
                        + rates[obj_idx, 2] * bz
                        + rates[obj_idx, 3]
                    )
                    out_vy[i, j, k] = (
                        rates[obj_idx, 4] * bx
                        + rates[obj_idx, 5] * by
                        + rates[obj_idx, 6] * bz
                        + rates[obj_idx, 7]
                    )
                    out_vz[i, j, k] = (
                        rates[obj_idx, 8] * bx
                        + rates[obj_idx, 9] * by
                        + rates[obj_idx, 10] * bz
                        + rates[obj_idx, 11]
                    )

    aggregate_mask[i, j, k] = aggregate_active


@cuda.jit(cache=True)
def rebuild_single_mask_from_objects(
    mask,
    out_vx,
    out_vy,
    out_vz,
    local_masks_flat,
    local_mask_offsets,
    local_mask_shapes,
    local_origins,
    bounds,
    inv_mats,
    rates,
    active_flags,
    delta,
    ox,
    oy,
    oz,
    compute_velocity_flag,
):
    i, j, k = cuda.grid(3)
    nx, ny, nz = mask.shape

    if i >= nx or j >= ny or k >= nz:
        return

    mask[i, j, k] = False
    if compute_velocity_flag:
        out_vx[i, j, k] = 0.0
        out_vy[i, j, k] = 0.0
        out_vz[i, j, k] = 0.0

    cell_active = False
    obj_count = active_flags.shape[0]
    x = np.float32(ox + i * delta)
    y = np.float32(oy + j * delta)
    z = np.float32(oz + k * delta)

    for obj_idx in range(obj_count):
        if not active_flags[obj_idx]:
            continue

        if (
            i < bounds[obj_idx, 0]
            or i > bounds[obj_idx, 1]
            or j < bounds[obj_idx, 2]
            or j > bounds[obj_idx, 3]
            or k < bounds[obj_idx, 4]
            or k > bounds[obj_idx, 5]
        ):
            continue

        bx = (
            inv_mats[obj_idx, 0] * x
            + inv_mats[obj_idx, 1] * y
            + inv_mats[obj_idx, 2] * z
            + inv_mats[obj_idx, 3]
        )
        by = (
            inv_mats[obj_idx, 4] * x
            + inv_mats[obj_idx, 5] * y
            + inv_mats[obj_idx, 6] * z
            + inv_mats[obj_idx, 7]
        )
        bz = (
            inv_mats[obj_idx, 8] * x
            + inv_mats[obj_idx, 9] * y
            + inv_mats[obj_idx, 10] * z
            + inv_mats[obj_idx, 11]
        )

        base_ox = local_origins[obj_idx, 0]
        base_oy = local_origins[obj_idx, 1]
        base_oz = local_origins[obj_idx, 2]
        bi = int(math.floor((bx - base_ox) / delta + 0.5))
        bj = int(math.floor((by - base_oy) / delta + 0.5))
        bk = int(math.floor((bz - base_oz) / delta + 0.5))

        bn_x = local_mask_shapes[obj_idx, 0]
        bn_y = local_mask_shapes[obj_idx, 1]
        bn_z = local_mask_shapes[obj_idx, 2]
        if 0 <= bi < bn_x and 0 <= bj < bn_y and 0 <= bk < bn_z:
            flat_index = local_mask_offsets[obj_idx] + (bi * bn_y + bj) * bn_z + bk
            if local_masks_flat[flat_index]:
                cell_active = True
                if compute_velocity_flag:
                    out_vx[i, j, k] = (
                        rates[obj_idx, 0] * bx
                        + rates[obj_idx, 1] * by
                        + rates[obj_idx, 2] * bz
                        + rates[obj_idx, 3]
                    )
                    out_vy[i, j, k] = (
                        rates[obj_idx, 4] * bx
                        + rates[obj_idx, 5] * by
                        + rates[obj_idx, 6] * bz
                        + rates[obj_idx, 7]
                    )
                    out_vz[i, j, k] = (
                        rates[obj_idx, 8] * bx
                        + rates[obj_idx, 9] * by
                        + rates[obj_idx, 10] * bz
                        + rates[obj_idx, 11]
                    )

    mask[i, j, k] = cell_active


def _iter_mask_entries(base_masks):
    if not base_masks:
        return

    first_entry = base_masks[0]
    if isinstance(first_entry, dict):
        for mask_entry in base_masks:
            yield 0, mask_entry
        return

    for target_idx, mask_entries in enumerate(base_masks):
        for mask_entry in mask_entries:
            yield target_idx, mask_entry


def _build_mask_pack(base_masks):
    flat_masks = []
    offsets = [0]
    shapes = []
    origins = []
    target_indices = []
    object_entries = []

    for target_idx, mask_entry in _iter_mask_entries(base_masks):
        base_voxels = mask_entry["voxels"]
        base_mask = np.ascontiguousarray(base_voxels["mask"], dtype=np.bool_)
        flat_masks.append(base_mask.reshape(-1))
        offsets.append(offsets[-1] + base_mask.size)
        shapes.append(base_mask.shape)
        origins.append(np.asarray(base_voxels["origin"], dtype=np.float32))
        target_indices.append(target_idx)
        object_entries.append(mask_entry)

    object_count = len(object_entries)
    if flat_masks:
        local_masks_flat = np.concatenate(flat_masks).astype(np.bool_, copy=False)
    else:
        local_masks_flat = np.empty(0, dtype=np.bool_)

    mask_shapes = np.asarray(shapes, dtype=np.int32) if shapes else np.zeros((0, 3), dtype=np.int32)
    mask_origins = np.asarray(origins, dtype=np.float32) if origins else np.zeros((0, 3), dtype=np.float32)
    target_indices_array = np.asarray(target_indices, dtype=np.int32)

    return {
        "object_entries": object_entries,
        "local_masks_flat": cuda.to_device(local_masks_flat),
        "local_mask_offsets": cuda.to_device(np.asarray(offsets, dtype=np.int32)),
        "local_mask_shapes": cuda.to_device(mask_shapes),
        "local_origins": cuda.to_device(mask_origins),
        "target_indices": cuda.to_device(target_indices_array),
        "bounds": cuda.to_device(np.zeros((object_count, 6), dtype=np.int32)),
        "inv_mats": cuda.to_device(np.zeros((object_count, 12), dtype=np.float32)),
        "rates": cuda.to_device(np.zeros((object_count, 12), dtype=np.float32)),
        "active_flags": cuda.to_device(np.zeros(object_count, dtype=np.bool_)),
        "count": object_count,
    }


def _get_mask_pack(base_masks):
    cache_key = id(base_masks)
    pack = _PACKED_MASK_CACHE.get(cache_key)
    if pack is None:
        pack = _build_mask_pack(base_masks)
        _PACKED_MASK_CACHE[cache_key] = pack
    return pack


def _update_frame_state(pack, t, delta, origin_x, origin_y, origin_z, shape):
    object_count = pack["count"]
    if object_count == 0:
        return True

    origin = np.asarray((origin_x, origin_y, origin_z), dtype=np.float32)
    bounds = np.zeros((object_count, 6), dtype=np.int32)
    inv_mats = np.zeros((object_count, 12), dtype=np.float32)
    rates = np.zeros((object_count, 12), dtype=np.float32)
    active_flags = np.zeros(object_count, dtype=np.bool_)

    for obj_idx, mask_entry in enumerate(pack["object_entries"]):
        mesh_object = mask_entry["mesh_object"]
        base_voxels = mask_entry["voxels"]

        object_matrix = helper_update_masks._world_matrix_at_time(mesh_object, t)
        rate = helper_update_masks._world_matrix_rate_at_time(mesh_object, t)
        bounds_center, bounds_extent = helper_update_masks._bounds_center_extent(
            base_voxels["bounds_min"],
            base_voxels["bounds_max"],
        )
        bounds_min, bounds_max = helper_update_masks._transform_bounds(
            bounds_center,
            bounds_extent,
            object_matrix,
        )
        inv = helper_update_masks._invert_affine_matrix(object_matrix)

        ix0, ix1, iy0, iy1, iz0, iz1 = helper_update_masks._bounds_to_indices(
            bounds_min,
            bounds_max,
            delta,
            origin,
            shape=shape,
        )

        inv_mats[obj_idx, :] = np.asarray(inv[:3, :4], dtype=np.float32).reshape(12)
        rates[obj_idx, :] = np.asarray(rate[:3, :4], dtype=np.float32).reshape(12)

        if ix0 > ix1 or iy0 > iy1 or iz0 > iz1:
            continue

        bounds[obj_idx, :] = (ix0, ix1, iy0, iy1, iz0, iz1)
        active_flags[obj_idx] = True

    pack["bounds"].copy_to_device(bounds)
    pack["inv_mats"].copy_to_device(inv_mats)
    pack["rates"].copy_to_device(rates)
    pack["active_flags"].copy_to_device(active_flags)
    return True


def update_masks(
    masks,
    base_masks,
    t,
    delta,
    origin_x,
    origin_y,
    origin_z,
    obstacle_velocity_x,
    obstacle_velocity_y,
    obstacle_velocity_z,
    compute_velocity_flag=False,
    aggregate_mask=None,
):
    pack = _get_mask_pack(base_masks)
    if not _update_frame_state(
        pack,
        t,
        delta,
        origin_x,
        origin_y,
        origin_z,
        masks.shape[-3:],
    ):
        return masks

    if len(masks.shape) == 4:
        if aggregate_mask is None:
            raise ValueError("aggregate_mask is required for stacked source masks")
        launch_shape = aggregate_mask.shape
        launch_blocks = (
            (int(launch_shape[0]) + _MASK_THREADS_PER_BLOCK[0] - 1) // _MASK_THREADS_PER_BLOCK[0],
            (int(launch_shape[1]) + _MASK_THREADS_PER_BLOCK[1] - 1) // _MASK_THREADS_PER_BLOCK[1],
            (int(launch_shape[2]) + _MASK_THREADS_PER_BLOCK[2] - 1) // _MASK_THREADS_PER_BLOCK[2],
        )
        rebuild_mask_stack_from_objects[launch_blocks, _MASK_THREADS_PER_BLOCK](
            masks,
            aggregate_mask,
            obstacle_velocity_x,
            obstacle_velocity_y,
            obstacle_velocity_z,
            pack["local_masks_flat"],
            pack["local_mask_offsets"],
            pack["local_mask_shapes"],
            pack["local_origins"],
            pack["target_indices"],
            pack["bounds"],
            pack["inv_mats"],
            pack["rates"],
            pack["active_flags"],
            np.float32(delta),
            np.float32(origin_x),
            np.float32(origin_y),
            np.float32(origin_z),
            compute_velocity_flag,
        )
        return masks

    launch_blocks = (
        (int(masks.shape[0]) + _MASK_THREADS_PER_BLOCK[0] - 1) // _MASK_THREADS_PER_BLOCK[0],
        (int(masks.shape[1]) + _MASK_THREADS_PER_BLOCK[1] - 1) // _MASK_THREADS_PER_BLOCK[1],
        (int(masks.shape[2]) + _MASK_THREADS_PER_BLOCK[2] - 1) // _MASK_THREADS_PER_BLOCK[2],
    )
    rebuild_single_mask_from_objects[launch_blocks, _MASK_THREADS_PER_BLOCK](
        masks,
        obstacle_velocity_x,
        obstacle_velocity_y,
        obstacle_velocity_z,
        pack["local_masks_flat"],
        pack["local_mask_offsets"],
        pack["local_mask_shapes"],
        pack["local_origins"],
        pack["bounds"],
        pack["inv_mats"],
        pack["rates"],
        pack["active_flags"],
        np.float32(delta),
        np.float32(origin_x),
        np.float32(origin_y),
        np.float32(origin_z),
        compute_velocity_flag,
    )
    return masks


