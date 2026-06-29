import math
import numpy as np
from numba import njit, prange

import Solver.General.update_masks as helper_update_masks

_MASK_THREADS_PER_BLOCK = (8, 8, 8)
_PACKED_MASK_CACHE = {}
_PACKED_VALUE_CACHE = {}


@njit(cache=True, parallel=True)
def update_source_mask(
    mask_stack,
    aggregate_mask,
    local_masks_flat,
    local_mask_offsets,
    local_mask_shapes,
    local_origins,
    target_indices,
    bounds,
    inv_mats,
    active_flags,
    delta,
    ox,
    oy,
    oz,
):
    """Rebuild all per-source masks and the aggregate source mask in one pass."""
    mask_count, nx, ny, nz = mask_stack.shape
    total = nx * ny * nz
    obj_count = target_indices.shape[0]

    for n in prange(total):
        i = n // (ny * nz)
        rem = n - i * ny * nz
        j = rem // nz
        k = rem - j * nz

        for mask_idx in range(mask_count):
            mask_stack[mask_idx, i, j, k] = False

        aggregate_active = False
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

        aggregate_mask[i, j, k] = aggregate_active


@njit(cache=True, parallel=True)
def update_source_value_stack(
    value_stack,
    local_masks_flat,
    local_values_flat,
    local_mask_offsets,
    local_mask_shapes,
    local_origins,
    target_indices,
    bounds,
    inv_mats,
    active_flags,
    delta,
    ox,
    oy,
    oz,
):
    """Rebuild stacked per-source scalar fields from local source values."""
    source_count, nx, ny, nz = value_stack.shape
    total = nx * ny * nz
    obj_count = target_indices.shape[0]

    for n in prange(total):
        i = n // (ny * nz)
        rem = n - i * ny * nz
        j = rem // nz
        k = rem - j * nz

        for source_idx in range(source_count):
            value_stack[source_idx, i, j, k] = 0.0

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
                    value_stack[target_idx, i, j, k] = local_values_flat[flat_index]


@njit(cache=True, parallel=True)
def update_obstacle_mask(
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
):
    """Rebuild the obstacle mask and obstacle velocities with last-write-wins overlap."""
    nx, ny, nz = mask.shape
    total = nx * ny * nz
    obj_count = active_flags.shape[0]

    for n in prange(total):
        i = n // (ny * nz)
        rem = n - i * ny * nz
        j = rem // nz
        k = rem - j * nz

        mask[i, j, k] = False
        out_vx[i, j, k] = 0.0
        out_vy[i, j, k] = 0.0
        out_vz[i, j, k] = 0.0

        cell_active = False
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


def _update_frame_state(pack, t, delta, origin_x, origin_y, origin_z, shape):
    """Refresh per-object transforms, bounds and activity flags for the current frame."""
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

    pack["bounds"] = bounds
    pack["inv_mats"] = inv_mats
    pack["rates"] = rates
    pack["active_flags"] = active_flags
    return True


def update_masks(
    masks,
    base_masks,
    t,
    delta,
    origin_x,
    origin_y,
    origin_z,
    obstacle_velocity_x=None,
    obstacle_velocity_y=None,
    obstacle_velocity_z=None,
    aggregate_mask=None,
):
    """Update either stacked source masks or one obstacle mask using cached packed object data."""
    cache_key = id(base_masks)
    pack = _PACKED_MASK_CACHE.get(cache_key)
    if pack is None:
        flat_masks = []
        offsets = [0]
        shapes = []
        origins = []
        target_indices = []
        object_entries = []

        if base_masks:
            first_entry = base_masks[0]
            if isinstance(first_entry, dict):
                iterable = ((0, mask_entry) for mask_entry in base_masks)
            else:
                iterable = (
                    (target_idx, mask_entry)
                    for target_idx, mask_entries in enumerate(base_masks)
                    for mask_entry in mask_entries
                )

            for target_idx, mask_entry in iterable:
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

        mask_shapes = (
            np.asarray(shapes, dtype=np.int32)
            if shapes
            else np.zeros((0, 3), dtype=np.int32)
        )
        mask_origins = (
            np.asarray(origins, dtype=np.float32)
            if origins
            else np.zeros((0, 3), dtype=np.float32)
        )
        target_indices_array = np.asarray(target_indices, dtype=np.int32)

        pack = {
            "object_entries": object_entries,
            "local_masks_flat": local_masks_flat,
            "local_mask_offsets": np.asarray(offsets, dtype=np.int32),
            "local_mask_shapes": mask_shapes,
            "local_origins": mask_origins,
            "target_indices": target_indices_array,
            "bounds": np.zeros((object_count, 6), dtype=np.int32),
            "inv_mats": np.zeros((object_count, 12), dtype=np.float32),
            "rates": np.zeros((object_count, 12), dtype=np.float32),
            "active_flags": np.zeros(object_count, dtype=np.bool_),
            "count": object_count,
        }
        _PACKED_MASK_CACHE[cache_key] = pack
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
        update_source_mask(
            masks,
            aggregate_mask,
            pack["local_masks_flat"],
            pack["local_mask_offsets"],
            pack["local_mask_shapes"],
            pack["local_origins"],
            pack["target_indices"],
            pack["bounds"],
            pack["inv_mats"],
            pack["active_flags"],
            np.float32(delta),
            np.float32(origin_x),
            np.float32(origin_y),
            np.float32(origin_z),
        )
        return masks

    update_obstacle_mask(
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
    )
    return masks


def update_source_values(
    value_stack,
    base_values,
    t,
    delta,
    origin_x,
    origin_y,
    origin_z,
):
    """Update stacked per-source scalar fields using cached packed local values."""
    cache_key = id(base_values)
    pack = _PACKED_VALUE_CACHE.get(cache_key)
    if pack is None:
        flat_masks = []
        flat_values = []
        offsets = [0]
        shapes = []
        origins = []
        target_indices = []
        object_entries = []

        for target_idx, value_entries in enumerate(base_values):
            for value_entry in value_entries:
                base_voxels = value_entry["voxels"]
                base_mask = np.ascontiguousarray(base_voxels["mask"], dtype=np.bool_)
                base_values_local = np.ascontiguousarray(value_entry["values"], dtype=np.float32)
                flat_masks.append(base_mask.reshape(-1))
                flat_values.append(base_values_local.reshape(-1))
                offsets.append(offsets[-1] + base_mask.size)
                shapes.append(base_mask.shape)
                origins.append(np.asarray(base_voxels["origin"], dtype=np.float32))
                target_indices.append(target_idx)
                object_entries.append(
                    {
                        "mesh_object": value_entry["mesh_object"],
                        "voxels": base_voxels,
                    }
                )

        object_count = len(object_entries)
        if flat_masks:
            local_masks_flat = np.concatenate(flat_masks).astype(np.bool_, copy=False)
            local_values_flat = np.concatenate(flat_values).astype(np.float32, copy=False)
        else:
            local_masks_flat = np.empty(0, dtype=np.bool_)
            local_values_flat = np.empty(0, dtype=np.float32)

        mask_shapes = (
            np.asarray(shapes, dtype=np.int32)
            if shapes
            else np.zeros((0, 3), dtype=np.int32)
        )
        mask_origins = (
            np.asarray(origins, dtype=np.float32)
            if origins
            else np.zeros((0, 3), dtype=np.float32)
        )
        target_indices_array = np.asarray(target_indices, dtype=np.int32)

        pack = {
            "object_entries": object_entries,
            "local_masks_flat": local_masks_flat,
            "local_values_flat": local_values_flat,
            "local_mask_offsets": np.asarray(offsets, dtype=np.int32),
            "local_mask_shapes": mask_shapes,
            "local_origins": mask_origins,
            "target_indices": target_indices_array,
            "bounds": np.zeros((object_count, 6), dtype=np.int32),
            "inv_mats": np.zeros((object_count, 12), dtype=np.float32),
            "rates": np.zeros((object_count, 12), dtype=np.float32),
            "active_flags": np.zeros(object_count, dtype=np.bool_),
            "count": object_count,
        }
        _PACKED_VALUE_CACHE[cache_key] = pack

    if not _update_frame_state(
        pack,
        t,
        delta,
        origin_x,
        origin_y,
        origin_z,
        value_stack.shape[-3:],
    ):
        return value_stack

    update_source_value_stack(
        value_stack,
        pack["local_masks_flat"],
        pack["local_values_flat"],
        pack["local_mask_offsets"],
        pack["local_mask_shapes"],
        pack["local_origins"],
        pack["target_indices"],
        pack["bounds"],
        pack["inv_mats"],
        pack["active_flags"],
        np.float32(delta),
        np.float32(origin_x),
        np.float32(origin_y),
        np.float32(origin_z),
    )
    return value_stack
