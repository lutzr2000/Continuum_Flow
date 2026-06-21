import math
from numba import cuda
import numpy as np
import Solver.General.update_masks as helper_update_masks

_MASK_THREADS_PER_BLOCK = (8, 8, 8)


@cuda.jit(cache=True)
def sample_mask_backwards(
    out,
    out_vx,
    out_vy,
    out_vz,
    base,
    delta,
    ox,
    oy,
    oz,
    ix0,
    ix1,
    iy0,
    iy1,
    iz0,
    iz1,
    box,
    inv,
    rate,
):
    """Back-sample a local reference mask into the world grid and evaluate wall velocity."""
    i, j, k = cuda.grid(3)
    nx, ny, nz = out.shape

    if i >= nx or j >= ny or k >= nz:
        return
    if i < ix0 or i > ix1 or j < iy0 or j > iy1 or k < iz0 or k > iz1:
        return

    base_ox, base_oy, base_oz = box
    bn_x, bn_y, bn_z = base.shape

    x = np.float32(ox + i * delta)
    y = np.float32(oy + j * delta)
    z = np.float32(oz + k * delta)

    bx = inv[0, 0] * x + inv[0, 1] * y + inv[0, 2] * z + inv[0, 3]
    by = inv[1, 0] * x + inv[1, 1] * y + inv[1, 2] * z + inv[1, 3]
    bz = inv[2, 0] * x + inv[2, 1] * y + inv[2, 2] * z + inv[2, 3]

    bi = int(math.floor((bx - base_ox) / delta + 0.5))
    bj = int(math.floor((by - base_oy) / delta + 0.5))
    bk = int(math.floor((bz - base_oz) / delta + 0.5))

    if 0 <= bi < bn_x and 0 <= bj < bn_y and 0 <= bk < bn_z and base[bi, bj, bk]:
        out[i, j, k] = True
        out_vx[i, j, k] = (
            rate[0, 0] * bx + rate[0, 1] * by + rate[0, 2] * bz + rate[0, 3]
        )
        out_vy[i, j, k] = (
            rate[1, 0] * bx + rate[1, 1] * by + rate[1, 2] * bz + rate[1, 3]
        )
        out_vz[i, j, k] = (
            rate[2, 0] * bx + rate[2, 1] * by + rate[2, 2] * bz + rate[2, 3]
        )

def _update_one_mask(
    mask,
    base_masks,
    t,
    delta,
    origin_x,
    origin_y,
    origin_z,
    obstacle_velocity_x,
    obstacle_velocity_y,
    obstacle_velocity_z,
):
    origin = np.asarray((origin_x, origin_y, origin_z), dtype=np.float32)

    for mask_entry in base_masks:
        mesh_object = mask_entry["mesh_object"]
        base_voxels = mask_entry["voxels"]
        base = mask_entry.get("_base_device")
        if base is None:
            base = cuda.to_device(np.ascontiguousarray(base_voxels["mask"], dtype=np.bool_))
            mask_entry["_base_device"] = base
        box = np.asarray(base_voxels["origin"], dtype=np.float32)

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
            shape=mask.shape,
        )

        if ix0 > ix1 or iy0 > iy1 or iz0 > iz1:
            continue

        sample_mask_backwards[
            (
                (int(mask.shape[0]) + _MASK_THREADS_PER_BLOCK[0] - 1) // _MASK_THREADS_PER_BLOCK[0],
                (int(mask.shape[1]) + _MASK_THREADS_PER_BLOCK[1] - 1) // _MASK_THREADS_PER_BLOCK[1],
                (int(mask.shape[2]) + _MASK_THREADS_PER_BLOCK[2] - 1) // _MASK_THREADS_PER_BLOCK[2],
            ),
            _MASK_THREADS_PER_BLOCK,
        ](
            mask,
            obstacle_velocity_x,
            obstacle_velocity_y,
            obstacle_velocity_z,
            base,
            np.float32(delta),
            np.float32(origin_x),
            np.float32(origin_y),
            np.float32(origin_z),
            int(ix0),
            int(ix1),
            int(iy0),
            int(iy1),
            int(iz0),
            int(iz1),
            box,
            inv,
            rate,
        )

    return mask


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
):
    if isinstance(masks, (list, tuple)):
        updated_masks = []
        mask_count = min(len(masks), len(base_masks))
        for idx in range(mask_count):
            updated_masks.append(
                _update_one_mask(
                    masks[idx],
                    base_masks[idx],
                    t,
                    delta,
                    origin_x,
                    origin_y,
                    origin_z,
                    obstacle_velocity_x,
                    obstacle_velocity_y,
                    obstacle_velocity_z,
                )
            )
        return updated_masks

    return _update_one_mask(
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
    )
