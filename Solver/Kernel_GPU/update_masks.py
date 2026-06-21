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
    base_ox,
    base_oy,
    base_oz,
    inv00,
    inv01,
    inv02,
    inv03,
    inv10,
    inv11,
    inv12,
    inv13,
    inv20,
    inv21,
    inv22,
    inv23,
    rate00,
    rate01,
    rate02,
    rate03,
    rate10,
    rate11,
    rate12,
    rate13,
    rate20,
    rate21,
    rate22,
    rate23,
    compute_velocity_flag=True
):
    """Back-sample a local reference mask into the world grid and evaluate wall velocity."""
    i, j, k = cuda.grid(3)
    nx, ny, nz = out.shape

    if i >= nx or j >= ny or k >= nz:
        return
    if i < ix0 or i > ix1 or j < iy0 or j > iy1 or k < iz0 or k > iz1:
        return

    bn_x, bn_y, bn_z = base.shape

    x = np.float32(ox + i * delta)
    y = np.float32(oy + j * delta)
    z = np.float32(oz + k * delta)

    bx = inv00 * x + inv01 * y + inv02 * z + inv03
    by = inv10 * x + inv11 * y + inv12 * z + inv13
    bz = inv20 * x + inv21 * y + inv22 * z + inv23

    bi = int(math.floor((bx - base_ox) / delta + 0.5))
    bj = int(math.floor((by - base_oy) / delta + 0.5))
    bk = int(math.floor((bz - base_oz) / delta + 0.5))

    if 0 <= bi < bn_x and 0 <= bj < bn_y and 0 <= bk < bn_z and base[bi, bj, bk]:
        out[i, j, k] = True
        if compute_velocity_flag:
            out_vx[i, j, k] = rate00 * bx + rate01 * by + rate02 * bz + rate03
            out_vy[i, j, k] = rate10 * bx + rate11 * by + rate12 * bz + rate13
            out_vz[i, j, k] = rate20 * bx + rate21 * by + rate22 * bz + rate23


@cuda.jit(cache=True)
def clear_mask_and_velocity(mask, out_vx, out_vy, out_vz):
    i, j, k = cuda.grid(3)
    nx, ny, nz = mask.shape

    if i >= nx or j >= ny or k >= nz:
        return

    mask[i, j, k] = False
    out_vx[i, j, k] = 0.0
    out_vy[i, j, k] = 0.0
    out_vz[i, j, k] = 0.0


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
    compute_velocity_flag
):
    origin = np.asarray((origin_x, origin_y, origin_z), dtype=np.float32)
    launch_blocks = (
        (int(mask.shape[0]) + _MASK_THREADS_PER_BLOCK[0] - 1) // _MASK_THREADS_PER_BLOCK[0],
        (int(mask.shape[1]) + _MASK_THREADS_PER_BLOCK[1] - 1) // _MASK_THREADS_PER_BLOCK[1],
        (int(mask.shape[2]) + _MASK_THREADS_PER_BLOCK[2] - 1) // _MASK_THREADS_PER_BLOCK[2],
    )

    clear_mask_and_velocity[launch_blocks, _MASK_THREADS_PER_BLOCK](
        mask,
        obstacle_velocity_x,
        obstacle_velocity_y,
        obstacle_velocity_z,
    )

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

        sample_mask_backwards[launch_blocks, _MASK_THREADS_PER_BLOCK](
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
            np.float32(box[0]),
            np.float32(box[1]),
            np.float32(box[2]),
            np.float32(inv[0, 0]),
            np.float32(inv[0, 1]),
            np.float32(inv[0, 2]),
            np.float32(inv[0, 3]),
            np.float32(inv[1, 0]),
            np.float32(inv[1, 1]),
            np.float32(inv[1, 2]),
            np.float32(inv[1, 3]),
            np.float32(inv[2, 0]),
            np.float32(inv[2, 1]),
            np.float32(inv[2, 2]),
            np.float32(inv[2, 3]),
            np.float32(rate[0, 0]),
            np.float32(rate[0, 1]),
            np.float32(rate[0, 2]),
            np.float32(rate[0, 3]),
            np.float32(rate[1, 0]),
            np.float32(rate[1, 1]),
            np.float32(rate[1, 2]),
            np.float32(rate[1, 3]),
            np.float32(rate[2, 0]),
            np.float32(rate[2, 1]),
            np.float32(rate[2, 2]),
            np.float32(rate[2, 3]),
            compute_velocity_flag
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
    compute_velocity_flag
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
                    compute_velocity_flag
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
        compute_velocity_flag
    )
