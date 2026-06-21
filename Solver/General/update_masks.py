from numba import njit, prange
import numpy as np


@njit(cache=True, parallel=True)
def sample_mask_backwards(
    out, base, delta, ox, oy, oz, ix0, ix1, iy0, iy1, iz0, iz1, box, inv
):
    """Back-sample a local reference mask into the world grid."""
    base_ox, base_oy, base_oz = box
    bn_x, bn_y, bn_z = base.shape
    sx, sy, sz = ix1 - ix0 + 1, iy1 - iy0 + 1, iz1 - iz0 + 1

    for n in prange(sx * sy * sz):
        i = ix0 + n // (sy * sz)
        r = n % (sy * sz)
        j = iy0 + r // sz
        k = iz0 + r % sz

        x = np.float32(ox + i * delta)
        y = np.float32(oy + j * delta)
        z = np.float32(oz + k * delta)

        bx = inv[0, 0] * x + inv[0, 1] * y + inv[0, 2] * z + inv[0, 3]
        by = inv[1, 0] * x + inv[1, 1] * y + inv[1, 2] * z + inv[1, 3]
        bz = inv[2, 0] * x + inv[2, 1] * y + inv[2, 2] * z + inv[2, 3]

        bi = int(np.floor((bx - base_ox) / delta + 0.5))
        bj = int(np.floor((by - base_oy) / delta + 0.5))
        bk = int(np.floor((bz - base_oz) / delta + 0.5))

        if 0 <= bi < bn_x and 0 <= bj < bn_y and 0 <= bk < bn_z and base[bi, bj, bk]:
            out[i, j, k] = True


def _as_f32(a):
    """Return a contiguous float32 view/copy of the given array-like input."""
    return np.ascontiguousarray(a, dtype=np.float32)


def _bounds_to_indices(
    bounds_min, bounds_max, delta, origin=(0.0, 0.0, 0.0), shape=None
):
    """Convert world-space bounds into inclusive voxel index bounds."""
    origin = np.asarray(origin, dtype=np.float32)
    lo = np.floor((bounds_min - origin) / delta).astype(np.int32)
    hi = np.ceil((bounds_max - origin) / delta).astype(np.int32)

    hi_limit = np.asarray(shape, dtype=np.int32) - 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, hi_limit)

    return int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]), int(lo[2]), int(hi[2])


def _transform_bounds(bounds_center, bounds_extent, matrix):
    """Transform one local AABB into world space using center/extent form."""
    matrix = np.asarray(matrix, dtype=np.float32)
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    center_world = _as_f32(linear @ bounds_center + translation)
    extent_world = _as_f32(np.abs(linear) @ bounds_extent)
    return center_world - extent_world, center_world + extent_world


def _bounds_center_extent(bounds_min, bounds_max):
    """Convert min/max bounds into center/extent form for cheap affine transforms."""
    bounds_min = np.asarray(bounds_min, dtype=np.float32)
    bounds_max = np.asarray(bounds_max, dtype=np.float32)
    center = (bounds_min + bounds_max) * np.float32(0.5)
    extent = (bounds_max - bounds_min) * np.float32(0.5)
    return _as_f32(center), _as_f32(extent)


def _initial_world_matrix(mesh_object):
    """Return the first exported world transform for one mesh object."""
    animation = mesh_object.get("transform_animation") or {}
    matrices = np.asarray(
        animation.get("matrices_world", (np.eye(4, dtype=np.float32),)),
        dtype=np.float32,
    ).reshape((-1, 4, 4))
    if matrices.size == 0:
        return np.eye(4, dtype=np.float32)
    return _as_f32(matrices[0])


def _invert_affine_matrix(matrix):
    """Invert one affine 4x4 matrix using its linear part and translation."""
    matrix = np.asarray(matrix, dtype=np.float32)
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    inv_linear = _as_f32(np.linalg.inv(linear))
    inv_translation = _as_f32(-(inv_linear @ translation))

    inv = np.eye(4, dtype=np.float32)
    inv[:3, :3] = inv_linear
    inv[:3, 3] = inv_translation
    return inv


def _world_matrix_at_time(mesh_object, time_value):
    animation = mesh_object.get("transform_animation") or {}
    times = np.asarray(animation.get("times", (0.0,)), dtype=np.float32)
    matrices = np.asarray(
        animation.get("matrices_world", (np.eye(4, dtype=np.float32),)),
        dtype=np.float32,
    ).reshape((-1, 4, 4))

    if matrices.size == 0:
        return np.eye(4, dtype=np.float32)
    if times.size <= 1 or matrices.shape[0] <= 1:
        return _as_f32(matrices[0])
    if time_value <= float(times[0]):
        return _as_f32(matrices[0])
    if time_value >= float(times[-1]):
        return _as_f32(matrices[min(len(matrices) - 1, len(times) - 1)])

    last_segment = min(len(times), len(matrices)) - 1
    for idx in range(last_segment):
        t0 = float(times[idx])
        t1 = float(times[idx + 1])
        if time_value <= t1:
            if t1 <= t0:
                return _as_f32(matrices[idx])
            alpha = np.float32((time_value - t0) / (t1 - t0))
            return _as_f32(matrices[idx] * (1.0 - alpha) + matrices[idx + 1] * alpha)

    return _as_f32(matrices[last_segment])


def update_masks(
    mask,
    base_masks,
    t,
    delta,
    origin_x,
    origin_y,
    origin_z,
):

    origin = np.asarray((origin_x, origin_y, origin_z), dtype=np.float32)

    mask.fill(False)

    for obstacle_entry in base_masks:
        mesh_object = obstacle_entry["mesh_object"]
        base_voxels = obstacle_entry["voxels"]
        base = base_voxels["mask"]
        box = base_voxels["origin"]

        object_matrix = _world_matrix_at_time(mesh_object, t)
        bounds_center, bounds_extent = _bounds_center_extent(
            base_voxels["bounds_min"],
            base_voxels["bounds_max"],
        )
        bounds_min, bounds_max = _transform_bounds(
            bounds_center,
            bounds_extent,
            object_matrix,
        )
        inv = _invert_affine_matrix(object_matrix)

        ix0, ix1, iy0, iy1, iz0, iz1 = _bounds_to_indices(
            bounds_min,
            bounds_max,
            delta,
            origin,
            shape=mask.shape,
        )

        if ix0 > ix1 or iy0 > iy1 or iz0 > iz1:
            continue

        sample_mask_backwards(
            mask,
            base,
            delta,
            origin_x,
            origin_y,
            origin_z,
            ix0,
            ix1,
            iy0,
            iy1,
            iz0,
            iz1,
            box,
            inv,
        )

    return mask
