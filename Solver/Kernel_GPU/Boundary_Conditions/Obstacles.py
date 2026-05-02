import math

import numpy as np
from numba import cuda, njit, prange

from Solver.Kernel_GPU.Kernel_Config import THREADS_PER_BLOCK_3D, volume_blocks_per_grid


# -----------------------------------------------------------------------------
# Small numba kernels
# -----------------------------------------------------------------------------

@njit(cache=True)
def _sort_prefix(a, n):
    """Sort a[:n] in-place. Small per-scanline arrays make insertion sort fine."""
    for i in range(1, n):
        x = a[i]
        j = i - 1
        while j >= 0 and a[j] > x:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = x


@njit(cache=True)
def _barycentric_yz(y, z, tri, eps):
    """Barycentric coordinates of point (y, z) in tri projected to the YZ plane."""
    y0, z0 = tri[0, 1], tri[0, 2]
    y1, z1 = tri[1, 1], tri[1, 2]
    y2, z2 = tri[2, 1], tri[2, 2]

    den = (z1 - z2) * (y0 - y2) + (y2 - y1) * (z0 - z2)
    if abs(den) <= eps:
        return False, 0.0, 0.0, 0.0

    w0 = ((z1 - z2) * (y - y2) + (y2 - y1) * (z - z2)) / den
    w1 = ((z2 - z0) * (y - y2) + (y0 - y2) * (z - z2)) / den
    w2 = 1.0 - w0 - w1
    return w0 >= -eps and w1 >= -eps and w2 >= -eps, w0, w1, w2


@njit(cache=True)
def _scanline_pass(iy0, iy1, iz0, iz1, z_span, line_counts, offsets, candidates, write):
    """
    Shared scanline pass.
    - If candidates.size == 0: count touched scanlines in line_counts.
    - Else: write triangle ids into candidates using write positions.
    """
    for t in range(iy0.shape[0]):
        for y in range(iy0[t], iy1[t] + 1):
            base = y * z_span
            for z in range(iz0[t], iz1[t] + 1):
                line = base + z
                if candidates.size == 0:
                    line_counts[line] += 1
                else:
                    pos = write[line]
                    candidates[pos] = t
                    write[line] = pos + 1


@njit(cache=True, parallel=True)
def _fill_scanlines(mask, triangles, delta, ox, oy, oz, line_offsets, candidates):
    """Fill a cropped mask by ray casting along X for every YZ scanline."""
    eps = np.float32(delta * 1.0e-5 + 1.0e-7)
    nx, ny, nz = mask.shape
    line_count = ny * nz

    for line in prange(line_count):
        j = line // nz
        k = line % nz
        start = line_offsets[line]
        end = line_offsets[line + 1]
        if end - start < 2:
            continue

        y = np.float32(oy + j * delta)
        z = np.float32(oz + k * delta)
        xs = np.empty(end - start, dtype=np.float32)
        n = 0

        for p in range(start, end):
            tri = triangles[candidates[p]]
            ok, w0, w1, w2 = _barycentric_yz(y, z, tri, eps)
            if ok:
                xs[n] = np.float32(w0 * tri[0, 0] + w1 * tri[1, 0] + w2 * tri[2, 0])
                n += 1

        if n < 2:
            continue

        _sort_prefix(xs, n)

        u = 0
        for p in range(n):
            if u == 0 or abs(xs[p] - xs[u - 1]) > eps:
                xs[u] = xs[p]
                u += 1

        for p in range(0, u - u % 2, 2):
            i0 = max(0, int(np.ceil((xs[p] - ox - eps) / delta)))
            i1 = min(nx - 1, int(np.floor((xs[p + 1] - ox + eps) / delta)))
            for i in range(i0, i1 + 1):
                mask[i, j, k] = True


@njit(cache=True, parallel=True)
def _sample_mask_backwards(out, base, delta, ox, oy, oz, ix0, ix1, iy0, iy1, iz0, iz1, box, inv):
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


@njit(cache=True, parallel=True)
def _sample_obstacle_data_backwards(
    out_mask, out_vx, out_vy, out_vz,
    base, delta, ox, oy, oz,
    ix0, ix1, iy0, iy1, iz0, iz1,
    box, inv, rate
):
    """Back-sample one moving obstacle mask and its wall velocity into the world grid."""
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
            out_mask[i, j, k] = True
            out_vx[i, j, k] = rate[0, 0] * bx + rate[0, 1] * by + rate[0, 2] * bz + rate[0, 3]
            out_vy[i, j, k] = rate[1, 0] * bx + rate[1, 1] * by + rate[1, 2] * bz + rate[1, 3]
            out_vz[i, j, k] = rate[2, 0] * bx + rate[2, 1] * by + rate[2, 2] * bz + rate[2, 3]


@cuda.jit(cache=True)
def _clear_mask_cuda(mask):
    i, j, k = cuda.grid(3)
    nx, ny, nz = mask.shape
    if i < nx and j < ny and k < nz:
        mask[i, j, k] = False


@cuda.jit(cache=True)
def _clear_mask_box_cuda(mask, ix0, iy0, iz0, sx, sy, sz):
    di, dj, dk = cuda.grid(3)
    if di < sx and dj < sy and dk < sz:
        mask[ix0 + di, iy0 + dj, iz0 + dk] = False


@cuda.jit(cache=True)
def _clear_obstacle_fields_cuda(mask, velocity_x, velocity_y, velocity_z):
    i, j, k = cuda.grid(3)
    nx, ny, nz = mask.shape
    if i < nx and j < ny and k < nz:
        mask[i, j, k] = False
        velocity_x[i, j, k] = 0.0
        velocity_y[i, j, k] = 0.0
        velocity_z[i, j, k] = 0.0


@cuda.jit(cache=True)
def _clear_obstacle_fields_box_cuda(mask, velocity_x, velocity_y, velocity_z, ix0, iy0, iz0, sx, sy, sz):
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
def _sample_mask_backwards_object_cuda(
    out,
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
    delta,
    ox, oy, oz,
):
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
            out[i, j, k] = True


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


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def _eps(delta):
    return np.float32(np.float32(delta) * 1.0e-5 + 1.0e-7)


def _as_f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def _triangle_bounds(triangles):
    v = triangles.reshape(-1, 3)
    return v.min(axis=0).astype(np.float32), v.max(axis=0).astype(np.float32)


def _bounds_center_extent(bounds_min, bounds_max):
    bounds_min = np.asarray(bounds_min, dtype=np.float32)
    bounds_max = np.asarray(bounds_max, dtype=np.float32)
    center = (bounds_min + bounds_max) * np.float32(0.5)
    extent = (bounds_max - bounds_min) * np.float32(0.5)
    return _as_f32(center), _as_f32(extent)


def _bounds_to_indices(bounds_min, bounds_max, delta, origin=(0.0, 0.0, 0.0), shape=None):
    origin = np.asarray(origin, dtype=np.float32)
    lo = np.floor((bounds_min - origin) / delta).astype(np.int32)
    hi = np.ceil((bounds_max - origin) / delta).astype(np.int32)

    if shape is not None:
        hi_limit = np.asarray(shape, dtype=np.int32) - 1
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, hi_limit)

    return int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]), int(lo[2]), int(hi[2])


def _object_bounds(mesh_object, triangles):
    bounds = mesh_object.get("bounds") or {}
    if bounds:
        return (
            np.asarray(bounds.get("min", (0.0, 0.0, 0.0)), dtype=np.float32),
            np.asarray(bounds.get("max", (0.0, 0.0, 0.0)), dtype=np.float32),
        )
    return _triangle_bounds(triangles)


def _load_mesh_triangles(mesh_object):
    """Load one mesh object's triangle payload into contiguous float32 shape (n, 3, 3)."""
    if mesh_object.get("triangles_file"):
        triangles = np.load(mesh_object["triangles_file"], allow_pickle=False)
    else:
        triangles = mesh_object.get("triangles", ())

    triangles = np.asarray(triangles, dtype=np.float32)
    if triangles.size == 0:
        return np.empty((0, 3, 3), dtype=np.float32)

    shape = mesh_object.get("triangles_shape") or (-1, 3, 3)
    return _as_f32(triangles.reshape(tuple(map(int, shape))))


# -----------------------------------------------------------------------------
# Static voxelization
# -----------------------------------------------------------------------------

def _scanline_candidates(triangles, delta, origin_y, origin_z, iy_min, iy_max, iz_min, iz_max):
    """Build compact candidate triangle lists for all cropped YZ scanlines."""
    line_count = int((iy_max - iy_min + 1) * (iz_max - iz_min + 1))
    empty_offsets = np.zeros(max(line_count, 0) + 1, dtype=np.int32)
    if triangles.size == 0 or line_count <= 0:
        return empty_offsets, np.empty(0, dtype=np.int32), np.empty((0, 3, 3), dtype=np.float32)

    eps = _eps(delta)
    yz = triangles[:, :, 1:3]
    den = (yz[:, 1, 1] - yz[:, 2, 1]) * (yz[:, 0, 0] - yz[:, 2, 0]) + \
          (yz[:, 2, 0] - yz[:, 1, 0]) * (yz[:, 0, 1] - yz[:, 2, 1])
    triangles = _as_f32(triangles[np.abs(den) > eps])
    if triangles.size == 0:
        return empty_offsets, np.empty(0, dtype=np.int32), triangles

    yz_min = triangles[:, :, 1:3].min(axis=1)
    yz_max = triangles[:, :, 1:3].max(axis=1)
    origin = np.asarray((origin_y, origin_z), dtype=np.float32)
    local_min = np.floor((yz_min - origin - eps) / delta).astype(np.int32) - np.asarray((iy_min, iz_min), dtype=np.int32)
    local_max = np.ceil((yz_max - origin + eps) / delta).astype(np.int32) - np.asarray((iy_min, iz_min), dtype=np.int32)

    limit = np.asarray((iy_max - iy_min, iz_max - iz_min), dtype=np.int32)
    local_min = np.minimum(np.maximum(local_min, 0), limit)
    local_max = np.minimum(np.maximum(local_max, 0), limit)

    z_span = int(limit[1] + 1)
    counts = np.zeros(line_count, dtype=np.int32)
    dummy = np.empty(0, dtype=np.int32)
    _scanline_pass(local_min[:, 0], local_max[:, 0], local_min[:, 1], local_max[:, 1], z_span, counts, empty_offsets, dummy, counts)

    offsets = np.empty(line_count + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, dtype=np.int32, out=offsets[1:])

    candidates = np.empty(int(offsets[-1]), dtype=np.int32)
    write = offsets[:-1].copy()
    _scanline_pass(local_min[:, 0], local_max[:, 0], local_min[:, 1], local_max[:, 1], z_span, counts, offsets, candidates, write)
    return offsets, candidates, triangles


def _voxelize_cropped(triangles, delta, origin, index_bounds):
    """Voxelize triangles into the inclusive cropped index bounds."""
    ix0, ix1, iy0, iy1, iz0, iz1 = index_bounds
    if triangles.size == 0 or ix0 > ix1 or iy0 > iy1 or iz0 > iz1:
        return None

    delta = np.float32(delta)
    origin = np.asarray(origin, dtype=np.float32)
    mask = np.zeros((ix1 - ix0 + 1, iy1 - iy0 + 1, iz1 - iz0 + 1), dtype=np.bool_)

    offsets, candidates, valid_triangles = _scanline_candidates(
        triangles, delta, origin[1], origin[2], iy0, iy1, iz0, iz1
    )
    if candidates.size == 0:
        return None

    cropped_origin = _as_f32(origin + delta * np.asarray((ix0, iy0, iz0), dtype=np.float32))
    _fill_scanlines(mask, valid_triangles, delta, cropped_origin[0], cropped_origin[1], cropped_origin[2], offsets, candidates)

    return {
        "mask": np.ascontiguousarray(mask),
        "origin": cropped_origin,
        "bounds_min": cropped_origin,
        "bounds_max": _as_f32(origin + delta * np.asarray((ix1, iy1, iz1), dtype=np.float32)),
        "index_min": np.asarray((ix0, iy0, iz0), dtype=np.int32),
        "index_max": np.asarray((ix1, iy1, iz1), dtype=np.int32),
    }


def mesh(nx, ny, nz, delta, mesh_objects, origin_x=0.0, origin_y=0.0, origin_z=0.0):
    """Create one combined boolean mask for static Blender-exported triangle meshes."""
    out = np.zeros((nx, ny, nz), dtype=np.bool_)
    if not mesh_objects:
        return out

    origin = np.asarray((origin_x, origin_y, origin_z), dtype=np.float32)
    delta = np.float32(delta)

    for obj in mesh_objects:
        triangles = _load_mesh_triangles(obj)
        if triangles.size == 0:
            continue

        bounds = _object_bounds(obj, triangles)
        index_bounds = _bounds_to_indices(bounds[0], bounds[1], delta, origin, shape=out.shape)
        voxels = _voxelize_cropped(triangles, delta, origin, index_bounds)
        if voxels is None or not np.any(voxels["mask"]):
            continue

        ix0, ix1, iy0, iy1, iz0, iz1 = index_bounds
        out[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1] |= voxels["mask"]

    return out


# -----------------------------------------------------------------------------
# Dynamic runtime
# -----------------------------------------------------------------------------

def _matrix_series(animation):
    """Normalize exported transform samples into a runtime-friendly series."""
    if not animation:
        return {"times": np.zeros(1, dtype=np.float32), "matrices_world": np.eye(4, dtype=np.float32)[None], "cursor": 0}

    times = np.asarray(animation.get("times", (0.0,)), dtype=np.float32)
    matrices = np.asarray(animation.get("matrices_world", (np.eye(4, dtype=np.float32),)), dtype=np.float32).reshape((-1, 4, 4))
    n = min(max(times.size, 1), matrices.shape[0])
    return {
        "times": _as_f32(times[:n] if times.size else np.zeros(1, dtype=np.float32)),
        "matrices_world": _as_f32(matrices[:n] if matrices.size else np.eye(4, dtype=np.float32)[None]),
        "cursor": 0,
    }


def _transform_series_is_animated(series):
    """Return whether a transform series can change over time."""
    times = series["times"]
    matrices = series["matrices_world"]
    if int(times.size) <= 1 or matrices.shape[0] <= 1:
        return False

    first = np.asarray(matrices[0], dtype=np.float32)
    for idx in range(1, matrices.shape[0]):
        if not np.allclose(matrices[idx], first, rtol=1.0e-5, atol=1.0e-6):
            return True
    return False


def _matrix_and_rate_at(series, t):
    """Return the interpolated world matrix and its piecewise-linear time derivative."""
    times = series["times"]
    matrices = series["matrices_world"]
    if times.size <= 1:
        return (
            np.asarray(matrices[0], dtype=np.float32),
            np.zeros((4, 4), dtype=np.float32),
        )

    if t < float(times[0]):
        return (
            np.asarray(matrices[0], dtype=np.float32),
            np.zeros((4, 4), dtype=np.float32),
        )

    cursor = min(int(series.get("cursor", 0)), int(times.size - 2))
    while cursor < times.size - 2 and t >= float(times[cursor + 1]):
        cursor += 1
    series["cursor"] = cursor

    if cursor >= times.size - 2 and t > float(times[-1]):
        return (
            np.asarray(matrices[-1], dtype=np.float32),
            np.zeros((4, 4), dtype=np.float32),
        )

    t0, t1 = float(times[cursor]), float(times[cursor + 1])
    if t1 <= t0:
        return (
            np.asarray(matrices[cursor], dtype=np.float32),
            np.zeros((4, 4), dtype=np.float32),
        )

    a = np.float32((t - t0) / (t1 - t0))
    matrix0 = np.asarray(matrices[cursor], dtype=np.float32)
    matrix1 = np.asarray(matrices[cursor + 1], dtype=np.float32)
    matrix = _as_f32(matrix0 * (1.0 - a) + matrix1 * a)
    rate = _as_f32((matrix1 - matrix0) / np.float32(t1 - t0))
    return matrix, rate


def _invert_affine_matrix(matrix):
    """Invert a 4x4 affine transform using only the 3x3 linear part and translation."""
    matrix = np.asarray(matrix, dtype=np.float32)
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    inv_linear = _as_f32(np.linalg.inv(linear))
    inv_translation = _as_f32(-(inv_linear @ translation))

    inv = np.eye(4, dtype=np.float32)
    inv[:3, :3] = inv_linear
    inv[:3, 3] = inv_translation
    return inv


def _transform_bounds(bounds_center, bounds_extent, matrix):
    """Transform an AABB using center/extent form to avoid rebuilding its eight corners."""
    matrix = np.asarray(matrix, dtype=np.float32)
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    center_world = _as_f32(linear @ bounds_center + translation)
    extent_world = _as_f32(np.abs(linear) @ bounds_extent)
    return center_world - extent_world, center_world + extent_world


def _resolve_dynamic_object_state(obj, time_value, delta, origin, shape):
    """Cache the expensive per-frame transform, inverse and index bounds for one object."""
    state = obj.get("dynamic_state")
    if state is not None and state.get("time_value") == float(time_value):
        return state

    matrix, matrix_rate = _matrix_and_rate_at(obj["transform_series"], time_value)
    bounds_min, bounds_max = _transform_bounds(
        obj["local_bounds_center"],
        obj["local_bounds_extent"],
        matrix,
    )
    index_bounds = _bounds_to_indices(bounds_min, bounds_max, delta, origin, shape=shape)
    active = not (index_bounds[0] > index_bounds[1] or index_bounds[2] > index_bounds[3] or index_bounds[4] > index_bounds[5])

    state = {
        "time_value": float(time_value),
        "matrix": matrix,
        "matrix_rate": matrix_rate,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "index_bounds": index_bounds,
        "active": bool(active),
    }
    if active:
        state["inv"] = _invert_affine_matrix(matrix)

    obj["dynamic_state"] = state
    return state


def _region_shape(index_bounds):
    ix0, ix1, iy0, iy1, iz0, iz1 = index_bounds
    return (
        int(ix1 - ix0 + 1),
        int(iy1 - iy0 + 1),
        int(iz1 - iz0 + 1),
    )


def _merge_index_bounds(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (
        min(int(a[0]), int(b[0])),
        max(int(a[1]), int(b[1])),
        min(int(a[2]), int(b[2])),
        max(int(a[3]), int(b[3])),
        min(int(a[4]), int(b[4])),
        max(int(a[5]), int(b[5])),
    )


def _voxelize_local(triangles, delta):
    """Voxelize a mesh once in local space for later transform-based sampling."""
    if triangles.size == 0:
        return None
    bounds = _triangle_bounds(triangles)
    voxels = _voxelize_cropped(triangles, delta, (0.0, 0.0, 0.0), _bounds_to_indices(*bounds, delta))
    if voxels is None:
        return None
    return {
        "mask": voxels["mask"],
        "origin": voxels["origin"],
        "bounds_min": voxels["bounds_min"],
        "bounds_max": voxels["bounds_max"],
    }


def build_dynamic_runtime(nx, ny, nz, delta, mesh_objects, origin_x=0.0, origin_y=0.0, origin_z=0.0):
    """Precompute local masks and animation samples for dynamic obstacle/source masks."""
    objects = []
    has_animation = False
    for obj in mesh_objects:
        triangles = _load_mesh_triangles(obj)
        voxels = _voxelize_local(triangles, delta)
        if voxels is None or not np.any(voxels["mask"]):
            continue
        bounds_center, bounds_extent = _bounds_center_extent(voxels["bounds_min"], voxels["bounds_max"])
        transform_series = _matrix_series(obj.get("transform_animation", {}))
        has_animation = has_animation or _transform_series_is_animated(transform_series)

        objects.append({
            "object_name": obj.get("object_name"),
            "local_mask": voxels["mask"],
            "local_origin": voxels["origin"],
            "local_bounds_min": voxels["bounds_min"],
            "local_bounds_max": voxels["bounds_max"],
            "local_bounds_center": bounds_center,
            "local_bounds_extent": bounds_extent,
            "transform_series": transform_series,
            "dynamic_state": None,
            "last_gpu_index_bounds": None,
        })

    return {
        "objects": objects,
        "shape": (int(nx), int(ny), int(nz)),
        "delta": np.float32(delta),
        "origin": np.asarray((origin_x, origin_y, origin_z), dtype=np.float32),
        "is_animated": bool(has_animation),
        "gpu_ready": False,
    }


def prepare_dynamic_runtime_for_gpu(runtime_data):
    """Upload static per-object voxel masks so dynamic sampling can run on the GPU."""
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

    runtime_data["gpu_ready"] = True
    return runtime_data


def update_dynamic_mask(runtime_data, time_value, out_mask=None):
    """Update one combined world-space mask by back-sampling all runtime objects."""
    shape = runtime_data["shape"]
    out = np.zeros(shape, dtype=np.bool_) if out_mask is None else out_mask
    out.fill(False)

    delta = np.float32(runtime_data["delta"])
    origin = np.asarray(runtime_data["origin"], dtype=np.float32)

    for obj in runtime_data["objects"]:
        state = _resolve_dynamic_object_state(obj, time_value, delta, origin, shape)
        if not state["active"]:
            continue
        ix0, ix1, iy0, iy1, iz0, iz1 = state["index_bounds"]

        _sample_mask_backwards(
            out,
            obj["local_mask"],
            delta,
            origin[0], origin[1], origin[2],
            ix0, ix1, iy0, iy1, iz0, iz1,
            obj["local_origin"],
            state["inv"],
        )

    return out


def update_dynamic_obstacle_data(
    runtime_data,
    time_value,
    out_mask=None,
    out_velocity_x=None,
    out_velocity_y=None,
    out_velocity_z=None,
):
    """Update one moving obstacle mask and its wall velocity fields on the host."""
    shape = runtime_data["shape"]
    out_mask = np.zeros(shape, dtype=np.bool_) if out_mask is None else out_mask
    out_velocity_x = np.zeros(shape, dtype=np.float32) if out_velocity_x is None else out_velocity_x
    out_velocity_y = np.zeros(shape, dtype=np.float32) if out_velocity_y is None else out_velocity_y
    out_velocity_z = np.zeros(shape, dtype=np.float32) if out_velocity_z is None else out_velocity_z

    out_mask.fill(False)
    out_velocity_x.fill(0.0)
    out_velocity_y.fill(0.0)
    out_velocity_z.fill(0.0)

    delta = np.float32(runtime_data["delta"])
    origin = np.asarray(runtime_data["origin"], dtype=np.float32)

    for obj in runtime_data["objects"]:
        state = _resolve_dynamic_object_state(obj, time_value, delta, origin, shape)
        if not state["active"]:
            continue
        ix0, ix1, iy0, iy1, iz0, iz1 = state["index_bounds"]

        _sample_obstacle_data_backwards(
            out_mask,
            out_velocity_x,
            out_velocity_y,
            out_velocity_z,
            obj["local_mask"],
            delta,
            origin[0], origin[1], origin[2],
            ix0, ix1, iy0, iy1, iz0, iz1,
            obj["local_origin"],
            state["inv"],
            state["matrix_rate"],
        )

    return out_mask, out_velocity_x, out_velocity_y, out_velocity_z


def update_dynamic_mask_gpu(runtime_data, time_value, out_mask):
    """Update one combined world-space mask directly on the GPU."""
    if out_mask is None:
        raise ValueError("A device output mask is required for GPU dynamic updates.")

    prepare_dynamic_runtime_for_gpu(runtime_data)

    shape = runtime_data["shape"]
    origin = np.asarray(runtime_data["origin"], dtype=np.float32)
    delta = np.float32(runtime_data["delta"])
    full_volume = int(shape[0] * shape[1] * shape[2])
    dirty_regions = []
    dirty_volume = 0
    active_objects = []

    for object_index, obj in enumerate(runtime_data["objects"]):
        state = _resolve_dynamic_object_state(obj, time_value, delta, origin, shape)
        previous_bounds = obj.get("last_gpu_index_bounds")
        current_bounds = state["index_bounds"] if state["active"] else None
        dirty_bounds = _merge_index_bounds(previous_bounds, current_bounds)
        if dirty_bounds is not None:
            dirty_regions.append(dirty_bounds)
            sx, sy, sz = _region_shape(dirty_bounds)
            dirty_volume += sx * sy * sz
        if state["active"]:
            active_objects.append((object_index, obj, state))

    if not runtime_data.get("gpu_mask_initialized", False) or dirty_volume >= full_volume:
        _clear_mask_cuda[volume_blocks_per_grid(shape, THREADS_PER_BLOCK_3D), THREADS_PER_BLOCK_3D](out_mask)
    else:
        for dirty_bounds in dirty_regions:
            ix0, ix1, iy0, iy1, iz0, iz1 = dirty_bounds
            sx, sy, sz = _region_shape(dirty_bounds)
            _clear_mask_box_cuda[
                volume_blocks_per_grid((sx, sy, sz), THREADS_PER_BLOCK_3D),
                THREADS_PER_BLOCK_3D,
            ](out_mask, int(ix0), int(iy0), int(iz0), sx, sy, sz)

    for object_index, obj, state in active_objects:
        ix0, ix1, iy0, iy1, iz0, iz1 = state["index_bounds"]
        sx, sy, sz = _region_shape(state["index_bounds"])
        inv = state["inv"]
        _sample_mask_backwards_object_cuda[
            volume_blocks_per_grid((sx, sy, sz), THREADS_PER_BLOCK_3D),
            THREADS_PER_BLOCK_3D,
        ](
            out_mask,
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
            delta,
            np.float32(origin[0]), np.float32(origin[1]), np.float32(origin[2]),
        )
        obj["last_gpu_index_bounds"] = state["index_bounds"]

    for obj in runtime_data["objects"]:
        if obj.get("dynamic_state") is None or not obj["dynamic_state"].get("active", False):
            obj["last_gpu_index_bounds"] = None

    runtime_data["gpu_mask_initialized"] = True
    has_obstacle = len(active_objects) > 0

    runtime_data["last_has_obstacle"] = bool(has_obstacle)
    return out_mask


def update_dynamic_obstacle_data_gpu(runtime_data, time_value, out_mask, out_velocity_x, out_velocity_y, out_velocity_z):
    """Update one moving obstacle mask and its wall velocity fields directly on the GPU."""
    if out_mask is None or out_velocity_x is None or out_velocity_y is None or out_velocity_z is None:
        raise ValueError("Device obstacle mask and velocity fields are required for GPU dynamic updates.")

    prepare_dynamic_runtime_for_gpu(runtime_data)

    shape = runtime_data["shape"]
    origin = np.asarray(runtime_data["origin"], dtype=np.float32)
    delta = np.float32(runtime_data["delta"])
    full_volume = int(shape[0] * shape[1] * shape[2])
    dirty_regions = []
    dirty_volume = 0
    active_objects = []

    for object_index, obj in enumerate(runtime_data["objects"]):
        state = _resolve_dynamic_object_state(obj, time_value, delta, origin, shape)
        previous_bounds = obj.get("last_gpu_index_bounds")
        current_bounds = state["index_bounds"] if state["active"] else None
        dirty_bounds = _merge_index_bounds(previous_bounds, current_bounds)
        if dirty_bounds is not None:
            dirty_regions.append(dirty_bounds)
            sx, sy, sz = _region_shape(dirty_bounds)
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
            ix0, ix1, iy0, iy1, iz0, iz1 = dirty_bounds
            sx, sy, sz = _region_shape(dirty_bounds)
            _clear_obstacle_fields_box_cuda[
                volume_blocks_per_grid((sx, sy, sz), THREADS_PER_BLOCK_3D),
                THREADS_PER_BLOCK_3D,
            ](out_mask, out_velocity_x, out_velocity_y, out_velocity_z, int(ix0), int(iy0), int(iz0), sx, sy, sz)

    for object_index, obj, state in active_objects:
        ix0, ix1, iy0, iy1, iz0, iz1 = state["index_bounds"]
        sx, sy, sz = _region_shape(state["index_bounds"])
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
