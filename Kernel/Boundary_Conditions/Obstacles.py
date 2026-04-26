import numpy as np
from numba import njit, prange


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


def _matrix_at(series, t):
    """Linear interpolation through sampled world matrices, with a monotonic cursor."""
    times = series["times"]
    matrices = series["matrices_world"]
    if times.size <= 1 or t <= float(times[0]):
        return np.asarray(matrices[0], dtype=np.float32)

    cursor = min(int(series.get("cursor", 0)), int(times.size - 2))
    while cursor < times.size - 2 and t >= float(times[cursor + 1]):
        cursor += 1
    series["cursor"] = cursor

    if cursor >= times.size - 2 and t >= float(times[-1]):
        return np.asarray(matrices[-1], dtype=np.float32)

    t0, t1 = float(times[cursor]), float(times[cursor + 1])
    if t1 <= t0:
        return np.asarray(matrices[cursor], dtype=np.float32)

    a = np.float32((t - t0) / (t1 - t0))
    return _as_f32(matrices[cursor] * (1.0 - a) + matrices[cursor + 1] * a)


def _transform_bounds(bounds_min, bounds_max, matrix):
    """Transform eight AABB corners and return the new conservative AABB."""
    lo, hi = bounds_min, bounds_max
    corners = np.asarray(
        [[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
        dtype=np.float32,
    )
    xyz = corners @ np.asarray(matrix, dtype=np.float32).T
    return xyz[:, :3].min(axis=0).astype(np.float32), xyz[:, :3].max(axis=0).astype(np.float32)


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
    for obj in mesh_objects:
        triangles = _load_mesh_triangles(obj)
        voxels = _voxelize_local(triangles, delta)
        if voxels is None or not np.any(voxels["mask"]):
            continue

        objects.append({
            "object_name": obj.get("object_name"),
            "local_mask": voxels["mask"],
            "local_origin": voxels["origin"],
            "local_bounds_min": voxels["bounds_min"],
            "local_bounds_max": voxels["bounds_max"],
            "transform_series": _matrix_series(obj.get("transform_animation", {})),
        })

    return {
        "objects": objects,
        "shape": (int(nx), int(ny), int(nz)),
        "delta": np.float32(delta),
        "origin": np.asarray((origin_x, origin_y, origin_z), dtype=np.float32),
    }


def update_dynamic_mask(runtime_data, time_value, out_mask=None):
    """Update one combined world-space mask by back-sampling all runtime objects."""
    shape = runtime_data["shape"]
    out = np.zeros(shape, dtype=np.bool_) if out_mask is None else out_mask
    out.fill(False)

    delta = np.float32(runtime_data["delta"])
    origin = np.asarray(runtime_data["origin"], dtype=np.float32)

    for obj in runtime_data["objects"]:
        matrix = _matrix_at(obj["transform_series"], time_value)
        bounds = _transform_bounds(obj["local_bounds_min"], obj["local_bounds_max"], matrix)
        ix0, ix1, iy0, iy1, iz0, iz1 = _bounds_to_indices(bounds[0], bounds[1], delta, origin, shape=shape)
        if ix0 > ix1 or iy0 > iy1 or iz0 > iz1:
            continue

        _sample_mask_backwards(
            out,
            obj["local_mask"],
            delta,
            origin[0], origin[1], origin[2],
            ix0, ix1, iy0, iy1, iz0, iz1,
            obj["local_origin"],
            _as_f32(np.linalg.inv(matrix)),
        )

    return out
