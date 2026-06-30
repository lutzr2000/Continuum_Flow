import numpy as np
import trimesh
from pathlib import Path
from numba import njit, prange
import Solver.General.update_masks as update_masks


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
def _scanline_pass(iy0, iy1, iz0, iz1, z_span, line_counts, candidates, write):
    """Count or write triangle ids for all touched cropped YZ scanlines."""
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
def sample_mask_backwards(
    out,
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
):
    """Back-sample a local reference mask into the world grid and evaluate wall velocity."""
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


_CONFIG_DIR = None


def set_config_dir(config_dir):
    global _CONFIG_DIR
    _CONFIG_DIR = Path(config_dir)


def _resolve_mesh_file(mesh_file):
    path = Path(mesh_file)

    if path.is_absolute():
        return path

    if _CONFIG_DIR is not None:
        return _CONFIG_DIR / path

    return path


def _eps(delta):
    """Return the geometric tolerance used by voxelization helpers for one cell size."""
    return np.float32(np.float32(delta) * 1.0e-5 + 1.0e-7)


def _triangle_bounds(triangles):
    """Compute axis-aligned bounds for a triangle array shaped like (n, 3, 3)."""
    vertices = triangles.reshape(-1, 3)
    return vertices.min(axis=0).astype(np.float32), vertices.max(axis=0).astype(np.float32)


def _local_bounds_to_indices(bounds_min, bounds_max, delta):
    """Convert local bounds into inclusive voxel index bounds without world-grid clipping."""
    bounds_min = np.asarray(bounds_min, dtype=np.float32)
    bounds_max = np.asarray(bounds_max, dtype=np.float32)
    delta = np.float32(delta)
    lo = np.floor(bounds_min / delta).astype(np.int32)
    hi = np.ceil(bounds_max / delta).astype(np.int32)
    return int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]), int(lo[2]), int(hi[2])


def _load_mesh_triangles(mesh_object):
    mesh_file = mesh_object.get("mesh_file")
    if not mesh_file:
        return np.empty((0, 3, 3), dtype=np.float32)

    path = _resolve_mesh_file(mesh_file)
    mesh = trimesh.load_mesh(str(path), process=False)

    if mesh.is_empty or len(mesh.faces) == 0:
        return np.empty((0, 3, 3), dtype=np.float32)

    triangles = mesh.vertices[mesh.faces]
    return update_masks._as_f32(np.asarray(triangles, dtype=np.float32))


def _scanline_candidates(triangles, delta, origin_y, origin_z, iy_min, iy_max, iz_min, iz_max):
    """Build compact candidate triangle lists for all cropped YZ scanlines."""
    line_count = int((iy_max - iy_min + 1) * (iz_max - iz_min + 1))
    empty_offsets = np.zeros(max(line_count, 0) + 1, dtype=np.int32)
    if triangles.size == 0 or line_count <= 0:
        return (
            empty_offsets,
            np.empty(0, dtype=np.int32),
            np.empty((0, 3, 3), dtype=np.float32),
        )

    eps = _eps(delta)
    yz = triangles[:, :, 1:3]
    den = (yz[:, 1, 1] - yz[:, 2, 1]) * (yz[:, 0, 0] - yz[:, 2, 0]) + (
        yz[:, 2, 0] - yz[:, 1, 0]
    ) * (yz[:, 0, 1] - yz[:, 2, 1])
    triangles = update_masks._as_f32(triangles[np.abs(den) > eps])
    if triangles.size == 0:
        return empty_offsets, np.empty(0, dtype=np.int32), triangles

    yz_min = triangles[:, :, 1:3].min(axis=1)
    yz_max = triangles[:, :, 1:3].max(axis=1)
    origin = np.asarray((origin_y, origin_z), dtype=np.float32)
    local_min = np.floor((yz_min - origin - eps) / delta).astype(np.int32) - np.asarray(
        (iy_min, iz_min), dtype=np.int32
    )
    local_max = np.ceil((yz_max - origin + eps) / delta).astype(np.int32) - np.asarray(
        (iy_min, iz_min), dtype=np.int32
    )

    limit = np.asarray((iy_max - iy_min, iz_max - iz_min), dtype=np.int32)
    local_min = np.minimum(np.maximum(local_min, 0), limit)
    local_max = np.minimum(np.maximum(local_max, 0), limit)

    z_span = int(limit[1] + 1)
    counts = np.zeros(line_count, dtype=np.int32)
    dummy = np.empty(0, dtype=np.int32)
    _scanline_pass(
        local_min[:, 0],
        local_max[:, 0],
        local_min[:, 1],
        local_max[:, 1],
        z_span,
        counts,
        dummy,
        counts,
    )

    offsets = np.empty(line_count + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, dtype=np.int32, out=offsets[1:])

    candidates = np.empty(int(offsets[-1]), dtype=np.int32)
    write = offsets[:-1].copy()
    _scanline_pass(
        local_min[:, 0],
        local_max[:, 0],
        local_min[:, 1],
        local_max[:, 1],
        z_span,
        counts,
        candidates,
        write,
    )
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

    cropped_origin = update_masks._as_f32(
        origin + delta * np.asarray((ix0, iy0, iz0), dtype=np.float32)
    )
    _fill_scanlines(
        mask,
        valid_triangles,
        delta,
        cropped_origin[0],
        cropped_origin[1],
        cropped_origin[2],
        offsets,
        candidates,
    )

    return {
        "mask": np.ascontiguousarray(mask),
        "origin": cropped_origin,
        "bounds_min": cropped_origin,
        "bounds_max": update_masks._as_f32(
            origin + delta * np.asarray((ix1, iy1, iz1), dtype=np.float32)
        ),
    }


def _voxelize_triangles(triangles, delta):
    """Voxelize one triangle array with the cropped numba scanline voxelizer."""
    if triangles.size == 0:
        return None

    bounds_min, bounds_max = _triangle_bounds(triangles)
    index_bounds = _local_bounds_to_indices(bounds_min, bounds_max, np.float32(delta))
    return _voxelize_cropped(triangles, np.float32(delta), (0.0, 0.0, 0.0), index_bounds)


def voxelise_base_mesh(mesh_object, delta):
    """Build one object-local voxel mask that stays fixed in local coordinates."""
    if not mesh_object:
        return None
    return _voxelize_triangles(_load_mesh_triangles(mesh_object), np.float32(delta))


def sample_base_mask_to_world(
    nx,
    ny,
    nz,
    delta,
    mesh_object,
    base_voxels,
    origin_x=0.0,
    origin_y=0.0,
    origin_z=0.0,
):
    """Sample one object-local voxel mask into the simulation world grid."""
    out = np.zeros((nx, ny, nz), dtype=np.bool_)
    if not mesh_object or base_voxels is None:
        return out

    origin = np.asarray((origin_x, origin_y, origin_z), dtype=np.float32)
    delta = np.float32(delta)

    object_matrix = update_masks._initial_world_matrix(mesh_object)
    bounds_center, bounds_extent = update_masks._bounds_center_extent(
        base_voxels["bounds_min"], base_voxels["bounds_max"]
    )
    bounds_min, bounds_max = update_masks._transform_bounds(
        bounds_center,
        bounds_extent,
        object_matrix,
    )
    object_matrix_inv = update_masks._invert_affine_matrix(object_matrix)

    ix0, ix1, iy0, iy1, iz0, iz1 = update_masks._bounds_to_indices(
        bounds_min,
        bounds_max,
        delta,
        origin,
        shape=out.shape,
    )
    if ix0 > ix1 or iy0 > iy1 or iz0 > iz1:
        return out

    sample_mask_backwards(
        out,
        base_voxels["mask"],
        delta,
        origin[0],
        origin[1],
        origin[2],
        ix0,
        ix1,
        iy0,
        iy1,
        iz0,
        iz1,
        base_voxels["origin"],
        object_matrix_inv,
    )

    return out


def voxelise_mesh(
    nx, ny, nz, delta, mesh_object, origin_x=0.0, origin_y=0.0, origin_z=0.0
):
    """Voxelise one exported triangle mesh into a boolean obstacle mask."""
    base_voxels = voxelise_base_mesh(mesh_object, delta)
    return sample_base_mask_to_world(
        nx,
        ny,
        nz,
        delta,
        mesh_object,
        base_voxels,
        origin_x=origin_x,
        origin_y=origin_y,
        origin_z=origin_z,
    )


def voxelise_mesh_all(
    nx, ny, nz, delta, mesh_objects, origin_x=0.0, origin_y=0.0, origin_z=0.0
):
    """Build combined local voxel data and the corresponding world obstacle mask."""
    out = np.zeros((nx, ny, nz), dtype=np.bool_)
    base_voxels = []
    if not mesh_objects:
        return base_voxels, out

    for mesh_object in mesh_objects:
        object_base_voxels = voxelise_base_mesh(mesh_object, delta)
        if object_base_voxels is not None:
            base_voxels.append(
                {
                    "mesh_object": mesh_object,
                    "voxels": object_base_voxels,
                }
            )
        out |= sample_base_mask_to_world(
            nx,
            ny,
            nz,
            delta,
            mesh_object,
            object_base_voxels,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_z=origin_z,
        )

    return base_voxels, out
