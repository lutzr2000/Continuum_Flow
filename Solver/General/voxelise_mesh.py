import numpy as np
import trimesh
from numba import njit, prange
from update_masks import _sample_mask_backwards


def _as_f32(a):
    """Return a contiguous float32 view/copy of the given array-like input."""
    return np.ascontiguousarray(a, dtype=np.float32)


@njit(cache=True, parallel=True)
def _sample_mask_backwards(
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


def _voxelize_triangles(triangles, delta):
    """Voxelize one triangle array with trimesh and return dense local mask data."""
    if triangles.size == 0:
        return None

    vertices = _as_f32(triangles.reshape(-1, 3))
    faces = np.arange(vertices.shape[0], dtype=np.int64).reshape((-1, 3))
    triangle_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
        validate=False,
    )
    voxel_grid = triangle_mesh.voxelized(pitch=float(delta)).fill()
    mask = np.ascontiguousarray(np.asarray(voxel_grid.matrix, dtype=np.bool_))
    if mask.size == 0 or not np.any(mask):
        return None

    origin = _as_f32(np.asarray(voxel_grid.translation, dtype=np.float32))
    bounds_max = _as_f32(
        origin
        + np.float32(delta)
        * (np.asarray(mask.shape, dtype=np.float32) - np.float32(1.0))
    )
    return {
        "mask": mask,
        "origin": origin,
        "bounds_min": origin,
        "bounds_max": bounds_max,
    }


def voxelise_mesh(nx, ny, nz, delta, mesh_object, origin_x=0.0, origin_y=0.0, origin_z=0.0):
    """Voxelise one exported triangle mesh into a boolean obstacle mask."""
    out = np.zeros((nx, ny, nz), dtype=np.bool_)
    if not mesh_object:
        return out

    origin = np.asarray((origin_x, origin_y, origin_z), dtype=np.float32)
    delta = np.float32(delta)
    identity_inv = np.eye(4, dtype=np.float32)

    voxels = _voxelize_triangles(_load_mesh_triangles(mesh_object), delta)
    if voxels is None:
        return out

    ix0, ix1, iy0, iy1, iz0, iz1 = _bounds_to_indices(
        voxels["bounds_min"],
        voxels["bounds_max"],
        delta,
        origin,
        shape=out.shape,
    )
    if ix0 > ix1 or iy0 > iy1 or iz0 > iz1:
        return out

    _sample_mask_backwards(
        out,
        voxels["mask"],
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
        voxels["origin"],
        identity_inv,
    )

    return out


def voxelise_mesh_all(nx, ny, nz, delta, mesh_objects, origin_x=0.0, origin_y=0.0, origin_z=0.0):
    """Voxelise multiple exported triangle meshes into one boolean obstacle mask."""
    out = np.zeros((nx, ny, nz), dtype=np.bool_)
    if not mesh_objects:
        return out

    for mesh_object in mesh_objects:
        out |= voxelise_mesh(
            nx,
            ny,
            nz,
            delta,
            mesh_object,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_z=origin_z,
        )

    return out

