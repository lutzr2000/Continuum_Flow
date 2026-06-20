import numpy as np
import trimesh
from Solver.General.update_masks import _sample_mask_backwards


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


def _bounds_center_extent(bounds_min, bounds_max):
    """Convert min/max bounds into center/extent form for cheap affine transforms."""
    bounds_min = np.asarray(bounds_min, dtype=np.float32)
    bounds_max = np.asarray(bounds_max, dtype=np.float32)
    center = (bounds_min + bounds_max) * np.float32(0.5)
    extent = (bounds_max - bounds_min) * np.float32(0.5)
    return _as_f32(center), _as_f32(extent)


def _transform_bounds(bounds_center, bounds_extent, matrix):
    """Transform one local AABB into world space using center/extent form."""
    matrix = np.asarray(matrix, dtype=np.float32)
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    center_world = _as_f32(linear @ bounds_center + translation)
    extent_world = _as_f32(np.abs(linear) @ bounds_extent)
    return center_world - extent_world, center_world + extent_world


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

    voxels = _voxelize_triangles(_load_mesh_triangles(mesh_object), delta)
    if voxels is None:
        return out

    object_matrix = _initial_world_matrix(mesh_object)
    bounds_center, bounds_extent = _bounds_center_extent(
        voxels["bounds_min"], voxels["bounds_max"]
    )
    bounds_min, bounds_max = _transform_bounds(
        bounds_center,
        bounds_extent,
        object_matrix,
    )
    object_matrix_inv = _invert_affine_matrix(object_matrix)

    ix0, ix1, iy0, iy1, iz0, iz1 = _bounds_to_indices(
        bounds_min,
        bounds_max,
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
        object_matrix_inv,
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
