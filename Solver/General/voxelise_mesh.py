import numpy as np
import trimesh
import Solver.General.update_masks as update_masks


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
    return update_masks._as_f32(triangles.reshape(tuple(map(int, shape))))


def _voxelize_triangles(triangles, delta):
    """Voxelize one triangle array with trimesh and return dense local mask data."""
    if triangles.size == 0:
        return None

    vertices = update_masks._as_f32(triangles.reshape(-1, 3))
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

    origin = update_masks._as_f32(np.asarray(voxel_grid.translation, dtype=np.float32))
    bounds_max = update_masks._as_f32(
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


def voxelise_mesh(
    nx, ny, nz, delta, mesh_object, origin_x=0.0, origin_y=0.0, origin_z=0.0
):
    """Voxelise one exported triangle mesh into a boolean obstacle mask."""
    out = np.zeros((nx, ny, nz), dtype=np.bool_)
    if not mesh_object:
        return out

    origin = np.asarray((origin_x, origin_y, origin_z), dtype=np.float32)
    delta = np.float32(delta)

    voxels = _voxelize_triangles(_load_mesh_triangles(mesh_object), delta)
    if voxels is None:
        return out

    object_matrix = update_masks._initial_world_matrix(mesh_object)
    bounds_center, bounds_extent = update_masks._bounds_center_extent(
        voxels["bounds_min"], voxels["bounds_max"]
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

    update_masks.sample_mask_backwards(
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


def voxelise_mesh_all(
    nx, ny, nz, delta, mesh_objects, origin_x=0.0, origin_y=0.0, origin_z=0.0
):
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
