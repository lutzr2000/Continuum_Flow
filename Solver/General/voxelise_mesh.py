import numpy as np
import trimesh
from pathlib import Path
from numba import njit, prange
import Solver.General.update_masks as update_masks

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
