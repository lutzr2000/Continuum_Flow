import numpy as np
from numba import cuda

import Obstacles as Obstacles

THREADS_PER_BLOCK_3D = (8, 8, 8)


def _combine_exported_obstacles(obstacle_entries):
    """Merge exported obstacle nodes into one kernel obstacle configuration."""
    mesh_objects = []
    for obstacle_entry in obstacle_entries:
        if obstacle_entry.get("shape") != "mesh":
            continue
        mesh_cfg = obstacle_entry.get("mesh", {})
        mesh_objects.extend(mesh_cfg.get("objects", ()))

    if mesh_objects:
        return {
            "shape": "mesh",
            "solid": True,
            "mesh": {
                "objects": mesh_objects,
            },
        }

    return {
        "shape": "empty",
        "solid": False,
        "mesh": {
            "objects": [],
        },
    }


def build_obstacle_data(domain_cfg, obstacle_entries):
    """
    build the voxel obstacle mask from exported obstacle nodes.

    Args:
        domain_cfg (dict): exported domain configuration
        obstacle_entries (list[dict]): exported obstacle node configurations
    Returns:
        dict: obstacle configuration and boolean obstacle mask
    """
    obstacle_cfg = _combine_exported_obstacles(obstacle_entries)
    nx = int(domain_cfg["grid"]["nx"])
    ny = int(domain_cfg["grid"]["ny"])
    nz = int(domain_cfg["grid"]["nz"])
    delta = float(domain_cfg["resolution"])
    origin_x = -0.5 * nx * delta
    origin_y = -0.5 * ny * delta
    origin_z = 0.0

    if obstacle_cfg["shape"] == "mesh":
        mesh_cfg = obstacle_cfg.get("mesh", {})
        mesh_objects = mesh_cfg.get("objects", mesh_cfg if isinstance(mesh_cfg, list) else [])
        obstacle_mask = Obstacles.mesh(
            nx,
            ny,
            nz,
            delta,
            mesh_objects,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_z=origin_z,
        )
        return {
            "config": obstacle_cfg,
            "mask": obstacle_mask,
        }

    if obstacle_cfg["shape"] == "empty":
        return {
            "config": obstacle_cfg,
            "mask": np.zeros(
                (nx, ny, nz),
                dtype=np.bool_,
            ),
        }

    raise ValueError(f"Unsupported obstacle shape '{obstacle_cfg['shape']}'")


@cuda.jit
def _obstacle_bc_kernel(u, v, w, p, smoke, fuel, flame, mask):
    """
    applies all obstacle zeroing conditions inside a 3D obstacle mask on the GPU.

    Each thread checks one obstacle cell and clears velocity, pressure and
    scalar values when the mask marks that cell as solid.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        smoke (device array): smoke field
        fuel (device array): fuel field
        flame (device array): flame indicator field
        mask (device array): boolean obstacle mask
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = mask.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if mask[i, j, k]:
        u[i, j, k] = 0.0
        v[i, j, k] = 0.0
        w[i, j, k] = 0.0
        p[i, j, k] = 0.0
        smoke[i, j, k] = 0.0
        fuel[i, j, k] = 0.0
        flame[i, j, k] = 0.0


def obstacle_bc(u, v, w, p, T, smoke, fuel, flame, obstacle_mask, threadsperblock=None):
    """
    applies all obstacle boundary conditions to the GPU field state.

    Velocity, pressure and scalar fields are cleared inside obstacle cells so
    obstacle regions stay empty and act as solid regions.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        T (device array): temperature field
        smoke (device array): smoke field
        fuel (device array): fuel field
        flame (device array): flame indicator field
        obstacle_mask (device array): boolean obstacle mask
        threadsperblock (tuple[int, int, int], optional): CUDA block shape for volume kernels
    Returns:
        tuple: updated velocity, pressure and scalar fields
    """
    if threadsperblock is None:
        threadsperblock = THREADS_PER_BLOCK_3D

    blockspergrid = (
        (obstacle_mask.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (obstacle_mask.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
        (obstacle_mask.shape[2] + threadsperblock[2] - 1) // threadsperblock[2],
    )

    _obstacle_bc_kernel[blockspergrid, threadsperblock](u, v, w, p, smoke, fuel, flame, obstacle_mask)

    return u, v, w, p, T, smoke, fuel, flame
