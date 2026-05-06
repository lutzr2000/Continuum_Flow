import numpy as np
from numba import cuda

import Solver.Kernel_GPU.Boundary_Conditions.obstacles as obstacles
from Solver.Kernel_GPU.kernel_config import THREADS_PER_BLOCK_3D, volume_blocks_per_grid


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
    zero_velocity_x = np.zeros((nx, ny, nz), dtype=np.float32)
    zero_velocity_y = np.zeros((nx, ny, nz), dtype=np.float32)
    zero_velocity_z = np.zeros((nx, ny, nz), dtype=np.float32)

    if obstacle_cfg["shape"] == "mesh":
        mesh_cfg = obstacle_cfg.get("mesh", {})
        mesh_objects = mesh_cfg.get("objects", mesh_cfg if isinstance(mesh_cfg, list) else [])
        obstacle_runtime = obstacles.build_dynamic_runtime(
            nx,
            ny,
            nz,
            delta,
            mesh_objects,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_z=origin_z,
        )
        obstacle_mask, obstacle_velocity_x, obstacle_velocity_y, obstacle_velocity_z = obstacles.update_dynamic_obstacle_data(
            obstacle_runtime,
            0.0,
        )
        return {
            "config": obstacle_cfg,
            "mask": obstacle_mask,
            "velocity_x": obstacle_velocity_x,
            "velocity_y": obstacle_velocity_y,
            "velocity_z": obstacle_velocity_z,
            "runtime": obstacle_runtime,
            "is_animated": bool(obstacle_runtime.get("is_animated", False)),
        }

    if obstacle_cfg["shape"] == "empty":
        return {
            "config": obstacle_cfg,
            "mask": np.zeros(
                (nx, ny, nz),
                dtype=np.bool_,
            ),
            "velocity_x": zero_velocity_x,
            "velocity_y": zero_velocity_y,
            "velocity_z": zero_velocity_z,
            "runtime": None,
            "is_animated": False,
        }

    raise ValueError(f"Unsupported obstacle shape '{obstacle_cfg['shape']}'")


def update_obstacle_mask(obstacle_data, time_value):
    """Update the combined obstacle mask for the current simulation time."""
    runtime = obstacle_data.get("runtime")
    if runtime is None:
        return obstacle_data["mask"]

    updated_mask, updated_velocity_x, updated_velocity_y, updated_velocity_z = obstacles.update_dynamic_obstacle_data(
        runtime,
        time_value,
        out_mask=obstacle_data["mask"],
        out_velocity_x=obstacle_data["velocity_x"],
        out_velocity_y=obstacle_data["velocity_y"],
        out_velocity_z=obstacle_data["velocity_z"],
    )
    obstacle_data["mask"] = updated_mask
    obstacle_data["velocity_x"] = updated_velocity_x
    obstacle_data["velocity_y"] = updated_velocity_y
    obstacle_data["velocity_z"] = updated_velocity_z
    return updated_mask


@cuda.jit
def _obstacle_bc_kernel(u, v, w, smoke, fuel, flame, mask, obstacle_velocity_x, obstacle_velocity_y, obstacle_velocity_z):
    """
    applies all obstacle zeroing conditions inside a 3D obstacle mask on the GPU.

    Each thread checks one obstacle cell and clears velocity and the supported
    scalar values when the mask marks that cell as solid.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
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
        u[i, j, k] = obstacle_velocity_x[i, j, k]
        v[i, j, k] = obstacle_velocity_y[i, j, k]
        w[i, j, k] = obstacle_velocity_z[i, j, k]
        smoke[i, j, k] = 0.0
        fuel[i, j, k] = 0.0
        flame[i, j, k] = 0.0


def obstacle_bc(
    u, v, w, smoke, fuel, flame,
    obstacle_mask, obstacle_velocity_x, obstacle_velocity_y, obstacle_velocity_z,
    threadsperblock=None,
):
    """
    applies all obstacle boundary conditions to the GPU field state.

    Velocity and supported scalar fields are cleared inside obstacle cells so
    obstacle regions stay empty and act as solid regions.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        smoke (device array): smoke field
        fuel (device array): fuel field
        flame (device array): flame indicator field
        obstacle_mask (device array): boolean obstacle mask
        threadsperblock (tuple[int, int, int], optional): CUDA block shape for volume kernels
    Returns:
        tuple: updated velocity and scalar fields
    """
    if threadsperblock is None:
        threadsperblock = THREADS_PER_BLOCK_3D

    blockspergrid = volume_blocks_per_grid(obstacle_mask.shape, threadsperblock)

    _obstacle_bc_kernel[blockspergrid, threadsperblock](
        u, v, w, smoke, fuel, flame,
        obstacle_mask, obstacle_velocity_x, obstacle_velocity_y, obstacle_velocity_z,
    )

    return u, v, w, smoke, fuel, flame
