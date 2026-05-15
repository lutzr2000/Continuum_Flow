from numba import cuda

import Solver.General.obstacles as general_obstacles
import Solver.Kernel_GPU.Boundary_Conditions.obstacles as obstacles
from Solver.Kernel_GPU.kernel_config import THREADS_PER_BLOCK_3D, volume_blocks_per_grid


def build_obstacle_data(domain_cfg, obstacle_entries):
    """
    build the voxel obstacle mask from exported obstacle nodes.

    Args:
        domain_cfg (dict): exported domain configuration
        obstacle_entries (list[dict]): exported obstacle node configurations
    Returns:
        dict: obstacle configuration and boolean obstacle mask
    """
    return general_obstacles.build_obstacle_data(domain_cfg, obstacle_entries, obstacles)


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


@cuda.jit(cache=True)
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
