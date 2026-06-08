import numpy as np
from numba import njit, prange

import Solver.General.obstacles as general_obstacles


def build_obstacle_data(domain_cfg, obstacle_entries):
    """
    Build the voxel obstacle mask from exported obstacle nodes for the CPU solver.

    The heavy mesh voxelization and dynamic obstacle sampling already exist as
    host-side helpers, so the CPU kernel reuses them directly instead of
    duplicating that geometry pipeline.
    """
    return general_obstacles.build_obstacle_data(
        domain_cfg, obstacle_entries, general_obstacles
    )


def update_obstacle_mask(obstacle_data, time_value):
    """
    Update the combined obstacle mask and wall velocities for the current simulation time.
    """
    runtime = obstacle_data.get("runtime")
    if runtime is None:
        return obstacle_data["mask"]

    updated_mask, updated_velocity_x, updated_velocity_y, updated_velocity_z = (
        general_obstacles.update_dynamic_obstacle_data(
            runtime,
            time_value,
            out_mask=obstacle_data["mask"],
            out_velocity_x=obstacle_data["velocity_x"],
            out_velocity_y=obstacle_data["velocity_y"],
            out_velocity_z=obstacle_data["velocity_z"],
        )
    )
    obstacle_data["mask"] = updated_mask
    obstacle_data["velocity_x"] = updated_velocity_x
    obstacle_data["velocity_y"] = updated_velocity_y
    obstacle_data["velocity_z"] = updated_velocity_z
    return updated_mask


@njit(cache=True, parallel=True)
def _obstacle_bc_kernel_cpu(
    u,
    v,
    w,
    smoke,
    fuel,
    flame,
    mask,
    obstacle_velocity_x,
    obstacle_velocity_y,
    obstacle_velocity_z,
):
    """
    Apply the obstacle overwrite rules over the whole volume on the CPU.

    Each worker scans a disjoint chunk of the flattened arrays and overwrites
    obstacle cells with obstacle wall velocities while clearing supported
    scalar fields.
    """
    total_size = mask.size
    u_flat = u.reshape(total_size)
    v_flat = v.reshape(total_size)
    w_flat = w.reshape(total_size)
    smoke_flat = smoke.reshape(total_size)
    fuel_flat = fuel.reshape(total_size)
    flame_flat = flame.reshape(total_size)
    mask_flat = mask.reshape(total_size)
    velocity_x_flat = obstacle_velocity_x.reshape(total_size)
    velocity_y_flat = obstacle_velocity_y.reshape(total_size)
    velocity_z_flat = obstacle_velocity_z.reshape(total_size)

    for idx in prange(total_size):
        if not mask_flat[idx]:
            continue

        u_flat[idx] = velocity_x_flat[idx]
        v_flat[idx] = velocity_y_flat[idx]
        w_flat[idx] = velocity_z_flat[idx]
        smoke_flat[idx] = 0.0
        fuel_flat[idx] = 0.0
        flame_flat[idx] = 0.0


@njit(cache=True, parallel=True)
def _obstacle_bc_zero_velocity_kernel_cpu(u, v, w, smoke, fuel, flame, mask):
    """
    Apply the obstacle overwrite rules when all obstacle wall velocities are zero.
    """
    total_size = mask.size
    u_flat = u.reshape(total_size)
    v_flat = v.reshape(total_size)
    w_flat = w.reshape(total_size)
    smoke_flat = smoke.reshape(total_size)
    fuel_flat = fuel.reshape(total_size)
    flame_flat = flame.reshape(total_size)
    mask_flat = mask.reshape(total_size)

    for idx in prange(total_size):
        if not mask_flat[idx]:
            continue

        u_flat[idx] = 0.0
        v_flat[idx] = 0.0
        w_flat[idx] = 0.0
        smoke_flat[idx] = 0.0
        fuel_flat[idx] = 0.0
        flame_flat[idx] = 0.0


def obstacle_bc(
    u,
    v,
    w,
    smoke,
    fuel,
    flame,
    obstacle_mask,
    obstacle_velocity_x,
    obstacle_velocity_y,
    obstacle_velocity_z,
):
    """
    Apply all obstacle boundary conditions to the CPU field state.

    Velocity and supported scalar fields are overwritten inside obstacle cells
    so obstacle regions stay empty and act as solid regions.
    """
    if obstacle_mask.size == 0 or not np.any(obstacle_mask):
        return u, v, w, smoke, fuel, flame

    if (
        obstacle_velocity_x is None
        or obstacle_velocity_y is None
        or obstacle_velocity_z is None
    ):
        _obstacle_bc_zero_velocity_kernel_cpu(
            u,
            v,
            w,
            smoke,
            fuel,
            flame,
            obstacle_mask,
        )
    else:
        _obstacle_bc_kernel_cpu(
            u,
            v,
            w,
            smoke,
            fuel,
            flame,
            obstacle_mask,
            obstacle_velocity_x,
            obstacle_velocity_y,
            obstacle_velocity_z,
        )

    return u, v, w, smoke, fuel, flame
