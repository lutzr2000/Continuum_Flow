from numba import cuda

import Solver.Kernel_GPU.Boundary_Conditions.obstacles as obstacles


def update_obstacle_mask(obstacle_data, time_value):
    """
    Update the combined obstacle mask for the current simulation time.

    """
    runtime = obstacle_data.get("runtime")
    if runtime is None:
        return obstacle_data["mask"]

    updated_mask, updated_velocity_x, updated_velocity_y, updated_velocity_z = (
        obstacles.update_dynamic_obstacle_data(
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


@cuda.jit(cache=True)
def obstacle_bc_kernel(
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
    applies all obstacle zeroing conditions inside a 3D obstacle mask on the GPU.

    Each thread checks one obstacle cell and clears velocity and the supported
    scalar values when the mask marks that cell as solid.

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
