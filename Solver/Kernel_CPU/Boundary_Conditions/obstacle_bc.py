from numba import njit, prange


@njit(cache=True, parallel=True)
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
    Apply obstacle boundary conditions inside the obstacle mask.
    """
    nx, ny, nz = mask.shape

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k]:
                    u[i, j, k] = obstacle_velocity_x[i, j, k]
                    v[i, j, k] = obstacle_velocity_y[i, j, k]
                    w[i, j, k] = obstacle_velocity_z[i, j, k]
                    smoke[i, j, k] = 0.0
                    fuel[i, j, k] = 0.0
                    flame[i, j, k] = 0.0
