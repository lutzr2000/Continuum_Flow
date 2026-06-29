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
    applies all obstacle zeroing conditions inside a 3D obstacle mask on the CPU.

    Each iteration checks one obstacle cell and clears velocity and the supported
    scalar values when the mask marks that cell as solid.
    """
    nx, ny, nz = mask.shape
    total = nx * ny * nz

    for n in prange(total):
        i = n // (ny * nz)
        rem = n - i * ny * nz
        j = rem // nz
        k = rem - j * nz

        if mask[i, j, k]:
            u[i, j, k] = obstacle_velocity_x[i, j, k]
            v[i, j, k] = obstacle_velocity_y[i, j, k]
            w[i, j, k] = obstacle_velocity_z[i, j, k]

            smoke[i, j, k] = 0.0
            fuel[i, j, k] = 0.0
            flame[i, j, k] = 0.0
