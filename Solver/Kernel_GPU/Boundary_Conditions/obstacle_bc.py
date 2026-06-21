import math
from numba import cuda


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
