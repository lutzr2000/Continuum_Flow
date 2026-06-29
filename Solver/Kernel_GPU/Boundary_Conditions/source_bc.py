import math
import numpy as np
from numba import cuda

@cuda.jit(cache=True)
def source_bc_kernel(
    u,
    v,
    w,
    T,
    smoke,
    fuel,
    source_mask,
    temperature_value,
    smoke_value,
    fuel_value,
    velocity_x_value,
    velocity_y_value,
    velocity_z_value,
    dt,
):
    """
    Apply source velocity/temperature and inject smoke/fuel rates on the GPU.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = source_mask.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
        return

    if not source_mask[i, j, k]:
        return

    if velocity_x_value != 0 and velocity_y_value != 0 and velocity_z_value != 0:
        u[i, j, k] = velocity_x_value
        v[i, j, k] = velocity_y_value
        w[i, j, k] = velocity_z_value

    T[i, j, k] = temperature_value

    if smoke_value != 0:
        smoke[i, j, k] = min(
            max(smoke[i, j, k] + dt * 10.0 * smoke_value, 0.0),
            100.0,
        )
    if fuel_value != 0:
        fuel[i, j, k] = min(
            max(fuel[i, j, k] + dt * 10.0 * fuel_value, 0.0),
            100.0,
        )
