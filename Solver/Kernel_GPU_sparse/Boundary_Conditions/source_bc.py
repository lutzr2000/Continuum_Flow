import math
import numpy as np
from numba import cuda


@cuda.jit(device=True, inline=True, cache=True)
def _clamp(value, lower, upper):
    return min(max(value, lower), upper)


@cuda.jit(device=True, inline=True, cache=True)
def _noise_multiplier(noise_value, noise_amplitude):
    return _clamp(1.0 + noise_amplitude * noise_value, 0.0, 2.0)


@cuda.jit(cache=True)
def source_bc_kernel(
    u,
    v,
    w,
    T,
    smoke,
    fuel,
    source_mask,
    source_noise,
    temperature_value,
    smoke_value,
    fuel_value,
    velocity_x_value,
    velocity_y_value,
    velocity_z_value,
    noise_amplitude,
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

    scalar_multiplier = _noise_multiplier(source_noise[i, j, k], noise_amplitude)

    if velocity_x_value != 0:
        u[i, j, k] = velocity_x_value
    if velocity_y_value != 0:
        v[i, j, k] = velocity_y_value
    if velocity_z_value != 0:
        w[i, j, k] = velocity_z_value

    T[i, j, k] = max(temperature_value * scalar_multiplier, 0.0)

    if smoke_value != 0:
        smoke[i, j, k] = min(
            max(smoke[i, j, k] + dt * 10.0 * smoke_value * scalar_multiplier, 0.0),
            100.0,
        )
    if fuel_value != 0:
        fuel[i, j, k] = min(
            max(fuel[i, j, k] + dt * 10.0 * fuel_value * scalar_multiplier, 0.0),
            100.0,
        )
