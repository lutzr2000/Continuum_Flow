from numba import njit, prange


@njit(cache=True, inline="always")
def _clamp(value, lower, upper):
    return min(max(value, lower), upper)


@njit(cache=True, inline="always")
def _noise_multiplier(noise_value, noise_amplitude):
    return _clamp(1.0 + noise_amplitude * noise_value, 0.0, 2.0)


@njit(cache=True, parallel=True)
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
    Apply source values and inject smoke and fuel rates.
    """
    nx, ny, nz = source_mask.shape

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
                    continue
                if not source_mask[i, j, k]:
                    continue

                scalar_multiplier = _noise_multiplier(source_noise[i, j, k], noise_amplitude)

                if velocity_x_value != 0 and velocity_y_value != 0 and velocity_z_value != 0:
                    u[i, j, k] = velocity_x_value
                    v[i, j, k] = velocity_y_value
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
