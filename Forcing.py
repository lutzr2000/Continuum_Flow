import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True, fastmath=True)
def forcing_turbulence(Fx, Fy, Fz, t, frequency, scale, amplitude, delta):
    """
    Computes a procedural turbulence forcing field.
    """
    nx, ny, nz = Fx.shape
    safe_scale = max(scale, delta)
    wave_number = 2.0 * np.pi / safe_scale
    omega = 2.0 * np.pi * max(frequency, 0.0)
    phase = omega * t

    Fx.fill(0.0)
    Fy.fill(0.0)
    Fz.fill(0.0)

    for i in prange(1, nx - 1):
        x = i * delta
        for j in range(1, ny - 1):
            y = j * delta
            for k in range(1, nz - 1):
                z = k * delta

                mode_x_1 = np.sin(wave_number * (0.91 * x + 0.63 * y + 0.37 * z) + 0.80 * phase)
                mode_y_1 = np.sin(wave_number * (-0.44 * x + 1.08 * y + 0.58 * z) + 1.30 * phase)
                mode_z_1 = np.sin(wave_number * (0.56 * x - 0.87 * y + 1.22 * z) + 1.50 * phase)

                Fx[i, j, k] = amplitude * mode_x_1
                Fy[i, j, k] = amplitude * mode_y_1
                Fz[i, j, k] = amplitude * mode_z_1

    return Fx, Fy, Fz
