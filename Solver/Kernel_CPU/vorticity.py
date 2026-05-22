import math

from numba import njit, prange


@njit(cache=True, parallel=True, fastmath=True)
def compute_vorticity(
    u, v, w, obstacle_mask, omega_x, omega_y, omega_z, omega_magnitude, delta
):
    """
    Compute vorticity components and scalar magnitude from the velocity field.
    """
    nx, ny, nz = omega_magnitude.shape
    half_inv_delta = 0.5 / delta

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if (
                    i < 1
                    or j < 1
                    or k < 1
                    or i >= nx - 1
                    or j >= ny - 1
                    or k >= nz - 1
                    or obstacle_mask[i, j, k]
                ):
                    omega_x[i, j, k] = 0.0
                    omega_y[i, j, k] = 0.0
                    omega_z[i, j, k] = 0.0
                    omega_magnitude[i, j, k] = 0.0
                    continue

                du_dy = (u[i, j + 1, k] - u[i, j - 1, k]) * half_inv_delta
                du_dz = (u[i, j, k + 1] - u[i, j, k - 1]) * half_inv_delta
                dv_dx = (v[i + 1, j, k] - v[i - 1, j, k]) * half_inv_delta
                dv_dz = (v[i, j, k + 1] - v[i, j, k - 1]) * half_inv_delta
                dw_dx = (w[i + 1, j, k] - w[i - 1, j, k]) * half_inv_delta
                dw_dy = (w[i, j + 1, k] - w[i, j - 1, k]) * half_inv_delta

                wx = dw_dy - dv_dz
                wy = du_dz - dw_dx
                wz = dv_dx - du_dy

                omega_x[i, j, k] = wx
                omega_y[i, j, k] = wy
                omega_z[i, j, k] = wz
                omega_magnitude[i, j, k] = math.sqrt(wx * wx + wy * wy + wz * wz)


@njit(cache=True, parallel=True, fastmath=True)
def apply_vorticity_confinement(
    obstacle_mask,
    omega_x,
    omega_y,
    omega_z,
    omega_magnitude,
    Fx,
    Fy,
    Fz,
    delta,
    vorticity_strength,
):
    """
    Accumulate vorticity confinement forces into the body-force fields on the CPU.
    """
    nx, ny, nz = omega_magnitude.shape
    half_inv_delta = 0.5 / delta

    for i in prange(2, nx - 2):
        for j in range(2, ny - 2):
            for k in range(2, nz - 2):
                if obstacle_mask[i, j, k]:
                    continue

                grad_x = (
                    omega_magnitude[i + 1, j, k] - omega_magnitude[i - 1, j, k]
                ) * half_inv_delta
                grad_y = (
                    omega_magnitude[i, j + 1, k] - omega_magnitude[i, j - 1, k]
                ) * half_inv_delta
                grad_z = (
                    omega_magnitude[i, j, k + 1] - omega_magnitude[i, j, k - 1]
                ) * half_inv_delta

                grad_length = math.sqrt(
                    grad_x * grad_x + grad_y * grad_y + grad_z * grad_z
                )
                if grad_length <= 1.0e-12:
                    continue

                nx_dir = grad_x / grad_length
                ny_dir = grad_y / grad_length
                nz_dir = grad_z / grad_length

                wx = omega_x[i, j, k]
                wy = omega_y[i, j, k]
                wz = omega_z[i, j, k]

                Fx[i, j, k] += vorticity_strength * (ny_dir * wz - nz_dir * wy)
                Fy[i, j, k] += vorticity_strength * (nz_dir * wx - nx_dir * wz)
                Fz[i, j, k] += vorticity_strength * (nx_dir * wy - ny_dir * wx)
