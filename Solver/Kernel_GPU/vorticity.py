from numba import cuda
import math

@cuda.jit(cache=True)
def compute_vorticity(u, v, w, obstacle_mask, omega_x, omega_y, omega_z, omega_magnitude, delta):
    """
    Compute vorticity components and scalar magnitude from the velocity field.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = omega_magnitude.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if (
        i < 1 or j < 1 or k < 1 or
        i >= nx - 1 or j >= ny - 1 or k >= nz - 1
    ):
        omega_x[i, j, k] = 0.0
        omega_y[i, j, k] = 0.0
        omega_z[i, j, k] = 0.0
        omega_magnitude[i, j, k] = 0.0
        return

    if obstacle_mask[i, j, k]:
        omega_x[i, j, k] = 0.0
        omega_y[i, j, k] = 0.0
        omega_z[i, j, k] = 0.0
        omega_magnitude[i, j, k] = 0.0
        return

    half_inv_delta = 0.5 / delta
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


@cuda.jit(cache=True)
def apply_vorticity_confinement(
    obstacle_mask, omega_x, omega_y, omega_z, omega_magnitude, Fx, Fy, Fz, delta, vorticity_strength
):
    """
    adds the vorticity confinement force to the body-force field on the GPU.

    Each thread computes the gradient of the precomputed vorticity magnitude,
    normalizes it to the confinement direction N and evaluates the force
    epsilon * (N x omega) in one interior fluid cell. The resulting force is
    accumulated into the existing force arrays so confinement combines with
    turbulence, buoyancy and any authored force fields.

    Args:
        obstacle_mask (device array): boolean obstacle mask used to skip solids
        omega_x (device array): x-component of the vorticity field
        omega_y (device array): y-component of the vorticity field
        omega_z (device array): z-component of the vorticity field
        omega_magnitude (device array): scalar vorticity magnitude field
        Fx (device array): x-direction force field updated in-place
        Fy (device array): y-direction force field updated in-place
        Fz (device array): z-direction force field updated in-place
        delta (float): grid spacing
        vorticity_strength (float): confinement strength epsilon from the UI
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = omega_magnitude.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if (
        i < 2 or j < 2 or k < 2 or
        i >= nx - 2 or j >= ny - 2 or k >= nz - 2 or
        obstacle_mask[i, j, k]
    ):
        return

    half_inv_delta = 0.5 / delta
    grad_x = (omega_magnitude[i + 1, j, k] - omega_magnitude[i - 1, j, k]) * half_inv_delta
    grad_y = (omega_magnitude[i, j + 1, k] - omega_magnitude[i, j - 1, k]) * half_inv_delta
    grad_z = (omega_magnitude[i, j, k + 1] - omega_magnitude[i, j, k - 1]) * half_inv_delta

    grad_length = math.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z)
    if grad_length <= 1.0e-12:
        return

    nx_dir = grad_x / grad_length
    ny_dir = grad_y / grad_length
    nz_dir = grad_z / grad_length

    wx = omega_x[i, j, k]
    wy = omega_y[i, j, k]
    wz = omega_z[i, j, k]

    Fx[i, j, k] += vorticity_strength * (ny_dir * wz - nz_dir * wy)
    Fy[i, j, k] += vorticity_strength * (nz_dir * wx - nx_dir * wz)
    Fz[i, j, k] += vorticity_strength * (nx_dir * wy - ny_dir * wx)
