from numba import cuda
import math

from Solver.Kernel_GPU.scalar_update import _active_tile_cell_indices


@cuda.jit(cache=True)
def compute_vorticity(
    u,
    v,
    w,
    obstacle_mask,
    omega_magnitude,
    delta,
    active_tile_mask,
):
    """
    Compute vorticity components and scalar magnitude from the velocity field.
    """
    tile_i, tile_j, tile_k, i, j, k, nx, ny, nz = _active_tile_cell_indices(
        omega_magnitude.shape
    )

    if (
        tile_i >= active_tile_mask.shape[0]
        or tile_j >= active_tile_mask.shape[1]
        or tile_k >= active_tile_mask.shape[2]
    ):
        return

    if i >= nx or j >= ny or k >= nz:
        return

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        omega_magnitude[i, j, k] = 0.0
        return

    if active_tile_mask[tile_i, tile_j, tile_k] == 0:
        omega_magnitude[i, j, k] = 0.0
        return

    if obstacle_mask[i, j, k]:
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

    omega_magnitude[i, j, k] = math.sqrt(wx * wx + wy * wy + wz * wz)


@cuda.jit(device=True, inline=True, cache=True)
def apply_vorticity_confinement(
    u,
    v,
    w,
    obstacle_mask,
    omega_magnitude,
    i,
    j,
    k,
    nx,
    ny,
    nz,
    delta,
    vorticity_strength,
):
    """
    Compute the local vorticity confinement force in one GPU cell.
    """
    if (
        i < 2
        or j < 2
        or k < 2
        or i >= nx - 2
        or j >= ny - 2
        or k >= nz - 2
        or obstacle_mask[i, j, k]
    ):
        return 0.0, 0.0, 0.0

    half_inv_delta = 0.5 / delta

    grad_x = (
        omega_magnitude[i + 1, j, k] - omega_magnitude[i - 1, j, k]
    ) * half_inv_delta

    grad_y = (
        omega_magnitude[i, j + 1, k] - omega_magnitude[i, j - 1, k]
    ) * half_inv_delta

    grad_z = (
        omega_magnitude[i, j, k + 1] - omega_magnitude[i, j, k - 1]
    ) * half_inv_delta

    grad_length = math.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z)

    if grad_length <= 1.0e-12:
        return 0.0, 0.0, 0.0

    nx_dir = grad_x / grad_length
    ny_dir = grad_y / grad_length
    nz_dir = grad_z / grad_length

    du_dy = (u[i, j + 1, k] - u[i, j - 1, k]) * half_inv_delta
    du_dz = (u[i, j, k + 1] - u[i, j, k - 1]) * half_inv_delta

    dv_dx = (v[i + 1, j, k] - v[i - 1, j, k]) * half_inv_delta
    dv_dz = (v[i, j, k + 1] - v[i, j, k - 1]) * half_inv_delta

    dw_dx = (w[i + 1, j, k] - w[i - 1, j, k]) * half_inv_delta
    dw_dy = (w[i, j + 1, k] - w[i, j - 1, k]) * half_inv_delta

    wx = dw_dy - dv_dz
    wy = du_dz - dw_dx
    wz = dv_dx - du_dy

    fx = vorticity_strength * (ny_dir * wz - nz_dir * wy)
    fy = vorticity_strength * (nz_dir * wx - nx_dir * wz)
    fz = vorticity_strength * (nx_dir * wy - ny_dir * wx)

    return fx, fy, fz
