import math
from numba import njit, prange

import Solver.Kernel_CPU.advection_schemes as advection_schemes
from Solver.Kernel_CPU.scalar_update import _active_tile_cell_indices

apply_vorticity_confinement = advection_schemes.apply_vorticity_confinement


@njit(cache=True, parallel=True)
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
    nx, ny, nz = omega_magnitude.shape
    total = nx * ny * nz

    for n in prange(total):
        tile_i, tile_j, tile_k, i, j, k, nx, ny, nz = _active_tile_cell_indices(
            n, omega_magnitude.shape
        )

        if (
            tile_i >= active_tile_mask.shape[0]
            or tile_j >= active_tile_mask.shape[1]
            or tile_k >= active_tile_mask.shape[2]
        ):
            continue

        if i >= nx or j >= ny or k >= nz:
            continue

        if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
            omega_magnitude[i, j, k] = 0.0
            continue

        if active_tile_mask[tile_i, tile_j, tile_k] == 0:
            omega_magnitude[i, j, k] = 0.0
            continue

        if obstacle_mask[i, j, k]:
            omega_magnitude[i, j, k] = 0.0
            continue

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
