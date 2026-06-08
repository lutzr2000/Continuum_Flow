import numpy as np
from numba import njit, prange

import Solver.Kernel_CPU.Boundary_Conditions.domain_bc as BC
import Solver.Kernel_CPU.kernel_config as kernel_config


@njit(cache=True, inline="always")
def _is_active_pressure_cell(active_tile_mask, i, j, k):
    tile_size = kernel_config.ACTIVE_TILE_SIZE
    return active_tile_mask[i // tile_size, j // tile_size, k // tile_size]


@njit(cache=True, parallel=True, fastmath=True)
def pressure_equation_right_side(
    u,
    v,
    w,
    T,
    b,
    dt,
    point_divergence,
    source_mask,
    source_entry_masks,
    source_extra_pressure_values,
    delta,
    rho,
    expansion_rate,
    t_reference,
    active_tile_mask,
):
    """
    Compute the right hand side of the pressure Poisson equation on the CPU.
    """
    nx, ny, nz = u.shape
    half_inv_delta = 0.5 / delta
    rho_over_dt = rho / dt

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
                    or not _is_active_pressure_cell(active_tile_mask, i, j, k)
                ):
                    b[i, j, k] = 0.0
                    continue

                du_dx = (u[i + 1, j, k] - u[i - 1, j, k]) * half_inv_delta
                dv_dy = (v[i, j + 1, k] - v[i, j - 1, k]) * half_inv_delta
                dw_dz = (w[i, j, k + 1] - w[i, j, k - 1]) * half_inv_delta

                divergence = du_dx + dv_dy + dw_dz
                thermal_divergence = expansion_rate * (T[i, j, k] - t_reference)
                authored_divergence = point_divergence[i, j, k]
                extra_pressure_term = 0.0

                if source_mask[i, j, k]:
                    for source_idx in range(source_entry_masks.shape[0]):
                        if source_entry_masks[source_idx, i, j, k]:
                            extra_pressure_term += source_extra_pressure_values[source_idx]

                b[i, j, k] = (
                    rho_over_dt
                    * (divergence - authored_divergence - thermal_divergence)
                    - extra_pressure_term
                )


@njit(cache=True, fastmath=True)
def _sum_and_count_active_rhs_cells(b, active_tile_mask):
    nx, ny, nz = b.shape
    rhs_sum = np.float32(0.0)
    active_count = 0

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                if not _is_active_pressure_cell(active_tile_mask, i, j, k):
                    continue
                rhs_sum += b[i, j, k]
                active_count += 1

    return rhs_sum, active_count


@njit(cache=True, parallel=True, fastmath=True)
def _subtract_rhs_mean_active_cells(b, rhs_mean, active_tile_mask):
    nx, ny, nz = b.shape

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                if not _is_active_pressure_cell(active_tile_mask, i, j, k):
                    continue
                b[i, j, k] -= rhs_mean


def _remove_rhs_mean(b, active_tile_mask):
    """
    Enforce the Neumann compatibility condition by removing the interior RHS mean.
    """
    rhs_sum, active_count = _sum_and_count_active_rhs_cells(b, active_tile_mask)
    if active_count <= 0:
        return

    rhs_mean = float(rhs_sum) / float(active_count)
    if abs(rhs_mean) <= 1.0e-12:
        return

    _subtract_rhs_mean_active_cells(b, rhs_mean, active_tile_mask)


@njit(cache=True, parallel=True, fastmath=True)
def _reset_inactive_pressure_cells(p, active_tile_mask):
    nx, ny, nz = p.shape

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
                    or not _is_active_pressure_cell(active_tile_mask, i, j, k)
                ):
                    p[i, j, k] = 0.0


@njit(cache=True, parallel=True, fastmath=True)
def _pressure_poisson_red_black_gauss_seidel_step(
    p, b, delta, parity, active_tile_mask
):
    """
    Perform one in-place red-black Gauss-Seidel color pass of the 3D pressure Poisson equation.
    """
    nx, ny, nz = p.shape
    delta2 = delta * delta

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                if ((i + j + k) & 1) != parity or not _is_active_pressure_cell(
                    active_tile_mask, i, j, k
                ):
                    continue

                p[i, j, k] = (
                    p[i + 1, j, k]
                    + p[i - 1, j, k]
                    + p[i, j + 1, k]
                    + p[i, j - 1, k]
                    + p[i, j, k + 1]
                    + p[i, j, k - 1]
                    - delta2 * b[i, j, k]
                ) / 6.0


@njit(cache=True, parallel=True, fastmath=True)
def project_velocity_kernel(
    u, v, w, p, obstacle_mask, dt, delta, rho, active_tile_mask
):
    """
    Apply the pressure projection `u <- u - dt/rho * grad(p)` to one interior cell.
    """
    nx, ny, nz = u.shape
    pressure_coeff = dt / (2.0 * rho * delta)

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                if obstacle_mask[i, j, k] or not _is_active_pressure_cell(
                    active_tile_mask, i, j, k
                ):
                    continue

                u[i, j, k] -= pressure_coeff * (p[i + 1, j, k] - p[i - 1, j, k])
                v[i, j, k] -= pressure_coeff * (p[i, j + 1, k] - p[i, j - 1, k])
                w[i, j, k] -= pressure_coeff * (p[i, j, k + 1] - p[i, j, k - 1])


def pressure_poisson(
    u,
    v,
    w,
    p,
    T,
    b,
    dt,
    point_divergence,
    source_mask,
    source_entry_masks,
    source_extra_pressure_values,
    delta,
    rho,
    expansion_rate,
    t_reference,
    active_tile_mask,
    max_iter=10,
):
    """
    CPU pressure Poisson solve that launches kernels for the RHS,
    red-black Gauss-Seidel iterations and Neumann boundary conditions.

    """
    pressure_equation_right_side(
        u,
        v,
        w,
        T,
        b,
        dt,
        point_divergence,
        source_mask,
        source_entry_masks,
        source_extra_pressure_values,
        delta,
        rho,
        expansion_rate,
        t_reference,
        active_tile_mask,
    )
    _remove_rhs_mean(b, active_tile_mask)
    p_old = p
    _reset_inactive_pressure_cells(p_old, active_tile_mask)

    for _ in range(max_iter):
        _pressure_poisson_red_black_gauss_seidel_step(
            p_old, b, delta, 0, active_tile_mask
        )
        _pressure_poisson_red_black_gauss_seidel_step(
            p_old, b, delta, 1, active_tile_mask
        )
        BC._pressure_poisson_apply_neumann_bcs(p_old)

    return p_old
