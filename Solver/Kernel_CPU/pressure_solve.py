import math

import numpy as np
from numba import njit, prange

import Solver.Kernel_CPU.Boundary_Conditions.domain_bc as BC


@njit(cache=True, parallel=True, fastmath=True)
def pressure_equation_right_side(
    u, v, w, T, obstacle_mask, b, omega_x, omega_y, omega_z, omega_magnitude,
    dt, point_divergence, delta, rho, expansion_rate, t_reference
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
                    i < 1 or j < 1 or k < 1 or
                    i >= nx - 1 or j >= ny - 1 or k >= nz - 1
                ):
                    b[i, j, k] = 0.0
                    omega_x[i, j, k] = 0.0
                    omega_y[i, j, k] = 0.0
                    omega_z[i, j, k] = 0.0
                    omega_magnitude[i, j, k] = 0.0
                    continue

                du_dx = (u[i + 1, j, k] - u[i - 1, j, k]) * half_inv_delta
                dv_dy = (v[i, j + 1, k] - v[i, j - 1, k]) * half_inv_delta
                dw_dz = (w[i, j, k + 1] - w[i, j, k - 1]) * half_inv_delta

                du_dy = (u[i, j + 1, k] - u[i, j - 1, k]) * half_inv_delta
                du_dz = (u[i, j, k + 1] - u[i, j, k - 1]) * half_inv_delta
                dv_dx = (v[i + 1, j, k] - v[i - 1, j, k]) * half_inv_delta
                dv_dz = (v[i, j, k + 1] - v[i, j, k - 1]) * half_inv_delta
                dw_dx = (w[i + 1, j, k] - w[i - 1, j, k]) * half_inv_delta
                dw_dy = (w[i, j + 1, k] - w[i, j - 1, k]) * half_inv_delta

                divergence = du_dx + dv_dy + dw_dz
                thermal_divergence = expansion_rate * (T[i, j, k] - t_reference)
                authored_divergence = point_divergence[i, j, k]

                b[i, j, k] = rho_over_dt * (divergence + authored_divergence - thermal_divergence)

                if obstacle_mask[i, j, k]:
                    omega_x[i, j, k] = 0.0
                    omega_y[i, j, k] = 0.0
                    omega_z[i, j, k] = 0.0
                    omega_magnitude[i, j, k] = 0.0
                    continue

                wx = dw_dy - dv_dz
                wy = du_dz - dw_dx
                wz = dv_dx - du_dy

                omega_x[i, j, k] = wx
                omega_y[i, j, k] = wy
                omega_z[i, j, k] = wz
                omega_magnitude[i, j, k] = math.sqrt(wx * wx + wy * wy + wz * wz)


def _remove_rhs_mean(b):
    """
    Enforce the Neumann compatibility condition by removing the RHS mean.
    """
    interior = b[1:-1, 1:-1, 1:-1]
    if interior.size == 0:
        return

    rhs_mean = float(np.mean(interior, dtype=np.float64))
    if abs(rhs_mean) <= 1.0e-12:
        return

    interior -= rhs_mean


@njit(cache=True, parallel=True, fastmath=True)
def _pressure_poisson_red_black_sor_step(p, b, delta, parity, relaxation_factor):
    """
    Perform one in-place red-black SOR color pass of the 3D pressure Poisson equation.
    """
    nx, ny, nz = p.shape
    delta2 = delta * delta

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                if ((i + j + k) & 1) != parity:
                    continue

                gauss_seidel_value = (
                    p[i + 1, j, k] + p[i - 1, j, k] +
                    p[i, j + 1, k] + p[i, j - 1, k] +
                    p[i, j, k + 1] + p[i, j, k - 1] -
                    delta2 * b[i, j, k]
                ) / 6.0
                p_old = p[i, j, k]
                p[i, j, k] = p_old + relaxation_factor * (gauss_seidel_value - p_old)


@njit(cache=True, parallel=True, fastmath=True)
def _project_velocity_kernel(u, v, w, p, obstacle_mask, dt, delta, rho):
    """
    Apply the pressure projection `u <- u - dt/rho * grad(p)` to one interior cell.
    """
    nx, ny, nz = u.shape
    pressure_coeff = dt / (2.0 * rho * delta)

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                if obstacle_mask[i, j, k]:
                    continue

                u[i, j, k] -= pressure_coeff * (p[i + 1, j, k] - p[i - 1, j, k])
                v[i, j, k] -= pressure_coeff * (p[i, j + 1, k] - p[i, j - 1, k])
                w[i, j, k] -= pressure_coeff * (p[i, j, k + 1] - p[i, j, k - 1])


def pressure_poisson(
    u, v, w, p, T, obstacle_mask, b, omega_x, omega_y, omega_z, omega_magnitude,
    dt, point_divergence, delta, rho, expansion_rate, t_reference,
    max_iter=10, relaxation_factor=1
):
    """
    Host-side CPU pressure Poisson solve matching the GPU-side API.
    """
    pressure_equation_right_side(
        u, v, w, T, obstacle_mask, b, omega_x, omega_y, omega_z, omega_magnitude,
        dt, point_divergence, delta, rho, expansion_rate, t_reference
    )
    _remove_rhs_mean(b)
    p_old = p

    for _ in range(max_iter):
        _pressure_poisson_red_black_sor_step(p_old, b, delta, 0, relaxation_factor)
        BC._pressure_poisson_apply_neumann_bcs(p_old)
        _pressure_poisson_red_black_sor_step(p_old, b, delta, 1, relaxation_factor)
        BC._pressure_poisson_apply_neumann_bcs(p_old)

    BC._pressure_poisson_apply_neumann_bcs(p_old)
    return p_old


def project_velocity(u, v, w, p, obstacle_mask, dt, delta, rho):
    """Project one intermediate velocity field with the solved pressure."""
    _project_velocity_kernel(u, v, w, p, obstacle_mask, dt, delta, rho)
    return u, v, w
