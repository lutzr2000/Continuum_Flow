import numpy as np
from numba import njit, prange

import Solver.Kernel_CPU.Boundary_Conditions.domain_bc as BC


@njit(cache=True, parallel=True, fastmath=True)
def pressure_equation_right_side(
    u, v, w, T, b, dt, point_divergence, source_extra_pressure, delta, rho, expansion_rate, t_reference
):
    """Compute the right hand side of the pressure Poisson equation on the CPU."""
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
                    continue

                du_dx = (u[i + 1, j, k] - u[i - 1, j, k]) * half_inv_delta
                dv_dy = (v[i, j + 1, k] - v[i, j - 1, k]) * half_inv_delta
                dw_dz = (w[i, j, k + 1] - w[i, j, k - 1]) * half_inv_delta

                divergence = du_dx + dv_dy + dw_dz
                thermal_divergence = expansion_rate * (T[i, j, k] - t_reference)
                authored_divergence = point_divergence[i, j, k]
                extra_pressure_term = source_extra_pressure[i, j, k]

                b[i, j, k] = rho_over_dt * (divergence - authored_divergence - thermal_divergence) - extra_pressure_term


def _remove_rhs_mean(b):
    """Enforce the Neumann compatibility condition by removing the interior RHS mean."""
    interior = b[1:-1, 1:-1, 1:-1]
    if interior.size == 0:
        return

    rhs_mean = float(np.mean(interior, dtype=np.float64))
    if abs(rhs_mean) <= 1.0e-12:
        return

    np.subtract(interior, rhs_mean, out=interior)


@njit(cache=True, parallel=True, fastmath=True)
def _pressure_poisson_red_black_gauss_seidel_step(p, b, delta, parity):
    """Perform one in-place red-black Gauss-Seidel color pass of the 3D pressure Poisson equation."""
    nx, ny, nz = p.shape
    delta2 = delta * delta

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                if ((i + j + k) & 1) != parity:
                    continue

                p[i, j, k] = (
                    p[i + 1, j, k] + p[i - 1, j, k] +
                    p[i, j + 1, k] + p[i, j - 1, k] +
                    p[i, j, k + 1] + p[i, j, k - 1] -
                    delta2 * b[i, j, k]
                ) / 6.0


@njit(cache=True, parallel=True, fastmath=True)
def project_velocity_kernel(u, v, w, p, obstacle_mask, dt, delta, rho):
    """Apply the pressure projection `u <- u - dt/rho * grad(p)` to one interior cell."""
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
    u,
    v,
    w,
    p,
    T,
    b,
    dt,
    point_divergence,
    source_extra_pressure,
    delta,
    rho,
    expansion_rate,
    t_reference,
    max_iter=10,
):
    """Host-side CPU pressure Poisson solve aligned with the GPU-side API."""
    pressure_equation_right_side(
        u, v, w, T, b, dt, point_divergence, source_extra_pressure, delta, rho, expansion_rate, t_reference
    )
    _remove_rhs_mean(b)
    p_old = p

    for _ in range(max_iter):
        _pressure_poisson_red_black_gauss_seidel_step(p_old, b, delta, 0)
        _pressure_poisson_red_black_gauss_seidel_step(p_old, b, delta, 1)
        BC._pressure_poisson_apply_neumann_bcs(p_old)

    return p_old


def project_velocity(u, v, w, p, obstacle_mask, dt, delta, rho):
    """Project one intermediate velocity field with the solved pressure."""
    project_velocity_kernel(u, v, w, p, obstacle_mask, dt, delta, rho)
    return u, v, w
