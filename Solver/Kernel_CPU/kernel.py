from time import perf_counter

import math
import os
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads

import Solver.General.helper_functions as helper_functions
import Solver.General.output_functions as output_functions
import Solver.General.update_data as general_update_data
import Solver.Kernel_CPU.Boundary_Conditions.domain_bc as BC
import Solver.Kernel_CPU.Boundary_Conditions.obstacle_bc as obstacle_bc
import Solver.Kernel_CPU.Boundary_Conditions.source_bc as source_bc
import Solver.Kernel_CPU.time_step as time_step
import Solver.Kernel_CPU.update_data as update_data


@njit(cache=True, parallel=True, fastmath=True)
def update_velocity(
    u, v, w, p, dt, Fx, Fy, Fz, un, vn, wn, delta, rho, nu,
    max_velocity_increment_factor
):
    """First-order upwind CPU velocity update for all three components."""
    nx, ny, nz = u.shape
    dt_over_delta = dt / delta
    pressure_coeff = dt / (2.0 * rho * delta)
    diffusion_coeff = nu * dt / (delta * delta)
    force_coeff = dt / rho
    max_increment = max_velocity_increment_factor * delta / dt

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                u_center = u[i, j, k]
                v_center = v[i, j, k]
                w_center = w[i, j, k]

                if u_center >= 0.0:
                    u_x_high = u_center
                    u_x_low = u[i - 1, j, k]
                    v_x_high = v_center
                    v_x_low = v[i - 1, j, k]
                    w_x_high = w_center
                    w_x_low = w[i - 1, j, k]
                else:
                    u_x_high = u[i + 1, j, k]
                    u_x_low = u_center
                    v_x_high = v[i + 1, j, k]
                    v_x_low = v_center
                    w_x_high = w[i + 1, j, k]
                    w_x_low = w_center

                if v_center >= 0.0:
                    u_y_high = u_center
                    u_y_low = u[i, j - 1, k]
                    v_y_high = v_center
                    v_y_low = v[i, j - 1, k]
                    w_y_high = w_center
                    w_y_low = w[i, j - 1, k]
                else:
                    u_y_high = u[i, j + 1, k]
                    u_y_low = u_center
                    v_y_high = v[i, j + 1, k]
                    v_y_low = v_center
                    w_y_high = w[i, j + 1, k]
                    w_y_low = w_center

                if w_center >= 0.0:
                    u_z_high = u_center
                    u_z_low = u[i, j, k - 1]
                    v_z_high = v_center
                    v_z_low = v[i, j, k - 1]
                    w_z_high = w_center
                    w_z_low = w[i, j, k - 1]
                else:
                    u_z_high = u[i, j, k + 1]
                    u_z_low = u_center
                    v_z_high = v[i, j, k + 1]
                    v_z_low = v_center
                    w_z_high = w[i, j, k + 1]
                    w_z_low = w_center

                convection_x = dt_over_delta * (
                    u_center * (u_x_high - u_x_low) +
                    v_center * (u_y_high - u_y_low) +
                    w_center * (u_z_high - u_z_low)
                )
                convection_y = dt_over_delta * (
                    u_center * (v_x_high - v_x_low) +
                    v_center * (v_y_high - v_y_low) +
                    w_center * (v_z_high - v_z_low)
                )
                convection_z = dt_over_delta * (
                    u_center * (w_x_high - w_x_low) +
                    v_center * (w_y_high - w_y_low) +
                    w_center * (w_z_high - w_z_low)
                )

                diffusion_x = diffusion_coeff * (
                    (u[i + 1, j, k] - 2.0 * u_center + u[i - 1, j, k]) +
                    (u[i, j + 1, k] - 2.0 * u_center + u[i, j - 1, k]) +
                    (u[i, j, k + 1] - 2.0 * u_center + u[i, j, k - 1])
                )
                diffusion_y = diffusion_coeff * (
                    (v[i + 1, j, k] - 2.0 * v_center + v[i - 1, j, k]) +
                    (v[i, j + 1, k] - 2.0 * v_center + v[i, j - 1, k]) +
                    (v[i, j, k + 1] - 2.0 * v_center + v[i, j, k - 1])
                )
                diffusion_z = diffusion_coeff * (
                    (w[i + 1, j, k] - 2.0 * w_center + w[i - 1, j, k]) +
                    (w[i, j + 1, k] - 2.0 * w_center + w[i, j - 1, k]) +
                    (w[i, j, k + 1] - 2.0 * w_center + w[i, j, k - 1])
                )

                pressure_gradient_x = pressure_coeff * (p[i + 1, j, k] - p[i - 1, j, k])
                pressure_gradient_y = pressure_coeff * (p[i, j + 1, k] - p[i, j - 1, k])
                pressure_gradient_z = pressure_coeff * (p[i, j, k + 1] - p[i, j, k - 1])

                u_raw = u_center - convection_x - pressure_gradient_x + diffusion_x + force_coeff * Fx[i, j, k]
                v_raw = v_center - convection_y - pressure_gradient_y + diffusion_y + force_coeff * Fy[i, j, k]
                w_raw = w_center - convection_z - pressure_gradient_z + diffusion_z + force_coeff * Fz[i, j, k]

                du = min(max(u_raw - u_center, -max_increment), max_increment)
                dv = min(max(v_raw - v_center, -max_increment), max_increment)
                dw = min(max(w_raw - w_center, -max_increment), max_increment)

                un[i, j, k] = u_center + du
                vn[i, j, k] = v_center + dv
                wn[i, j, k] = w_center + dw


@njit(cache=True, parallel=True, fastmath=True)
def update_velocity_second_order_upwind(
    u, v, w, p, dt, Fx, Fy, Fz, un, vn, wn, delta, rho, nu,
    max_velocity_increment_factor
):
    """Second-order upwind CPU velocity update with first-order fallback near boundaries."""
    nx, ny, nz = u.shape
    dt_over_delta = dt / delta
    pressure_coeff = dt / (2.0 * rho * delta)
    diffusion_coeff = nu * dt / (delta * delta)
    force_coeff = dt / rho
    max_increment = max_velocity_increment_factor * delta / dt

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                u_center = u[i, j, k]
                v_center = v[i, j, k]
                w_center = w[i, j, k]

                has_im2 = i >= 2
                has_ip2 = i < nx - 2
                has_jm2 = j >= 2
                has_jp2 = j < ny - 2
                has_km2 = k >= 2
                has_kp2 = k < nz - 2

                if u_center >= 0.0:
                    if has_im2:
                        du_dx_term = 0.5 * (3.0 * u_center - 4.0 * u[i - 1, j, k] + u[i - 2, j, k])
                        dv_dx_term = 0.5 * (3.0 * v_center - 4.0 * v[i - 1, j, k] + v[i - 2, j, k])
                        dw_dx_term = 0.5 * (3.0 * w_center - 4.0 * w[i - 1, j, k] + w[i - 2, j, k])
                    else:
                        du_dx_term = u_center - u[i - 1, j, k]
                        dv_dx_term = v_center - v[i - 1, j, k]
                        dw_dx_term = w_center - w[i - 1, j, k]
                else:
                    if has_ip2:
                        du_dx_term = 0.5 * (-3.0 * u_center + 4.0 * u[i + 1, j, k] - u[i + 2, j, k])
                        dv_dx_term = 0.5 * (-3.0 * v_center + 4.0 * v[i + 1, j, k] - v[i + 2, j, k])
                        dw_dx_term = 0.5 * (-3.0 * w_center + 4.0 * w[i + 1, j, k] - w[i + 2, j, k])
                    else:
                        du_dx_term = u[i + 1, j, k] - u_center
                        dv_dx_term = v[i + 1, j, k] - v_center
                        dw_dx_term = w[i + 1, j, k] - w_center

                if v_center >= 0.0:
                    if has_jm2:
                        du_dy_term = 0.5 * (3.0 * u_center - 4.0 * u[i, j - 1, k] + u[i, j - 2, k])
                        dv_dy_term = 0.5 * (3.0 * v_center - 4.0 * v[i, j - 1, k] + v[i, j - 2, k])
                        dw_dy_term = 0.5 * (3.0 * w_center - 4.0 * w[i, j - 1, k] + w[i, j - 2, k])
                    else:
                        du_dy_term = u_center - u[i, j - 1, k]
                        dv_dy_term = v_center - v[i, j - 1, k]
                        dw_dy_term = w_center - w[i, j - 1, k]
                else:
                    if has_jp2:
                        du_dy_term = 0.5 * (-3.0 * u_center + 4.0 * u[i, j + 1, k] - u[i, j + 2, k])
                        dv_dy_term = 0.5 * (-3.0 * v_center + 4.0 * v[i, j + 1, k] - v[i, j + 2, k])
                        dw_dy_term = 0.5 * (-3.0 * w_center + 4.0 * w[i, j + 1, k] - w[i, j + 2, k])
                    else:
                        du_dy_term = u[i, j + 1, k] - u_center
                        dv_dy_term = v[i, j + 1, k] - v_center
                        dw_dy_term = w[i, j + 1, k] - w_center

                if w_center >= 0.0:
                    if has_km2:
                        du_dz_term = 0.5 * (3.0 * u_center - 4.0 * u[i, j, k - 1] + u[i, j, k - 2])
                        dv_dz_term = 0.5 * (3.0 * v_center - 4.0 * v[i, j, k - 1] + v[i, j, k - 2])
                        dw_dz_term = 0.5 * (3.0 * w_center - 4.0 * w[i, j, k - 1] + w[i, j, k - 2])
                    else:
                        du_dz_term = u_center - u[i, j, k - 1]
                        dv_dz_term = v_center - v[i, j, k - 1]
                        dw_dz_term = w_center - w[i, j, k - 1]
                else:
                    if has_kp2:
                        du_dz_term = 0.5 * (-3.0 * u_center + 4.0 * u[i, j, k + 1] - u[i, j, k + 2])
                        dv_dz_term = 0.5 * (-3.0 * v_center + 4.0 * v[i, j, k + 1] - v[i, j, k + 2])
                        dw_dz_term = 0.5 * (-3.0 * w_center + 4.0 * w[i, j, k + 1] - w[i, j, k + 2])
                    else:
                        du_dz_term = u[i, j, k + 1] - u_center
                        dv_dz_term = v[i, j, k + 1] - v_center
                        dw_dz_term = w[i, j, k + 1] - w_center

                convection_x = dt_over_delta * (u_center * du_dx_term + v_center * du_dy_term + w_center * du_dz_term)
                convection_y = dt_over_delta * (u_center * dv_dx_term + v_center * dv_dy_term + w_center * dv_dz_term)
                convection_z = dt_over_delta * (u_center * dw_dx_term + v_center * dw_dy_term + w_center * dw_dz_term)

                diffusion_x = diffusion_coeff * (
                    (u[i + 1, j, k] - 2.0 * u_center + u[i - 1, j, k]) +
                    (u[i, j + 1, k] - 2.0 * u_center + u[i, j - 1, k]) +
                    (u[i, j, k + 1] - 2.0 * u_center + u[i, j, k - 1])
                )
                diffusion_y = diffusion_coeff * (
                    (v[i + 1, j, k] - 2.0 * v_center + v[i - 1, j, k]) +
                    (v[i, j + 1, k] - 2.0 * v_center + v[i, j - 1, k]) +
                    (v[i, j, k + 1] - 2.0 * v_center + v[i, j, k - 1])
                )
                diffusion_z = diffusion_coeff * (
                    (w[i + 1, j, k] - 2.0 * w_center + w[i - 1, j, k]) +
                    (w[i, j + 1, k] - 2.0 * w_center + w[i, j - 1, k]) +
                    (w[i, j, k + 1] - 2.0 * w_center + w[i, j, k - 1])
                )

                pressure_gradient_x = pressure_coeff * (p[i + 1, j, k] - p[i - 1, j, k])
                pressure_gradient_y = pressure_coeff * (p[i, j + 1, k] - p[i, j - 1, k])
                pressure_gradient_z = pressure_coeff * (p[i, j, k + 1] - p[i, j, k - 1])

                u_raw = u_center - convection_x - pressure_gradient_x + diffusion_x + force_coeff * Fx[i, j, k]
                v_raw = v_center - convection_y - pressure_gradient_y + diffusion_y + force_coeff * Fy[i, j, k]
                w_raw = w_center - convection_z - pressure_gradient_z + diffusion_z + force_coeff * Fz[i, j, k]

                du = min(max(u_raw - u_center, -max_increment), max_increment)
                dv = min(max(v_raw - v_center, -max_increment), max_increment)
                dw = min(max(w_raw - w_center, -max_increment), max_increment)

                un[i, j, k] = u_center + du
                vn[i, j, k] = v_center + dv
                wn[i, j, k] = w_center + dw


@njit(cache=True, parallel=True, fastmath=True)
def pressure_equation_right_side(
    u, v, w, T, obstacle_mask, b, omega_x, omega_y, omega_z, omega_magnitude,
    dt, point_divergence, delta, rho, expansion_rate, t_reference
):
    """Compute the pressure RHS and vorticity fields on the CPU."""
    nx, ny, nz = u.shape
    half_inv_delta = 0.5 / delta
    rho_over_dt = rho / dt

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
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
                nonlinear = (
                    du_dx * du_dx +
                    dv_dy * dv_dy +
                    dw_dz * dw_dz +
                    2.0 * (du_dy * dv_dx + du_dz * dw_dx + dv_dz * dw_dy)
                )

                thermal_divergence = expansion_rate * (T[i, j, k] - t_reference)
                authored_divergence = point_divergence[i, j, k]
                b[i, j, k] = rho_over_dt * (divergence + authored_divergence - thermal_divergence) - rho * nonlinear

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


@njit(cache=True, parallel=True, fastmath=True)
def _pressure_poisson_red_black_gauss_seidel_step(p, b, delta, parity):
    """Perform one in-place red/black Gauss-Seidel color sweep on the CPU."""
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


@njit(cache=True, fastmath=True)
def _pressure_poisson_apply_neumann_bcs(p):
    """Apply zero-gradient pressure boundary conditions on all six domain faces."""
    nx, ny, nz = p.shape

    for j in range(ny):
        for k in range(nz):
            p[0, j, k] = p[1, j, k]
            p[nx - 1, j, k] = p[nx - 2, j, k]

    for i in range(nx):
        for k in range(nz):
            p[i, 0, k] = p[i, 1, k]
            p[i, ny - 1, k] = p[i, ny - 2, k]

    for i in range(nx):
        for j in range(ny):
            p[i, j, 0] = p[i, j, 1]
            p[i, j, nz - 1] = p[i, j, nz - 2]


def _pressure_poisson_make_rhs_compatible(b):
    """
    Remove the global mean from the pressure RHS for the Neumann pressure solve.
    """
    b -= np.mean(b, dtype=np.float64)


def pressure_poisson(
    u, v, w, p, T, obstacle_mask, b, omega_x, omega_y, omega_z, omega_magnitude,
    dt, point_divergence, delta, rho, expansion_rate, t_reference,
    max_iter=10
):
    """Host-side CPU pressure solve using red-black Gauss-Seidel iterations."""
    pressure_equation_right_side(
        u, v, w, T, obstacle_mask, b, omega_x, omega_y, omega_z, omega_magnitude,
        dt, point_divergence, delta, rho, expansion_rate, t_reference
    )
    # The pressure solve uses Neumann boundaries, so the discrete RHS must have
    # zero mean; removing the average keeps the Poisson problem compatible.
    # The general issue is otherwise that the absolute value of pressure is unedfined
    # without dirichlet conditions
    _pressure_poisson_make_rhs_compatible(b)

    for _ in range(max_iter):
        _pressure_poisson_red_black_gauss_seidel_step(p, b, delta, 0)
        _pressure_poisson_apply_neumann_bcs(p)
        _pressure_poisson_red_black_gauss_seidel_step(p, b, delta, 1)
        _pressure_poisson_apply_neumann_bcs(p)
    return p


@njit(cache=True, parallel=True, fastmath=True)
def update_force_fields(
    Fx_base, Fy_base, Fz_base,
    turbulence_Fx_a, turbulence_Fy_a, turbulence_Fz_a,
    turbulence_Fx_b, turbulence_Fy_b, turbulence_Fz_b,
    turbulence_cos_coeffs, turbulence_sin_coeffs,
    turbulence_count, animated_force_x, animated_force_y, animated_force_z,
    Fx, Fy, Fz
):
    """Update body-force fields from base, animated and turbulence contributions."""
    nx, ny, nz = Fx.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                fx = Fx_base[i, j, k] + animated_force_x
                fy = Fy_base[i, j, k] + animated_force_y
                fz = Fz_base[i, j, k] + animated_force_z

                for turbulence_index in range(turbulence_count):
                    cos_coeff = turbulence_cos_coeffs[turbulence_index]
                    sin_coeff = turbulence_sin_coeffs[turbulence_index]
                    fx += (
                        cos_coeff * turbulence_Fx_a[turbulence_index, i, j, k] +
                        sin_coeff * turbulence_Fx_b[turbulence_index, i, j, k]
                    )
                    fy += (
                        cos_coeff * turbulence_Fy_a[turbulence_index, i, j, k] +
                        sin_coeff * turbulence_Fy_b[turbulence_index, i, j, k]
                    )
                    fz += (
                        cos_coeff * turbulence_Fz_a[turbulence_index, i, j, k] +
                        sin_coeff * turbulence_Fz_b[turbulence_index, i, j, k]
                    )

                Fx[i, j, k] = fx
                Fy[i, j, k] = fy
                Fz[i, j, k] = fz


@njit(cache=True, parallel=True, fastmath=True)
def buoyancy_approximation(T, Fz, buoyancy_factor, t_reference):
    """Accumulate Boussinesq buoyancy into the z-force field on the CPU."""
    nx, ny, nz = T.shape
    g = 9.81
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                Fz[i, j, k] += g * buoyancy_factor * (T[i, j, k] - t_reference)


@njit(cache=True, parallel=True, fastmath=True)
def apply_vorticity_confinement(
    obstacle_mask, omega_x, omega_y, omega_z, omega_magnitude, Fx, Fy, Fz, delta, vorticity_strength
):
    """Accumulate vorticity confinement forces into the body-force fields on the CPU."""
    nx, ny, nz = omega_magnitude.shape
    half_inv_delta = 0.5 / delta
    for i in prange(2, nx - 2):
        for j in range(2, ny - 2):
            for k in range(2, nz - 2):
                if obstacle_mask[i, j, k]:
                    continue

                grad_x = (omega_magnitude[i + 1, j, k] - omega_magnitude[i - 1, j, k]) * half_inv_delta
                grad_y = (omega_magnitude[i, j + 1, k] - omega_magnitude[i, j - 1, k]) * half_inv_delta
                grad_z = (omega_magnitude[i, j, k + 1] - omega_magnitude[i, j, k - 1]) * half_inv_delta

                grad_length = math.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z)
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


@njit(cache=True, parallel=True, fastmath=True)
def update_scalar_fields(
    T, smoke, fuel, u, v, w, dt, T_out, smoke_out, fuel_out, flame_out,
    delta, temperature_dissipation_rate, temperature_production_rate,
    smoke_dissipation_rate, smoke_production_rate,
    fuel_burn_rate, fuel_ignition_temperature, t_reference
):
    """Update temperature, smoke and fuel in one CPU transport sweep."""
    nx, ny, nz = u.shape
    dt_over_delta = dt / delta

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                uijk = u[i, j, k]
                vijk = v[i, j, k]
                wijk = w[i, j, k]

                T_center = T[i, j, k]
                T_xm = T[i - 1, j, k]
                T_xp = T[i + 1, j, k]
                T_ym = T[i, j - 1, k]
                T_yp = T[i, j + 1, k]
                T_zm = T[i, j, k - 1]
                T_zp = T[i, j, k + 1]

                smoke_center = smoke[i, j, k]
                smoke_xm = smoke[i - 1, j, k]
                smoke_xp = smoke[i + 1, j, k]
                smoke_ym = smoke[i, j - 1, k]
                smoke_yp = smoke[i, j + 1, k]
                smoke_zm = smoke[i, j, k - 1]
                smoke_zp = smoke[i, j, k + 1]

                fuel_center = fuel[i, j, k]
                fuel_xm = fuel[i - 1, j, k]
                fuel_xp = fuel[i + 1, j, k]
                fuel_ym = fuel[i, j - 1, k]
                fuel_yp = fuel[i, j + 1, k]
                fuel_zm = fuel[i, j, k - 1]
                fuel_zp = fuel[i, j, k + 1]

                if uijk >= 0.0:
                    temp_dx = T_center - T_xm
                    smoke_dx = smoke_center - smoke_xm
                    fuel_dx = fuel_center - fuel_xm
                else:
                    temp_dx = T_xp - T_center
                    smoke_dx = smoke_xp - smoke_center
                    fuel_dx = fuel_xp - fuel_center

                if vijk >= 0.0:
                    temp_dy = T_center - T_ym
                    smoke_dy = smoke_center - smoke_ym
                    fuel_dy = fuel_center - fuel_ym
                else:
                    temp_dy = T_yp - T_center
                    smoke_dy = smoke_yp - smoke_center
                    fuel_dy = fuel_yp - fuel_center

                if wijk >= 0.0:
                    temp_dz = T_center - T_zm
                    smoke_dz = smoke_center - smoke_zm
                    fuel_dz = fuel_center - fuel_zm
                else:
                    temp_dz = T_zp - T_center
                    smoke_dz = smoke_zp - smoke_center
                    fuel_dz = fuel_zp - fuel_center

                temp_convection = dt_over_delta * (uijk * temp_dx + vijk * temp_dy + wijk * temp_dz)
                smoke_convection = dt_over_delta * (uijk * smoke_dx + vijk * smoke_dy + wijk * smoke_dz)
                fuel_convection = dt_over_delta * (uijk * fuel_dx + vijk * fuel_dy + wijk * fuel_dz)

                if T_center > fuel_ignition_temperature:
                    fuel_source = -fuel_burn_rate * fuel_center
                else:
                    fuel_source = 0.0

                temperature_source = (
                    -temperature_dissipation_rate * (T_center - t_reference) +
                    temperature_production_rate * (-fuel_source)
                )
                smoke_source = smoke_production_rate * (-fuel_source) - smoke_dissipation_rate * smoke_center

                T_updated = T_center - temp_convection + dt * temperature_source
                smoke_updated = smoke_center - smoke_convection + dt * smoke_source
                fuel_updated = fuel_center - fuel_convection + dt * fuel_source

                T_out[i, j, k] = max(T_updated, 0.0)
                smoke_out[i, j, k] = max(smoke_updated, 0.0)
                fuel_out[i, j, k] = max(fuel_updated, 0.0)
                flame_out[i, j, k] = 1.0 if fuel_source < 0.0 else 0.0


def apply_all_BC(
    u, v, w, p, T, smoke, fuel, flame,
    bc_config,
    has_obstacle, obstacle_mask, obstacle_velocity_x, obstacle_velocity_y, obstacle_velocity_z,
    has_source, source_mask, source_velocity_mask, source_temperature, source_smoke, source_fuel,
    source_velocity_x, source_velocity_y, source_velocity_z,
):
    """Apply domain, obstacle and source constraints in the fixed overwrite order."""
    u, v, w, p, T = BC.apply_all_BC(u, v, w, p, T, bc_config)

    if has_obstacle:
        u, v, w, smoke, fuel, flame = obstacle_bc.obstacle_bc(
            u, v, w, smoke, fuel, flame,
            obstacle_mask, obstacle_velocity_x, obstacle_velocity_y, obstacle_velocity_z,
        )

    if has_source:
        u, v, w, T, smoke, fuel = source_bc.source_bc(
            u, v, w, T, smoke, fuel,
            source_mask, source_velocity_mask,
            source_temperature, source_smoke, source_fuel,
            source_velocity_x, source_velocity_y, source_velocity_z,
        )
    return u, v, w, p, T, smoke, fuel, flame


def _enqueue_host_output(
    write_queue,
    buffer_pool,
    buffered_variables,
    field_map,
    output_index,
    time_value,
    output_field_config,
):
    """Copy one CPU output frame into shared memory and enqueue it for writing."""
    fields = buffer_pool.get()
    for variable_name in buffered_variables:
        np.copyto(fields[variable_name]["array"], field_map[variable_name])
    write_queue.put((int(output_index), float(time_value), fields, output_field_config))


def main(config=None):
    total_start_time = perf_counter()
    memory_tracker = helper_functions.MemoryUsageTracker("RAM", helper_functions._sample_process_memory_usage)
    simulation_params = helper_functions.apply_config(config)
    if config is not None:
        simulations = config.get("simulations") or []
        if simulations:
            simulation_params["_simulation_cfg"] = simulations[0]
    update_data.rebuild_cpu_boundary_data(simulation_params)

    cancel_flag_path = (((simulation_params.get("meta") or {}).get("cancel_flag_path")) or "").strip()

    print("################################################################")
    print("Initialise")
    print("Cell count: ", int(simulation_params["NX"] * simulation_params["NY"] * simulation_params["NZ"]))

    available_cores = os.cpu_count() or 1
    reserved_writer_cores = max(0, int(simulation_params.get("OUTPUT_FORWARDER_COUNT", 0)))
    cpu_count = max(1, available_cores - 2 - reserved_writer_cores)
    set_num_threads(cpu_count)

    cpu_fields, cpu_constants = update_data.upload_simulation_state_to_cpu(simulation_params)
    memory_tracker.sample()

    u = cpu_fields["u"]
    v = cpu_fields["v"]
    w = cpu_fields["w"]
    u_work = cpu_fields["u_work"]
    v_work = cpu_fields["v_work"]
    w_work = cpu_fields["w_work"]
    p = cpu_fields["p"]
    pressure_rhs = cpu_fields["pressure_rhs"]
    T = cpu_fields["T"]
    temperature_work = cpu_fields["temperature_work"]
    smoke = cpu_fields["smoke"]
    smoke_work = cpu_fields["smoke_work"]
    fuel = cpu_fields["fuel"]
    fuel_work = cpu_fields["fuel_work"]
    flame = cpu_fields["flame"]
    flame_work = cpu_fields["flame_work"]
    vorticity_x = cpu_fields["vorticity_x"]
    vorticity_y = cpu_fields["vorticity_y"]
    vorticity_z = cpu_fields["vorticity_z"]
    vorticity_magnitude = cpu_fields["vorticity_magnitude"]
    Fx = cpu_fields["Fx"]
    Fy = cpu_fields["Fy"]
    Fz = cpu_fields["Fz"]
    Fx_base = cpu_fields["Fx_base"]
    Fy_base = cpu_fields["Fy_base"]
    Fz_base = cpu_fields["Fz_base"]
    point_divergence = cpu_fields["point_divergence"]
    turbulence_Fx_a = cpu_fields["turbulence_Fx_a"]
    turbulence_Fy_a = cpu_fields["turbulence_Fy_a"]
    turbulence_Fz_a = cpu_fields["turbulence_Fz_a"]
    turbulence_Fx_b = cpu_fields["turbulence_Fx_b"]
    turbulence_Fy_b = cpu_fields["turbulence_Fy_b"]
    turbulence_Fz_b = cpu_fields["turbulence_Fz_b"]
    turbulence_cos_coeffs = cpu_fields["turbulence_cos_coeffs"]
    turbulence_sin_coeffs = cpu_fields["turbulence_sin_coeffs"]
    turbulence_angular_frequencies = np.asarray(
        simulation_params["force_field_data"]["turbulence"]["angular_frequencies"],
        dtype=simulation_params["PRECISION"],
    )
    turbulence_count = int(turbulence_angular_frequencies.size)
    obstacle_mask = cpu_fields["obstacle_mask"]
    obstacle_velocity_x = cpu_fields["obstacle_velocity_x"]
    obstacle_velocity_y = cpu_fields["obstacle_velocity_y"]
    obstacle_velocity_z = cpu_fields["obstacle_velocity_z"]
    source_mask = cpu_fields["source_mask"]
    source_velocity_mask = cpu_fields["source_velocity_mask"]
    source_temperature = cpu_fields["source_temperature"]
    source_smoke = cpu_fields["source_smoke"]
    source_fuel = cpu_fields["source_fuel"]
    source_velocity_x = cpu_fields["source_velocity_x"]
    source_velocity_y = cpu_fields["source_velocity_y"]
    source_velocity_z = cpu_fields["source_velocity_z"]
    velocity_maxima = cpu_fields["velocity_maxima"]

    host_fields = {
        "u": u,
        "v": v,
        "w": w,
        "p": p,
        "T": T,
        "smoke": smoke,
        "fuel": fuel,
        "flame": flame,
    }

    update_data.update_dynamic_boundary_data_on_cpu(simulation_params, cpu_fields, cpu_constants, 0.0)
    memory_tracker.sample()

    u, v, w, p, T, smoke, fuel, flame = apply_all_BC(
        u, v, w, p, T, smoke, fuel, flame,
        simulation_params["BC_CONFIG"],
        cpu_constants["HAS_OBSTACLE"], obstacle_mask, obstacle_velocity_x, obstacle_velocity_y, obstacle_velocity_z,
        cpu_constants["HAS_SOURCE"], source_mask, source_velocity_mask, source_temperature, source_smoke, source_fuel,
        source_velocity_x, source_velocity_y, source_velocity_z,
    )

    t = 0.0
    general_update_data.update_animated_constants(simulation_params, cpu_constants, t)
    animated_force = update_data.update_animated_source_force_values(simulation_params, cpu_fields, t)
    fx_max, fy_max, fz_max = helper_functions.estimate_theoretical_force_maxima(
        cpu_constants,
        simulation_params["ANIMATION_STATE"],
    )

    write_queue = None
    writer_threads = None
    shared_memory_blocks = None
    solver_diverged = False
    cancel_requested = False

    dt, solver_diverged = time_step.compute_new_timestep_cpu(
        u, v, w, velocity_maxima, fx_max, fy_max, fz_max,
        cpu_constants["RHO"], cpu_constants["DELTA"], cpu_constants["NU"], simulation_params["CFL_MAX"],
        max_dt=1.0 / simulation_params["OUTPUT_FPS"],
    )
    if solver_diverged:
        print("ERROR: The solver diverged before output setup, stopping the simulation!")
    else:
        next_output_time = 0.0
        output_index = 0

        write_queue, buffer_pool, writer_threads, shared_memory_blocks = output_functions.setup_output(
            simulation_params["OUTPATH"],
            simulation_params["FRAME_START"],
            simulation_params["OUTPUT_BUFFER_VARIABLES"],
            host_fields,
            simulation_params["WRITE_QUEUE_SIZE"],
            simulation_params["OUTPUT_FORWARDER_COUNT"],
            simulation_params["DELTA"],
            simulation_params["HOST_VDB_WRITER"],
            storage_dtype=simulation_params["OUTPUT_DTYPE"],
        )

        print("Start time iteration")
        helper_functions.emit_progress(0.0, t)
        memory_tracker.sample()

        while t < simulation_params["T_MAX"]:
            if cancel_flag_path and Path(cancel_flag_path).exists():
                cancel_requested = True
                print("Bake cancellation requested. Stopping the simulation cleanly...")
                break

            if simulation_params.get("HAS_DYNAMIC_BOUNDARIES", False):
                update_data.update_dynamic_boundary_data_on_cpu(simulation_params, cpu_fields, cpu_constants, t)
                obstacle_mask = cpu_fields["obstacle_mask"]
                obstacle_velocity_x = cpu_fields["obstacle_velocity_x"]
                obstacle_velocity_y = cpu_fields["obstacle_velocity_y"]
                obstacle_velocity_z = cpu_fields["obstacle_velocity_z"]
                source_mask = cpu_fields["source_mask"]
                source_velocity_mask = cpu_fields["source_velocity_mask"]
                source_temperature = cpu_fields["source_temperature"]
                source_smoke = cpu_fields["source_smoke"]
                source_fuel = cpu_fields["source_fuel"]
                source_velocity_x = cpu_fields["source_velocity_x"]
                source_velocity_y = cpu_fields["source_velocity_y"]
                source_velocity_z = cpu_fields["source_velocity_z"]

            general_update_data.update_animated_constants(simulation_params, cpu_constants, t)
            animated_force = update_data.update_animated_source_force_values(simulation_params, cpu_fields, t)
            source_mask = cpu_fields["source_mask"]
            source_velocity_mask = cpu_fields["source_velocity_mask"]
            source_temperature = cpu_fields["source_temperature"]
            source_smoke = cpu_fields["source_smoke"]
            source_fuel = cpu_fields["source_fuel"]
            source_velocity_x = cpu_fields["source_velocity_x"]
            source_velocity_y = cpu_fields["source_velocity_y"]
            source_velocity_z = cpu_fields["source_velocity_z"]

            if turbulence_count > 0:
                turbulence_cos_coeffs[:] = np.cos(turbulence_angular_frequencies * t)
                turbulence_sin_coeffs[:] = np.sin(turbulence_angular_frequencies * t)

            update_force_fields(
                Fx_base, Fy_base, Fz_base,
                turbulence_Fx_a, turbulence_Fy_a, turbulence_Fz_a,
                turbulence_Fx_b, turbulence_Fy_b, turbulence_Fz_b,
                turbulence_cos_coeffs, turbulence_sin_coeffs,
                turbulence_count,
                np.float32(animated_force["x"]),
                np.float32(animated_force["y"]),
                np.float32(animated_force["z"]),
                Fx, Fy, Fz,
            )
            buoyancy_approximation(T, Fz, cpu_constants["BUOANCY_FACTOR"], cpu_constants["T_REFERENCE"])

            p = pressure_poisson(
                u, v, w, p, T, obstacle_mask, pressure_rhs,
                vorticity_x, vorticity_y, vorticity_z, vorticity_magnitude, dt, point_divergence,
                cpu_constants["DELTA"], cpu_constants["RHO"], cpu_constants["EXPANSION_RATE"],
                cpu_constants["T_REFERENCE"], simulation_params["MAX_ITER"],
            )

            if cpu_constants["VORTICITY"] > 0.0:
                apply_vorticity_confinement(
                    obstacle_mask, vorticity_x, vorticity_y, vorticity_z, vorticity_magnitude,
                    Fx, Fy, Fz, cpu_constants["DELTA"], cpu_constants["VORTICITY"],
                )

            if simulation_params["VELOCITY_ADVECTION_SCHEME"] == "FIRST_ORDER_UPWIND":
                update_velocity(
                    u, v, w, p, dt, Fx, Fy, Fz, u_work, v_work, w_work,
                    cpu_constants["DELTA"], cpu_constants["RHO"], cpu_constants["NU"],
                    simulation_params["MAX_VELOCITY_INCREMENT_FACTOR"],
                )
            else:
                update_velocity_second_order_upwind(
                    u, v, w, p, dt, Fx, Fy, Fz, u_work, v_work, w_work,
                    cpu_constants["DELTA"], cpu_constants["RHO"], cpu_constants["NU"],
                    simulation_params["MAX_VELOCITY_INCREMENT_FACTOR"],
                )

            u, u_work = u_work, u
            v, v_work = v_work, v
            w, w_work = w_work, w

            update_scalar_fields(
                T, smoke, fuel, u, v, w, dt,
                temperature_work, smoke_work, fuel_work, flame_work,
                cpu_constants["DELTA"],
                cpu_constants["TEMPERATURE_DISSIPATION_RATE"],
                cpu_constants["TEMPERATURE_PRODUCTION_RATE"],
                cpu_constants["SMOKE_DISSIPATION_RATE"],
                cpu_constants["SMOKE_PRODUCTION_RATE"],
                cpu_constants["FUEL_BURN_RATE"],
                cpu_constants["FUEL_IGNITION_TEMPERATURE"],
                cpu_constants["T_REFERENCE"],
            )

            T, temperature_work = temperature_work, T
            smoke, smoke_work = smoke_work, smoke
            fuel, fuel_work = fuel_work, fuel
            flame, flame_work = flame_work, flame

            u, v, w, p, T, smoke, fuel, flame = apply_all_BC(
                u, v, w, p, T, smoke, fuel, flame,
                simulation_params["BC_CONFIG"],
                cpu_constants["HAS_OBSTACLE"], obstacle_mask, obstacle_velocity_x, obstacle_velocity_y, obstacle_velocity_z,
                cpu_constants["HAS_SOURCE"], source_mask, source_velocity_mask, source_temperature, source_smoke, source_fuel,
                source_velocity_x, source_velocity_y, source_velocity_z,
            )

            host_fields["u"] = u
            host_fields["v"] = v
            host_fields["w"] = w
            host_fields["p"] = p
            host_fields["T"] = T
            host_fields["smoke"] = smoke
            host_fields["fuel"] = fuel
            host_fields["flame"] = flame

            while t >= next_output_time:
                _enqueue_host_output(
                    write_queue,
                    buffer_pool,
                    simulation_params["OUTPUT_BUFFER_VARIABLES"],
                    host_fields,
                    output_index,
                    t,
                    simulation_params["OUTPUT_FIELD_CONFIG"],
                )

                output_index += 1
                next_output_time += simulation_params["OUTPUT_TIME_STEP"]
                helper_functions.emit_progress(t / simulation_params["T_MAX"] * 100.0, t)

                if simulation_params["OUTPUT_STATUS"]:
                    print("#################################################")
                    print(f"Simulation time {t} sec")
                    print(f"Current dt: {np.round(dt, 5)}")

            t += dt

            dt_new, solver_diverged = time_step.compute_new_timestep_cpu(
                u, v, w, velocity_maxima, fx_max, fy_max, fz_max,
                cpu_constants["RHO"], cpu_constants["DELTA"], cpu_constants["NU"], simulation_params["CFL_MAX"],
                max_dt=1.0 / simulation_params["OUTPUT_FPS"],
            )
            memory_tracker.sample()
            if solver_diverged:
                print("ERROR: The solver diverged, stopping the simulation!")
                break
            dt = dt_new

    if write_queue is not None:
        output_functions.shutdown_output(write_queue, writer_threads, shared_memory_blocks)

    if solver_diverged:
        print("Simulation stopped after solver divergence.")
    elif cancel_requested:
        print("Simulation cancelled after clean shutdown.")
    else:
        helper_functions.emit_progress(100.0, simulation_params["T_MAX"])
        print("Simulation finished!")
    memory_tracker.sample()
    memory_tracker.print_summary()
    total_runtime = perf_counter() - total_start_time
    print(f"Solver runtime: {total_runtime:.3f} s")
    print("################################################################")
