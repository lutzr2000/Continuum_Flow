from time import perf_counter

import math
from pathlib import Path
import numpy as np
from numba import cuda

import Solver.Kernel_GPU.Boundary_Conditions.domain_bc as BC
import Solver.General.Helper_Functions as Helper_Functions
import Solver.General.output_functions as Output_Functions
import Solver.Kernel_GPU.Update_data as Update_data
import Solver.Kernel_GPU.kernel_config as kernel_config
import Solver.Kernel_GPU.Boundary_Conditions.obstacle_bc as Obstacle_BC
import Solver.Kernel_GPU.Boundary_Conditions.Source_BC as source_bc
import Solver.Kernel_GPU.time_step as Time_Step

# ===============================
# Methods
# ===============================

@cuda.jit(cache=True)
def update_velocity(
    u, v, w, p, dt, Fx, Fy, Fz, un, vn, wn, delta, rho, nu,
    max_velocity_increment_factor
):
    """
    CUDA kernel that updates all three velocity components based on the
    momentum equation. Convection is done by first order upwind, diffusion with
    central differences.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        dt (float): timestep size
        Fx (device array): x-direction body force field
        Fy (device array): y-direction body force field
        Fz (device array): z-direction body force field
        un (device array): output array for updated x-velocity
        vn (device array): output array for updated y-velocity
        wn (device array): output array for updated z-velocity
        delta (float): grid spacing
        rho (float): density
        nu (float): kinematic viscosity
        max_velocity_increment_factor (float): maximum allowed per-step velocity
            change relative to delta / dt
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape
    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    dt_over_delta = dt / delta
    pressure_coeff = dt / (2.0 * rho * delta)
    diffusion_coeff = nu * dt / (delta * delta)
    force_coeff = dt / rho

    u_center = u[i, j, k]
    v_center = v[i, j, k]
    w_center = w[i, j, k]

    #------------Upwinding-------------------
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

    #------------Convection-------------------
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

    #------------Diffusion-------------------
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

    #------------Pressure-------------------
    pressure_gradient_x = pressure_coeff * (p[i + 1, j, k] - p[i - 1, j, k])
    pressure_gradient_y = pressure_coeff * (p[i, j + 1, k] - p[i, j - 1, k])
    pressure_gradient_z = pressure_coeff * (p[i, j, k + 1] - p[i, j, k - 1])

    #------------Update-------------------
    u_raw = u_center - convection_x - pressure_gradient_x + diffusion_x + force_coeff * Fx[i, j, k]
    v_raw = v_center - convection_y - pressure_gradient_y + diffusion_y + force_coeff * Fy[i, j, k]
    w_raw = w_center - convection_z - pressure_gradient_z + diffusion_z + force_coeff * Fz[i, j, k]

    max_increment = max_velocity_increment_factor * delta / dt
    du = min(max(u_raw - u_center, -max_increment), max_increment)
    dv = min(max(v_raw - v_center, -max_increment), max_increment)
    dw = min(max(w_raw - w_center, -max_increment), max_increment)

    un[i, j, k] = u_center + du
    vn[i, j, k] = v_center + dv
    wn[i, j, k] = w_center + dw

@cuda.jit(cache=True)
def update_velocity_second_order_upwind(
    u, v, w, p, dt, Fx, Fy, Fz, un, vn, wn, delta, rho, nu,
    max_velocity_increment_factor
):
    """
    CUDA kernel that updates all three velocity components based on the
    momentum equation. Convection is done by second order upwind where enough
    stencil points exist, otherwise first order upwind is used near boundaries.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        dt (float): timestep size
        Fx (device array): x-direction body force field
        Fy (device array): y-direction body force field
        Fz (device array): z-direction body force field
        un (device array): output array for updated x-velocity
        vn (device array): output array for updated y-velocity
        wn (device array): output array for updated z-velocity
        delta (float): grid spacing
        rho (float): density
        nu (float): kinematic viscosity
        max_velocity_increment_factor (float): maximum allowed per-step velocity
            change relative to delta / dt
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape
    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    dt_over_delta = dt / delta
    pressure_coeff = dt / (2.0 * rho * delta)
    diffusion_coeff = nu * dt / (delta * delta)
    force_coeff = dt / rho

    u_center = u[i, j, k]
    v_center = v[i, j, k]
    w_center = w[i, j, k]

    has_im2 = i >= 2
    has_ip2 = i < nx - 2
    has_jm2 = j >= 2
    has_jp2 = j < ny - 2
    has_km2 = k >= 2
    has_kp2 = k < nz - 2

    # Use a second-order upwind stencil where available and fall back to the
    # original first-order stencil one cell away from the domain boundary.
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

    convection_x = dt_over_delta * (
        u_center * du_dx_term +
        v_center * du_dy_term +
        w_center * du_dz_term
    )

    convection_y = dt_over_delta * (
        u_center * dv_dx_term +
        v_center * dv_dy_term +
        w_center * dv_dz_term
    )

    convection_z = dt_over_delta * (
        u_center * dw_dx_term +
        v_center * dw_dy_term +
        w_center * dw_dz_term
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

    max_increment = max_velocity_increment_factor * delta / dt
    du = min(max(u_raw - u_center, -max_increment), max_increment)
    dv = min(max(v_raw - v_center, -max_increment), max_increment)
    dw = min(max(w_raw - w_center, -max_increment), max_increment)

    un[i, j, k] = u_center + du
    vn[i, j, k] = v_center + dv
    wn[i, j, k] = w_center + dw

@cuda.jit(cache=True)
def pressure_equation_right_side(
    u, v, w, T, obstacle_mask, b, omega_x, omega_y, omega_z, omega_magnitude,
    dt, point_divergence, delta, rho, expansion_rate, t_reference
):
    """
    CUDA kernel that computes the right hand side of the pressure Poisson equation.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        T (device array): temperature field
        obstacle_mask (device array): boolean obstacle mask used to zero solid vorticity
        b (device array): output array for the pressure equation right hand side
        omega_x (device array): output array for the x-component of vorticity
        omega_y (device array): output array for the y-component of vorticity
        omega_z (device array): output array for the z-component of vorticity
        omega_magnitude (device array): output array for the vorticity magnitude
        dt (float): timestep size
        point_divergence (device array): authored divergence source field
        delta (float): grid spacing
        rho (float): density
        expansion_rate (float): thermal expansion coupling
        t_reference (float): reference temperature
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape

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

    half_inv_delta = 0.5 / delta
    rho_over_dt = rho / dt

    #------------Derivatives on main diagonal-------------------
    du_dx = (u[i + 1, j, k] - u[i - 1, j, k]) * half_inv_delta
    dv_dy = (v[i, j + 1, k] - v[i, j - 1, k]) * half_inv_delta
    dw_dz = (w[i, j, k + 1] - w[i, j, k - 1]) * half_inv_delta

    #------------Of-diagonal derivatives-------------------
    du_dy = (u[i, j + 1, k] - u[i, j - 1, k]) * half_inv_delta
    du_dz = (u[i, j, k + 1] - u[i, j, k - 1]) * half_inv_delta
    dv_dx = (v[i + 1, j, k] - v[i - 1, j, k]) * half_inv_delta
    dv_dz = (v[i, j, k + 1] - v[i, j, k - 1]) * half_inv_delta
    dw_dx = (w[i + 1, j, k] - w[i - 1, j, k]) * half_inv_delta
    dw_dy = (w[i, j + 1, k] - w[i, j - 1, k]) * half_inv_delta

    # This comes from the divergence of the time derivative
    divergence = du_dx + dv_dy + dw_dz

    # This comes from the divergence of the convection term
    nonlinear = (
        du_dx * du_dx +
        dv_dy * dv_dy +
        dw_dz * dw_dz +
        2.0 * (du_dy * dv_dx + du_dz * dw_dx + dv_dz * dw_dy)
    )

    #------------Artifical thermal divergence-------------------
    thermal_divergence = expansion_rate * (T[i, j, k] - t_reference)
    authored_divergence = point_divergence[i, j, k]

    #------------Right hand side-------------------
    b[i, j, k] = rho_over_dt * (divergence + authored_divergence - thermal_divergence) - rho * nonlinear

    if obstacle_mask[i, j, k]:
        omega_x[i, j, k] = 0.0
        omega_y[i, j, k] = 0.0
        omega_z[i, j, k] = 0.0
        omega_magnitude[i, j, k] = 0.0
        return

    wx = dw_dy - dv_dz
    wy = du_dz - dw_dx
    wz = dv_dx - du_dy

    omega_x[i, j, k] = wx
    omega_y[i, j, k] = wy
    omega_z[i, j, k] = wz
    omega_magnitude[i, j, k] = math.sqrt(wx * wx + wy * wy + wz * wz)

@cuda.jit(cache=True)
def _pressure_poisson_red_black_gauss_seidel_step(p, b, delta, parity):
    """
    Perform one in-place red-black Gauss-Seidel color pass of the 3D pressure
    Poisson equation on the GPU.

    Only interior cells whose `(i + j + k) % 2` matches `parity` are updated in
    this launch. The complementary color is expected to be updated by a second
    launch in the same iteration.

    Args:
        p (device array): pressure field updated in place
        b (device array): right hand side of the pressure Poisson equation
        delta (float): grid spacing
        parity (int): `0` for red cells, `1` for black cells
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = p.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if (
        i < 1 or j < 1 or k < 1 or
        i >= nx - 1 or j >= ny - 1 or k >= nz - 1 or
        ((i + j + k) & 1) != parity
    ):
        return

    delta2 = delta * delta
    p[i, j, k] = (
        p[i + 1, j, k] + p[i - 1, j, k] +
        p[i, j + 1, k] + p[i, j - 1, k] +
        p[i, j, k + 1] + p[i, j, k - 1] -
        delta2 * b[i, j, k]
    ) / 6.0


@cuda.jit(cache=True)
def _pressure_poisson_apply_neumann_bcs(p):
    """
    applies the hard-coded zero-gradient pressure boundary conditions on all
    six domain faces on the GPU.

    The pressure Poisson solve uses homogeneous Neumann boundary conditions,
    meaning the pressure at the boundary is copied from the adjacent interior
    cell. This kernel writes the boundary values after each Jacobi iteration so
    the next iteration starts from a pressure field with valid boundary values.

    Args:
        p (device array): pressure field whose domain boundaries will be updated
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = p.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if i == 0:
        p[i, j, k] = p[1, j, k]
    elif i == nx - 1:
        p[i, j, k] = p[nx - 2, j, k]

    if j == 0:
        p[i, j, k] = p[i, 1, k]
    elif j == ny - 1:
        p[i, j, k] = p[i, ny - 2, k]

    if k == 0:
        p[i, j, k] = p[i, j, 1]
    elif k == nz - 1:
        p[i, j, k] = p[i, j, nz - 2]


def pressure_poisson(
    u, v, w, p, T, obstacle_mask, p_work, b, omega_x, omega_y, omega_z, omega_magnitude,
    dt, point_divergence, delta, rho, expansion_rate, t_reference,
    max_iter=10, threadsperblock_3d=None
):
    """
    Host-side pressure Poisson solve that launches CUDA kernels for the RHS,
    Jacobi iterations and Neumann boundary conditions.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        T (device array): temperature field
        obstacle_mask (device array): boolean obstacle mask used to zero solid vorticity
        p_work (device array): work array for the pressure iteration
        b (device array): work array for the pressure equation right hand side
        omega_x (device array): work array for the x-component of vorticity
        omega_y (device array): work array for the y-component of vorticity
        omega_z (device array): work array for the z-component of vorticity
        omega_magnitude (device array): work array for the vorticity magnitude
        dt (float): timestep size
        point_divergence (device array): authored divergence source field
        delta (float): grid spacing
        rho (float): density
        expansion_rate (float): thermal expansion coupling
        t_reference (float): reference temperature
        max_iter (int): number of pressure iterations
        pressure_solver (str): `jacobi` or `red_black_gauss_seidel` / `rbgs`
    Returns:
        device array: updated pressure field
    """
    if threadsperblock_3d is None:
        threadsperblock_3d = kernel_config.THREADS_PER_BLOCK_3D
    blockspergrid_3d = kernel_config.volume_blocks_per_grid(u.shape, threadsperblock_3d)

    pressure_equation_right_side[blockspergrid_3d, threadsperblock_3d](
        u, v, w, T, obstacle_mask, b, omega_x, omega_y, omega_z, omega_magnitude,
        dt, point_divergence, delta, rho, expansion_rate, t_reference
    )

    p_old = p

    for _ in range(max_iter):
        _pressure_poisson_red_black_gauss_seidel_step[blockspergrid_3d, threadsperblock_3d](p_old, b, delta, 0)
        _pressure_poisson_apply_neumann_bcs[blockspergrid_3d, threadsperblock_3d](p_old)
        _pressure_poisson_red_black_gauss_seidel_step[blockspergrid_3d, threadsperblock_3d](p_old, b, delta, 1)
        _pressure_poisson_apply_neumann_bcs[blockspergrid_3d, threadsperblock_3d](p_old)
    return p_old



@cuda.jit(cache=True)
def update_force_fields(Fx_base, Fy_base, Fz_base,
                        turbulence_Fx_a, turbulence_Fy_a, turbulence_Fz_a,
                        turbulence_Fx_b, turbulence_Fy_b, turbulence_Fz_b,
                        turbulence_cos_coeffs, turbulence_sin_coeffs,
                        turbulence_count, animated_force_x, animated_force_y, animated_force_z,
                        Fx, Fy, Fz):
    """
    Update body-force fields from base fields and animated turbulence bases.

    The expensive smooth turbulence fields are precomputed on the host. Per
    timestep this kernel only mixes them with host-computed sine/cosine
    coefficients, keeping the force update bandwidth-bound and predictable.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = Fx.shape

    if i >= nx or j >= ny or k >= nz:
        return

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

@cuda.jit(cache=True)
def buoyancy_approximation(T, Fz, buoyancy_factor, t_reference):
    """
    computes the buoyancy force in z-direction with the Boussinesq approximation on the GPU.

    Each thread updates one interior cell of the z-direction body-force field.
    The force is derived from the local temperature difference to the reference
    temperature and scaled by gravity and the configured buoyancy factor.

    Args:
        T (device array): temperature field
        Fz (device array): z-direction force field that will be updated in-place
        buoyancy_factor (float): thermal expansion coefficient used by the model
        t_reference (float): reference temperature
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = T.shape

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    g = 9.81
    Fz[i, j, k] += g * buoyancy_factor * (T[i, j, k] - t_reference)

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


@cuda.jit(cache=True)
def update_scalar_fields(T, smoke, fuel, u, v, w, dt, T_out, smoke_out, fuel_out, flame_out,
                         delta, temperature_dissipation_rate, temperature_production_rate,
                         smoke_dissipation_rate, smoke_production_rate,
                         fuel_burn_rate, fuel_ignition_temperature, t_reference):
    """
    updates temperature, smoke and fuel in one GPU transport sweep.

    Convection is evaluated with first-order upwinding and the source terms
    model fuel ignition, temperature release and smoke production. A flame
    indicator is written alongside the updated scalar fields.

    Args:
        T (device array): temperature field
        smoke (device array): smoke density field
        fuel (device array): fuel density field
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        dt (float): timestep size
        T_out (device array): output array for updated temperature
        smoke_out (device array): output array for updated smoke density
        fuel_out (device array): output array for updated fuel density
        flame_out (device array): output array for the flame indicator
        delta (float): grid spacing
        temperature_dissipation_rate (float): temperature dissipation coefficient
        temperature_production_rate (float): temperature production coefficient
        smoke_dissipation_rate (float): smoke dissipation coefficient
        smoke_production_rate (float): smoke production coefficient
        fuel_burn_rate (float): burning rate for ignited fuel
        fuel_ignition_temperature (float): ignition threshold for fuel burning
        t_reference (float): reference temperature
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    dt_over_delta = dt / delta

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
    """
    Apply domain, obstacle and source constraints in the fixed overwrite order.
    """
    u, v, w, p, T = BC.apply_all_BC(u, v, w, p, T, bc_config)

    if has_obstacle:
        u, v, w, smoke, fuel, flame = Obstacle_BC.obstacle_bc(
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


# ===============================
# Main
# ===============================

def main(config=None):
    #------------Initialise-------------------
    total_start_time = perf_counter()
    simulation_params = Helper_Functions.apply_config(config)
    cancel_flag_path = (((simulation_params.get("meta") or {}).get("cancel_flag_path")) or "").strip()

    print("################################################################")
    print('Initialise')
    print('Cell count: ', int(simulation_params["NX"] * simulation_params["NY"] * simulation_params["NZ"]))

    #------------Fields-------------------
    gpu_fields, gpu_constants = Update_data.upload_simulation_state_to_gpu(simulation_params)

    u = gpu_fields["u"]
    v = gpu_fields["v"]
    w = gpu_fields["w"]
    u_work = gpu_fields["u_work"]
    v_work = gpu_fields["v_work"]
    w_work = gpu_fields["w_work"]
    p = gpu_fields["p"]
    pressure_work = gpu_fields["pressure_work"]
    pressure_rhs = gpu_fields["pressure_rhs"]
    T = gpu_fields["T"]
    temperature_work = gpu_fields["temperature_work"]
    smoke = gpu_fields["smoke"]
    smoke_work = gpu_fields["smoke_work"]
    fuel = gpu_fields["fuel"]
    fuel_work = gpu_fields["fuel_work"]
    flame = gpu_fields["flame"]
    flame_work = gpu_fields["flame_work"]
    vorticity_x = gpu_fields["vorticity_x"]
    vorticity_y = gpu_fields["vorticity_y"]
    vorticity_z = gpu_fields["vorticity_z"]
    vorticity_magnitude = gpu_fields["vorticity_magnitude"]
    Fx = gpu_fields["Fx"]
    Fy = gpu_fields["Fy"]
    Fz = gpu_fields["Fz"]
    Fx_base = gpu_fields["Fx_base"]
    Fy_base = gpu_fields["Fy_base"]
    Fz_base = gpu_fields["Fz_base"]
    point_divergence = gpu_fields["point_divergence"]
    turbulence_Fx_a = gpu_fields["turbulence_Fx_a"]
    turbulence_Fy_a = gpu_fields["turbulence_Fy_a"]
    turbulence_Fz_a = gpu_fields["turbulence_Fz_a"]
    turbulence_Fx_b = gpu_fields["turbulence_Fx_b"]
    turbulence_Fy_b = gpu_fields["turbulence_Fy_b"]
    turbulence_Fz_b = gpu_fields["turbulence_Fz_b"]
    turbulence_cos_coeffs = gpu_fields["turbulence_cos_coeffs"]
    turbulence_sin_coeffs = gpu_fields["turbulence_sin_coeffs"]
    turbulence_angular_frequencies = np.asarray(
        simulation_params["force_field_data"]["turbulence"]["angular_frequencies"],
        dtype=simulation_params["PRECISION"],
    )
    turbulence_count = int(turbulence_angular_frequencies.size)
    obstacle_mask = gpu_fields["obstacle_mask"]
    obstacle_velocity_x = gpu_fields["obstacle_velocity_x"]
    obstacle_velocity_y = gpu_fields["obstacle_velocity_y"]
    obstacle_velocity_z = gpu_fields["obstacle_velocity_z"]
    source_mask = gpu_fields["source_mask"]
    source_velocity_mask = gpu_fields["source_velocity_mask"]
    source_temperature = gpu_fields["source_temperature"]
    source_smoke = gpu_fields["source_smoke"]
    source_fuel = gpu_fields["source_fuel"]
    source_velocity_x = gpu_fields["source_velocity_x"]
    source_velocity_y = gpu_fields["source_velocity_y"]
    source_velocity_z = gpu_fields["source_velocity_z"]
    velocity_maxima = gpu_fields["velocity_maxima"]
    velocity_maxima_host_zeros = np.zeros(3, dtype=np.float32)
    device_fields = {
        "u": u,
        "v": v,
        "w": w,
        "p": p,
        "T": T,
        "smoke": smoke,
        "fuel": fuel,
        "flame": flame,
    }

    #------------Update dynamic masks-------------------
    Update_data.update_dynamic_boundary_data_on_gpu(simulation_params, gpu_fields, gpu_constants, 0.0)

    #------------Apply all BCs-------------------
    u, v, w, p, T, smoke, fuel, flame = apply_all_BC(
        u, v, w, p, T, smoke, fuel, flame,
        simulation_params["BC_CONFIG"],
        gpu_constants["HAS_OBSTACLE"], obstacle_mask, obstacle_velocity_x, obstacle_velocity_y, obstacle_velocity_z,
        gpu_constants["HAS_SOURCE"], source_mask, source_velocity_mask, source_temperature, source_smoke, source_fuel,
        source_velocity_x, source_velocity_y, source_velocity_z,
    )

    t = 0.0
    Update_data.update_animated_gpu_constants(simulation_params, gpu_constants, t)
    animated_force = Update_data.update_animated_source_force_values(
        simulation_params,
        gpu_fields,
        t,
    )
    fx_max, fy_max, fz_max = Helper_Functions.estimate_theoretical_force_maxima(
        gpu_constants,
        simulation_params["ANIMATION_STATE"],
    )

    write_queue = None
    writer_threads = None
    shared_memory_blocks = None
    solver_diverged = False
    cancel_requested = False

    #------------Dynamic time step-------------------
    Time_Step.reset_velocity_maxima(velocity_maxima, velocity_maxima_host_zeros)
    dt, solver_diverged = Time_Step.compute_new_timestep_gpu(
        u, v, w, velocity_maxima, fx_max, fy_max, fz_max,
        gpu_constants["RHO"], gpu_constants["DELTA"], gpu_constants["NU"], simulation_params["CFL_MAX"],
        max_dt=1.0 / simulation_params["OUTPUT_FPS"],
    )
    if solver_diverged:
        print('ERROR: The solver diverged before output setup, stopping the simulation!')
    else:
        #------------Prepare Output-------------------
        next_output_time = 0.0
        output_index = 0

        write_queue, buffer_pool, writer_threads, shared_memory_blocks = Output_Functions.setup_output(
            simulation_params["OUTPATH"],
            simulation_params["FRAME_START"],
            simulation_params["OUTPUT_BUFFER_VARIABLES"],
            Helper_Functions.select_fields(device_fields, simulation_params["OUTPUT_BUFFER_VARIABLES"]),
            simulation_params["WRITE_QUEUE_SIZE"],
            simulation_params["OUTPUT_FORWARDER_COUNT"],
            simulation_params["DELTA"],
            simulation_params["HOST_VDB_WRITER"],
            storage_dtype=simulation_params["OUTPUT_DTYPE"],
        )

        #------------Main time loop-------------------
        print('Start time iteration')
        Helper_Functions.emit_progress(0.0, t)
        blockspergrid_3d = kernel_config.volume_blocks_per_grid(u.shape, kernel_config.THREADS_PER_BLOCK_3D)

        while t < simulation_params["T_MAX"]:
            if cancel_flag_path and Path(cancel_flag_path).exists():
                cancel_requested = True
                print('Bake cancellation requested. Stopping the simulation cleanly...')
                break

            #------------Update-------------------
            if simulation_params.get("HAS_DYNAMIC_BOUNDARIES", False):
                Update_data.update_dynamic_boundary_data_on_gpu(simulation_params, gpu_fields, gpu_constants, t)

            Update_data.update_animated_gpu_constants(simulation_params, gpu_constants, t)
            animated_force = Update_data.update_animated_source_force_values(
                simulation_params,
                gpu_fields,
                t,
            )

            #------------Forces-------------------
            if turbulence_count > 0:
                turbulence_cos_coeffs.copy_to_device(np.cos(turbulence_angular_frequencies * t))
                turbulence_sin_coeffs.copy_to_device(np.sin(turbulence_angular_frequencies * t))
            update_force_fields[blockspergrid_3d, kernel_config.THREADS_PER_BLOCK_3D](
                Fx_base, Fy_base, Fz_base,
                turbulence_Fx_a, turbulence_Fy_a, turbulence_Fz_a,
                turbulence_Fx_b, turbulence_Fy_b, turbulence_Fz_b,
                turbulence_cos_coeffs, turbulence_sin_coeffs,
                turbulence_count,
                np.float32(animated_force["x"]),
                np.float32(animated_force["y"]),
                np.float32(animated_force["z"]),
                Fx, Fy, Fz
            )

            buoyancy_approximation[blockspergrid_3d, kernel_config.THREADS_PER_BLOCK_3D](
                T, Fz, gpu_constants["BUOANCY_FACTOR"], gpu_constants["T_REFERENCE"]
            )
            #------------Pressure-------------------
            p = pressure_poisson(
                u, v, w, p, T, obstacle_mask, pressure_work, pressure_rhs,
                vorticity_x, vorticity_y, vorticity_z, vorticity_magnitude, dt, point_divergence,
                gpu_constants["DELTA"], gpu_constants["RHO"], gpu_constants["EXPANSION_RATE"],
                gpu_constants["T_REFERENCE"], simulation_params["MAX_ITER"]
            )

            if gpu_constants["VORTICITY"] > 0.0:
                apply_vorticity_confinement[blockspergrid_3d, kernel_config.THREADS_PER_BLOCK_3D](
                    obstacle_mask, vorticity_x, vorticity_y, vorticity_z, vorticity_magnitude,
                    Fx, Fy, Fz, gpu_constants["DELTA"], gpu_constants["VORTICITY"]
                )

            #------------Velocity-------------------
            if simulation_params["VELOCITY_ADVECTION_SCHEME"] == "FIRST_ORDER_UPWIND":
                update_velocity[blockspergrid_3d, kernel_config.THREADS_PER_BLOCK_3D](
                    u, v, w, p, dt, Fx, Fy, Fz, u_work, v_work, w_work,
                    gpu_constants["DELTA"], gpu_constants["RHO"], gpu_constants["NU"],
                    simulation_params["MAX_VELOCITY_INCREMENT_FACTOR"],
                )
            else:
                update_velocity_second_order_upwind[blockspergrid_3d, kernel_config.THREADS_PER_BLOCK_3D](
                    u, v, w, p, dt, Fx, Fy, Fz, u_work, v_work, w_work,
                    gpu_constants["DELTA"], gpu_constants["RHO"], gpu_constants["NU"],
                    simulation_params["MAX_VELOCITY_INCREMENT_FACTOR"],
                )

            u, u_work = u_work, u
            v, v_work = v_work, v
            w, w_work = w_work, w

            #------------Scalars-------------------
            update_scalar_fields[blockspergrid_3d, kernel_config.THREADS_PER_BLOCK_3D](
                T, smoke, fuel, u, v, w, dt,
                temperature_work, smoke_work, fuel_work, flame_work,
                gpu_constants["DELTA"],
                gpu_constants["TEMPERATURE_DISSIPATION_RATE"],
                gpu_constants["TEMPERATURE_PRODUCTION_RATE"],
                gpu_constants["SMOKE_DISSIPATION_RATE"],
                gpu_constants["SMOKE_PRODUCTION_RATE"],
                gpu_constants["FUEL_BURN_RATE"],
                gpu_constants["FUEL_IGNITION_TEMPERATURE"],
                gpu_constants["T_REFERENCE"],
            )

            #------------Swap-------------------
            T, temperature_work = temperature_work, T
            smoke, smoke_work = smoke_work, smoke
            fuel, fuel_work = fuel_work, fuel
            flame, flame_work = flame_work, flame

            #------------Apply all BCs-------------------
            u, v, w, p, T, smoke, fuel, flame = apply_all_BC(
                u, v, w, p, T, smoke, fuel, flame,
                simulation_params["BC_CONFIG"],
                gpu_constants["HAS_OBSTACLE"], obstacle_mask, obstacle_velocity_x, obstacle_velocity_y, obstacle_velocity_z,
                gpu_constants["HAS_SOURCE"], source_mask, source_velocity_mask, source_temperature, source_smoke, source_fuel,
                source_velocity_x, source_velocity_y, source_velocity_z,
            )

            #------------Output-------------------
            device_fields["u"] = u
            device_fields["v"] = v
            device_fields["w"] = w
            device_fields["p"] = p
            device_fields["T"] = T
            device_fields["smoke"] = smoke
            device_fields["fuel"] = fuel
            device_fields["flame"] = flame

            while t >= next_output_time:
                Output_Functions.enqueue_device_output(
                    write_queue,
                    buffer_pool,
                    simulation_params["OUTPUT_BUFFER_VARIABLES"],
                    Helper_Functions.select_fields(device_fields, simulation_params["OUTPUT_BUFFER_VARIABLES"]),
                    output_index,
                    t,
                    simulation_params["OUTPUT_FIELD_CONFIG"],
                )

                output_index += 1
                next_output_time += simulation_params["OUTPUT_TIME_STEP"]
                Helper_Functions.emit_progress(t / simulation_params["T_MAX"] * 100.0, t)

                if simulation_params["OUTPUT_STATUS"]:
                    print('#################################################')
                    print(f'Simulation time {t} sec')
                    print(f'Current dt: {np.round(dt, 5)}')

            #------------Dynamic time step-------------------
            t += dt

            Time_Step.reset_velocity_maxima(velocity_maxima, velocity_maxima_host_zeros)
            dt_new, solver_diverged = Time_Step.compute_new_timestep_gpu(
                u, v, w, velocity_maxima, fx_max, fy_max, fz_max,
                gpu_constants["RHO"], gpu_constants["DELTA"], gpu_constants["NU"], simulation_params["CFL_MAX"],
                max_dt=1.0 / simulation_params["OUTPUT_FPS"],
            )
            if solver_diverged:
                print('ERROR: The solver diverged, stopping the simulation!')
                break
            dt = dt_new

    #------------Empty write queue-------------------
    if write_queue is not None:
        Output_Functions.shutdown_output(write_queue, writer_threads, shared_memory_blocks)

    #------------Conclusion-------------------
    if solver_diverged:
        print('Simulation stopped after solver divergence.')
    elif cancel_requested:
        print('Simulation cancelled after clean shutdown.')
    else:
        Helper_Functions.emit_progress(100.0, simulation_params["T_MAX"])
        print('Simulation finished!')
    total_runtime = perf_counter() - total_start_time
    print(f'Solver runtime: {total_runtime:.3f} s')
    print("################################################################")
