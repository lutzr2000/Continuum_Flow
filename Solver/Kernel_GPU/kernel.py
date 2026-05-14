from time import perf_counter

import math
from pathlib import Path
import warnings
import numpy as np
from numba import cuda
from numba.cuda.core.errors import NumbaPerformanceWarning as CudaNumbaPerformanceWarning

import Solver.Kernel_GPU.Boundary_Conditions.domain_bc as BC
import Solver.General.helper_functions as helper_functions
import Solver.General.output_functions as output_functions
import Solver.General.update_data as general_update_data
import Solver.Kernel_GPU.update_data as update_data
import Solver.Kernel_GPU.kernel_config as kernel_config
import Solver.Kernel_GPU.Boundary_Conditions.obstacle_bc as obstacle_bc
import Solver.Kernel_GPU.Boundary_Conditions.source_bc as source_bc
import Solver.Kernel_GPU.time_step as time_step

warnings.filterwarnings(
    "ignore",
    message=r"Grid size .* will likely result in GPU under-utilization due to low occupancy\.",
    category=CudaNumbaPerformanceWarning,
)

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
    u, v, w, p, T, obstacle_mask, b, omega_x, omega_y, omega_z, omega_magnitude,
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
                         fuel_burn_rate, fuel_ignition_temperature, minimum_oxygen_concentration, t_reference):
    """
    updates temperature, smoke and fuel in one GPU transport sweep.

    Convection is evaluated with first-order upwinding and the source terms
    model fuel ignition, temperature release and smoke production. A continuous
    flame intensity field is written alongside the updated scalar fields.

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
        flame_out (device array): output array for the flame intensity
        delta (float): grid spacing
        temperature_dissipation_rate (float): temperature dissipation coefficient
        temperature_production_rate (float): temperature production coefficient
        smoke_dissipation_rate (float): smoke dissipation coefficient
        smoke_production_rate (float): smoke production coefficient
        fuel_burn_rate (float): burning rate for ignited fuel
        fuel_ignition_temperature (float): ignition threshold for fuel burning
        minimum_oxygen_concentration (float): minimum oxygen concentration
            required for fuel burning
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

    oxygen_center = 100.0 - smoke_center

    if T_center > fuel_ignition_temperature and fuel_center > 0.0 and oxygen_center >= minimum_oxygen_concentration:
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
    smoke_out[i, j, k] = min(max(smoke_updated, 0.0), 100.0)
    fuel_out[i, j, k] = min(max(fuel_updated, 0.0), 100.0)
    flame_out[i, j, k] = max(-fuel_source, 0.0)


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
    u, v, w, p, T, smoke, fuel = BC.apply_all_BC(u, v, w, p, T, smoke, fuel, bc_config)

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


def _initialise_solver(config):
    """
    Initialise the GPU solver state before the first simulation step.

    This prepares the runtime configuration, uploads all simulation fields to
    the GPU, applies the initial boundary conditions, evaluates animated
    constants at time zero and computes the first stable timestep.

    Args:
        config (dict | None): optional simulation override configuration

    Returns:
        dict: mutable solver state used by main() and _run_time_step()
    """
    #------------Runtime config-------------------
    total_start_time = perf_counter()
    timing_stats = {}
    step_count = 0
    output_frame_count = 0

    section_start = perf_counter()
    memory_tracker = helper_functions.MemoryUsageTracker("VRAM", helper_functions._sample_gpu_memory_usage)
    simulation_params = helper_functions.apply_config(config)
    cancel_flag_path = (((simulation_params.get("meta") or {}).get("cancel_flag_path")) or "").strip()
    helper_functions._record_timing(timing_stats, "init_config", perf_counter() - section_start)

    print("################################################################")
    print('Initialise')
    print('Cell count: ', int(simulation_params["NX"] * simulation_params["NY"] * simulation_params["NZ"]))

    #------------Upload fields-------------------
    section_start = perf_counter()
    gpu_fields, gpu_constants = update_data.upload_simulation_state_to_gpu(simulation_params)
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "init_upload_fields", perf_counter() - section_start)

    section_start = perf_counter()
    memory_tracker.sample()
    helper_functions._record_timing(timing_stats, "init_memory_sample", perf_counter() - section_start)

    u = gpu_fields["u"]
    v = gpu_fields["v"]
    w = gpu_fields["w"]
    u_work = gpu_fields["u_work"]
    v_work = gpu_fields["v_work"]
    w_work = gpu_fields["w_work"]
    p = gpu_fields["p"]
    T = gpu_fields["T"]
    temperature_work = gpu_fields["temperature_work"]
    smoke = gpu_fields["smoke"]
    smoke_work = gpu_fields["smoke_work"]
    fuel = gpu_fields["fuel"]
    fuel_work = gpu_fields["fuel_work"]
    flame = gpu_fields["flame"]
    flame_work = gpu_fields["flame_work"]
    turbulence_angular_frequencies = np.asarray(
        simulation_params["force_field_data"]["turbulence"]["angular_frequencies"],
        dtype=simulation_params["PRECISION"],
    )
    turbulence_count = int(turbulence_angular_frequencies.size)
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

    #------------Initial dynamic boundaries-------------------
    section_start = perf_counter()
    update_data.update_dynamic_boundary_data_on_gpu(simulation_params, gpu_fields, gpu_constants, 0.0)
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "init_dynamic_boundaries", perf_counter() - section_start)

    section_start = perf_counter()
    memory_tracker.sample()
    helper_functions._record_timing(timing_stats, "init_memory_sample", perf_counter() - section_start)

    #------------Initial boundary conditions-------------------
    section_start = perf_counter()
    u, v, w, p, T, smoke, fuel, flame = apply_all_BC(
        u, v, w, p, T, smoke, fuel, flame,
        simulation_params["BC_CONFIG"],
        gpu_constants["HAS_OBSTACLE"],
        gpu_fields["obstacle_mask"],
        gpu_fields["obstacle_velocity_x"],
        gpu_fields["obstacle_velocity_y"],
        gpu_fields["obstacle_velocity_z"],
        gpu_constants["HAS_SOURCE"],
        gpu_fields["source_mask"],
        gpu_fields["source_velocity_mask"],
        gpu_fields["source_temperature"],
        gpu_fields["source_smoke"],
        gpu_fields["source_fuel"],
        gpu_fields["source_velocity_x"],
        gpu_fields["source_velocity_y"],
        gpu_fields["source_velocity_z"],
    )
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "init_apply_bc", perf_counter() - section_start)

    #------------Initial animated state-------------------
    t = 0.0
    section_start = perf_counter()
    general_update_data.update_animated_constants(simulation_params, gpu_constants, t)
    animated_force = update_data.update_animated_source_force_values(
        simulation_params,
        gpu_fields,
        t,
    )
    fx_max, fy_max, fz_max = helper_functions.estimate_theoretical_force_maxima(
        gpu_constants,
        simulation_params["ANIMATION_STATE"],
    )
    helper_functions._record_timing(timing_stats, "init_animated_state", perf_counter() - section_start)

    write_queue = None
    writer_threads = None
    shared_memory_blocks = None
    solver_diverged = False
    cancel_requested = False

    #------------Initial timestep-------------------
    section_start = perf_counter()
    time_step.reset_velocity_maxima(gpu_fields["velocity_maxima"], velocity_maxima_host_zeros)
    dt, solver_diverged = time_step.compute_new_timestep_gpu(
        u, v, w, gpu_fields["velocity_maxima"], fx_max, fy_max, fz_max,
        gpu_constants["RHO"], gpu_constants["DELTA"], gpu_constants["NU"], simulation_params["CFL_MAX"],
        max_dt=1.0 / simulation_params["OUTPUT_FPS"],
    )
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "init_timestep", perf_counter() - section_start)

    return {
        "total_start_time": total_start_time,
        "timing_stats": timing_stats,
        "step_count": step_count,
        "output_frame_count": output_frame_count,
        "memory_tracker": memory_tracker,
        "simulation_params": simulation_params,
        "cancel_flag_path": cancel_flag_path,
        "gpu_fields": gpu_fields,
        "gpu_constants": gpu_constants,
        "u": u,
        "v": v,
        "w": w,
        "u_work": u_work,
        "v_work": v_work,
        "w_work": w_work,
        "p": p,
        "T": T,
        "temperature_work": temperature_work,
        "smoke": smoke,
        "smoke_work": smoke_work,
        "fuel": fuel,
        "fuel_work": fuel_work,
        "flame": flame,
        "flame_work": flame_work,
        "turbulence_angular_frequencies": turbulence_angular_frequencies,
        "turbulence_count": turbulence_count,
        "velocity_maxima_host_zeros": velocity_maxima_host_zeros,
        "device_fields": device_fields,
        "t": t,
        "dt": dt,
        "fx_max": fx_max,
        "fy_max": fy_max,
        "fz_max": fz_max,
        "write_queue": write_queue,
        "writer_threads": writer_threads,
        "shared_memory_blocks": shared_memory_blocks,
        "solver_diverged": solver_diverged,
        "cancel_requested": cancel_requested,
    }


def _run_time_step(state, blockspergrid_3d):
    """
    Advance the GPU solver by exactly one simulation timestep.

    The step updates animated inputs, recomputes force fields, solves pressure,
    advances velocity and scalar buffers, swaps working arrays, reapplies all
    boundary conditions and stores the current field references back into the
    shared solver state.

    Args:
        state (dict): mutable solver state assembled by _initialise_solver()
        blockspergrid_3d (tuple): CUDA launch grid for volume kernels

    Returns:
        dict: updated solver state after one timestep
    """
    simulation_params = state["simulation_params"]
    gpu_fields = state["gpu_fields"]
    gpu_constants = state["gpu_constants"]
    timing_stats = state["timing_stats"]
    device_fields = state["device_fields"]
    turbulence_angular_frequencies = state["turbulence_angular_frequencies"]
    turbulence_count = state["turbulence_count"]

    u = state["u"]
    v = state["v"]
    w = state["w"]
    u_work = state["u_work"]
    v_work = state["v_work"]
    w_work = state["w_work"]
    p = state["p"]
    T = state["T"]
    temperature_work = state["temperature_work"]
    smoke = state["smoke"]
    smoke_work = state["smoke_work"]
    fuel = state["fuel"]
    fuel_work = state["fuel_work"]
    flame = state["flame"]
    flame_work = state["flame_work"]
    t = state["t"]
    dt = state["dt"]

    #------------Update dynamic inputs-------------------
    if simulation_params.get("HAS_DYNAMIC_BOUNDARIES", False):
        section_start = perf_counter()
        update_data.update_dynamic_boundary_data_on_gpu(simulation_params, gpu_fields, gpu_constants, t)
        cuda.synchronize()
        helper_functions._record_timing(timing_stats, "loop_dynamic_boundaries", perf_counter() - section_start)

    section_start = perf_counter()
    general_update_data.update_animated_constants(simulation_params, gpu_constants, t)
    animated_force = update_data.update_animated_source_force_values(
        simulation_params,
        gpu_fields,
        t,
    )
    helper_functions._record_timing(timing_stats, "loop_animated_state", perf_counter() - section_start)

    #------------Update force fields-------------------
    if turbulence_count > 0:
        section_start = perf_counter()
        gpu_fields["turbulence_cos_coeffs"].copy_to_device(np.cos(turbulence_angular_frequencies * t))
        gpu_fields["turbulence_sin_coeffs"].copy_to_device(np.sin(turbulence_angular_frequencies * t))
        cuda.synchronize()
        helper_functions._record_timing(timing_stats, "loop_turbulence_upload", perf_counter() - section_start)

    section_start = perf_counter()
    update_force_fields[blockspergrid_3d, kernel_config.THREADS_PER_BLOCK_3D](
        gpu_fields["Fx_base"], gpu_fields["Fy_base"], gpu_fields["Fz_base"],
        gpu_fields["turbulence_Fx_a"], gpu_fields["turbulence_Fy_a"], gpu_fields["turbulence_Fz_a"],
        gpu_fields["turbulence_Fx_b"], gpu_fields["turbulence_Fy_b"], gpu_fields["turbulence_Fz_b"],
        gpu_fields["turbulence_cos_coeffs"], gpu_fields["turbulence_sin_coeffs"],
        turbulence_count,
        np.float32(animated_force["x"]),
        np.float32(animated_force["y"]),
        np.float32(animated_force["z"]),
        gpu_fields["Fx"], gpu_fields["Fy"], gpu_fields["Fz"]
    )
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "loop_force_fields", perf_counter() - section_start)

    section_start = perf_counter()
    buoyancy_approximation[blockspergrid_3d, kernel_config.THREADS_PER_BLOCK_3D](
        T, gpu_fields["Fz"], gpu_constants["BUOANCY_FACTOR"], gpu_constants["T_REFERENCE"]
    )
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "loop_buoyancy", perf_counter() - section_start)

    #------------Pressure solve-------------------
    section_start = perf_counter()
    p = pressure_poisson(
        u, v, w, p, T, gpu_fields["obstacle_mask"], gpu_fields["pressure_rhs"],
        gpu_fields["vorticity_x"], gpu_fields["vorticity_y"], gpu_fields["vorticity_z"], gpu_fields["vorticity_magnitude"], dt, gpu_fields["point_divergence"],
        gpu_constants["DELTA"], gpu_constants["RHO"], gpu_constants["EXPANSION_RATE"],
        gpu_constants["T_REFERENCE"], simulation_params["MAX_ITER"]
    )
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "loop_pressure", perf_counter() - section_start)

    if gpu_constants["VORTICITY"] > 0.0:
        section_start = perf_counter()
        apply_vorticity_confinement[blockspergrid_3d, kernel_config.THREADS_PER_BLOCK_3D](
            gpu_fields["obstacle_mask"],
            gpu_fields["vorticity_x"],
            gpu_fields["vorticity_y"],
            gpu_fields["vorticity_z"],
            gpu_fields["vorticity_magnitude"],
            gpu_fields["Fx"],
            gpu_fields["Fy"],
            gpu_fields["Fz"],
            gpu_constants["DELTA"],
            gpu_constants["VORTICITY"]
        )
        cuda.synchronize()
        helper_functions._record_timing(timing_stats, "loop_vorticity", perf_counter() - section_start)

    #------------Velocity update-------------------
    section_start = perf_counter()
    if simulation_params["VELOCITY_ADVECTION_SCHEME"] == "FIRST_ORDER_UPWIND":
        update_velocity[blockspergrid_3d, kernel_config.THREADS_PER_BLOCK_3D](
            u, v, w, p, dt, gpu_fields["Fx"], gpu_fields["Fy"], gpu_fields["Fz"], u_work, v_work, w_work,
            gpu_constants["DELTA"], gpu_constants["RHO"], gpu_constants["NU"],
            simulation_params["MAX_VELOCITY_INCREMENT_FACTOR"],
        )
    else:
        update_velocity_second_order_upwind[blockspergrid_3d, kernel_config.THREADS_PER_BLOCK_3D](
            u, v, w, p, dt, gpu_fields["Fx"], gpu_fields["Fy"], gpu_fields["Fz"], u_work, v_work, w_work,
            gpu_constants["DELTA"], gpu_constants["RHO"], gpu_constants["NU"],
            simulation_params["MAX_VELOCITY_INCREMENT_FACTOR"],
        )
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "loop_velocity", perf_counter() - section_start)

    u, u_work = u_work, u
    v, v_work = v_work, v
    w, w_work = w_work, w

    #------------Scalar update-------------------
    section_start = perf_counter()
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
        gpu_constants["MINIMUM_OXYGEN_CONCENTRATION"],
        gpu_constants["T_REFERENCE"],
    )
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "loop_scalars", perf_counter() - section_start)

    T, temperature_work = temperature_work, T
    smoke, smoke_work = smoke_work, smoke
    fuel, fuel_work = fuel_work, fuel
    flame, flame_work = flame_work, flame

    #------------Boundary conditions-------------------
    section_start = perf_counter()
    u, v, w, p, T, smoke, fuel, flame = apply_all_BC(
        u, v, w, p, T, smoke, fuel, flame,
        simulation_params["BC_CONFIG"],
        gpu_constants["HAS_OBSTACLE"],
        gpu_fields["obstacle_mask"],
        gpu_fields["obstacle_velocity_x"],
        gpu_fields["obstacle_velocity_y"],
        gpu_fields["obstacle_velocity_z"],
        gpu_constants["HAS_SOURCE"],
        gpu_fields["source_mask"],
        gpu_fields["source_velocity_mask"],
        gpu_fields["source_temperature"],
        gpu_fields["source_smoke"],
        gpu_fields["source_fuel"],
        gpu_fields["source_velocity_x"],
        gpu_fields["source_velocity_y"],
        gpu_fields["source_velocity_z"],
    )
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "loop_apply_bc", perf_counter() - section_start)

    #------------Publish current fields-------------------
    device_fields["u"] = u
    device_fields["v"] = v
    device_fields["w"] = w
    device_fields["p"] = p
    device_fields["T"] = T
    device_fields["smoke"] = smoke
    device_fields["fuel"] = fuel
    device_fields["flame"] = flame

    state.update({
        "u": u,
        "v": v,
        "w": w,
        "u_work": u_work,
        "v_work": v_work,
        "w_work": w_work,
        "p": p,
        "T": T,
        "temperature_work": temperature_work,
        "smoke": smoke,
        "smoke_work": smoke_work,
        "fuel": fuel,
        "fuel_work": fuel_work,
        "flame": flame,
        "flame_work": flame_work,
    })
    return state


# ===============================
# Main
# ===============================

def main(config=None):
    """
    Run the complete GPU fluid simulation from setup to shutdown.

    main() orchestrates solver initialisation, output preparation, repeated
    timestep execution, adaptive timestep updates, progress reporting, output
    shutdown and final timing and memory summaries.

    Args:
        config (dict | None): optional simulation override configuration
    """
    #------------Initialise-------------------
    state = _initialise_solver(config)

    if state["solver_diverged"]:
        print('ERROR: The solver diverged before output setup, stopping the simulation!')
    else:
        #------------Prepare output-------------------
        section_start = perf_counter()
        write_queue, buffer_pool, writer_threads, shared_memory_blocks = output_functions.setup_output(
            state["simulation_params"]["OUTPATH"],
            state["simulation_params"]["FRAME_START"],
            state["simulation_params"]["OUTPUT_BUFFER_VARIABLES"],
            helper_functions.select_fields(
                state["device_fields"],
                state["simulation_params"]["OUTPUT_BUFFER_VARIABLES"],
            ),
            state["simulation_params"]["WRITE_QUEUE_SIZE"],
            state["simulation_params"]["OUTPUT_FORWARDER_COUNT"],
            state["simulation_params"]["DELTA"],
            state["simulation_params"]["HOST_VDB_WRITER"],
            storage_dtype=state["simulation_params"]["OUTPUT_DTYPE"],
        )
        state["write_queue"] = write_queue
        state["writer_threads"] = writer_threads
        state["shared_memory_blocks"] = shared_memory_blocks
        helper_functions._record_timing(state["timing_stats"], "init_output_setup", perf_counter() - section_start)

        print('Start time iteration')
        helper_functions.emit_progress(0.0, state["t"])
        blockspergrid_3d = kernel_config.volume_blocks_per_grid(
            state["u"].shape,
            kernel_config.THREADS_PER_BLOCK_3D,
        )

        section_start = perf_counter()
        state["memory_tracker"].sample()
        helper_functions._record_timing(state["timing_stats"], "loop_memory_sample", perf_counter() - section_start)

        #------------Time iteration-------------------
        next_output_time = 0.0
        output_index = 0

        while state["t"] < state["simulation_params"]["T_MAX"]:
            step_start = perf_counter()
            if state["cancel_flag_path"] and Path(state["cancel_flag_path"]).exists():
                state["cancel_requested"] = True
                print('Bake cancellation requested. Stopping the simulation cleanly...')
                break

            #------------One solver step-------------------
            state = _run_time_step(state, blockspergrid_3d)

            #------------Output-------------------
            section_start = perf_counter()
            while state["t"] >= next_output_time:
                output_functions.enqueue_device_output(
                    write_queue,
                    buffer_pool,
                    state["simulation_params"]["OUTPUT_BUFFER_VARIABLES"],
                    helper_functions.select_fields(
                        state["device_fields"],
                        state["simulation_params"]["OUTPUT_BUFFER_VARIABLES"],
                    ),
                    output_index,
                    state["t"],
                    state["simulation_params"]["OUTPUT_FIELD_CONFIG"],
                )

                output_index += 1
                state["output_frame_count"] += 1
                next_output_time += state["simulation_params"]["OUTPUT_TIME_STEP"]
                helper_functions.emit_progress(
                    state["t"] / state["simulation_params"]["T_MAX"] * 100.0,
                    state["t"],
                )

                if state["simulation_params"]["OUTPUT_STATUS"]:
                    print('#################################################')
                    print(f'Simulation time {state["t"]} sec')
                    print(f'Current dt: {np.round(state["dt"], 5)}')
            helper_functions._record_timing(state["timing_stats"], "loop_output", perf_counter() - section_start)

            #------------Adaptive timestep-------------------
            state["t"] += state["dt"]

            section_start = perf_counter()
            time_step.reset_velocity_maxima(
                state["gpu_fields"]["velocity_maxima"],
                state["velocity_maxima_host_zeros"],
            )
            dt_new, state["solver_diverged"] = time_step.compute_new_timestep_gpu(
                state["u"],
                state["v"],
                state["w"],
                state["gpu_fields"]["velocity_maxima"],
                state["fx_max"],
                state["fy_max"],
                state["fz_max"],
                state["gpu_constants"]["RHO"],
                state["gpu_constants"]["DELTA"],
                state["gpu_constants"]["NU"],
                state["simulation_params"]["CFL_MAX"],
                max_dt=1.0 / state["simulation_params"]["OUTPUT_FPS"],
            )
            cuda.synchronize()
            helper_functions._record_timing(state["timing_stats"], "loop_timestep", perf_counter() - section_start)

            section_start = perf_counter()
            state["memory_tracker"].sample()
            helper_functions._record_timing(state["timing_stats"], "loop_memory_sample", perf_counter() - section_start)

            if state["solver_diverged"]:
                print('ERROR: The solver diverged, stopping the simulation!')
                break

            state["dt"] = dt_new
            state["step_count"] += 1
            helper_functions._record_timing(state["timing_stats"], "loop_total", perf_counter() - step_start)

    #------------Shutdown output-------------------
    if state["write_queue"] is not None:
        section_start = perf_counter()
        output_functions.shutdown_output(
            state["write_queue"],
            state["writer_threads"],
            state["shared_memory_blocks"],
        )
        helper_functions._record_timing(state["timing_stats"], "shutdown_output", perf_counter() - section_start)

    #------------Conclusion-------------------
    if state["solver_diverged"]:
        print('Simulation stopped after solver divergence.')
    elif state["cancel_requested"]:
        print('Simulation cancelled after clean shutdown.')
    else:
        helper_functions.emit_progress(100.0, state["simulation_params"]["T_MAX"])
        print('Simulation finished!')

    section_start = perf_counter()
    state["memory_tracker"].sample()
    helper_functions._record_timing(state["timing_stats"], "final_memory_sample", perf_counter() - section_start)
    state["memory_tracker"].print_summary()

    total_runtime = perf_counter() - state["total_start_time"]
    helper_functions._print_timing_summary(
        state["timing_stats"],
        total_runtime,
        state["step_count"],
        state["output_frame_count"],
    )
    print(f'Solver runtime: {total_runtime:.3f} s')
    print("################################################################")
