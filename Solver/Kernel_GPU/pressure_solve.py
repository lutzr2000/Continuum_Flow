import math

import numpy as np
from numba import cuda

import Solver.Kernel_GPU.Boundary_Conditions.domain_bc as BC
import Solver.Kernel_GPU.kernel_config as kernel_config

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
        b[i, j, k] = 0.0
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

    #------------Artifical thermal divergence-------------------
    thermal_divergence = expansion_rate * (T[i, j, k] - t_reference)
    authored_divergence = point_divergence[i, j, k]

    #------------Right hand side-------------------
    b[i, j, k] = rho_over_dt * (divergence + authored_divergence - thermal_divergence)

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
def _sum_rhs_kernel(b, rhs_sum):
    """
    Accumulate the zero-boundary RHS into one scalar sum on the GPU.

    This cached kernel is intentionally simple and avoids the more complex
    reduction pattern that triggered CUDA code-library serialization issues.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = b.shape

    if i >= nx or j >= ny or k >= nz:
        return

    cuda.atomic.add(rhs_sum, 0, b[i, j, k])


@cuda.jit(cache=True)
def _subtract_rhs_mean_kernel(b, rhs_mean):
    """Subtract the interior RHS mean from interior cells only."""
    i, j, k = cuda.grid(3)
    nx, ny, nz = b.shape

    if (
        i < 1 or j < 1 or k < 1 or
        i >= nx - 1 or j >= ny - 1 or k >= nz - 1
    ):
        return

    b[i, j, k] -= rhs_mean


def _remove_rhs_mean(b, threadsperblock_3d):
    """
    Enforce the Neumann compatibility condition by removing the RHS mean.

    The expensive sum is reduced on the GPU into a small partial buffer; only
    those block sums are transferred to the host for the final scalar sum.
    """
    nx, ny, nz = b.shape
    interior_cell_count = max((nx - 2) * (ny - 2) * (nz - 2), 1)
    rhs_sum = cuda.to_device(np.zeros(1, dtype=np.float32))
    blockspergrid_3d = kernel_config.volume_blocks_per_grid(b.shape, threadsperblock_3d)

    _sum_rhs_kernel[blockspergrid_3d, threadsperblock_3d](b, rhs_sum)

    rhs_mean = float(rhs_sum.copy_to_host()[0]) / float(interior_cell_count)
    if abs(rhs_mean) <= 1.0e-12:
        return

    _subtract_rhs_mean_kernel[blockspergrid_3d, threadsperblock_3d](b, rhs_mean)


@cuda.jit(cache=True)
def _pressure_poisson_red_black_sor_step(p, b, delta, parity, relaxation_factor):
    """
    Perform one in-place red-black SOR color pass of the 3D pressure Poisson
    equation on the GPU.

    Only interior cells whose `(i + j + k) % 2` matches `parity` are updated in
    this launch. Each update first computes the red-black Gauss-Seidel value and
    then applies the SOR relaxation increment in place.

    Args:
        p (device array): pressure field updated in place
        b (device array): right hand side of the pressure Poisson equation
        delta (float): grid spacing
        parity (int): `0` for red cells, `1` for black cells
        relaxation_factor (float): SOR relaxation factor, typically in `(1, 2)`
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
    gauss_seidel_value = (
        p[i + 1, j, k] + p[i - 1, j, k] +
        p[i, j + 1, k] + p[i, j - 1, k] +
        p[i, j, k + 1] + p[i, j, k - 1] -
        delta2 * b[i, j, k]
    ) / 6.0
    p_old = p[i, j, k]
    p[i, j, k] = p_old + relaxation_factor * (gauss_seidel_value - p_old)


@cuda.jit(cache=True)
def _project_velocity_kernel(u, v, w, p, obstacle_mask, dt, delta, rho):
    """
    Apply the pressure projection `u <- u - dt/rho * grad(p)` to one interior cell.

    Obstacle cells are skipped because their wall velocities are restored by the
    obstacle boundary conditions after the projection pass.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape

    if (
        i < 1 or j < 1 or k < 1 or
        i >= nx - 1 or j >= ny - 1 or k >= nz - 1
    ):
        return

    if obstacle_mask[i, j, k]:
        return

    pressure_coeff = dt / (2.0 * rho * delta)
    u[i, j, k] -= pressure_coeff * (p[i + 1, j, k] - p[i - 1, j, k])
    v[i, j, k] -= pressure_coeff * (p[i, j + 1, k] - p[i, j - 1, k])
    w[i, j, k] -= pressure_coeff * (p[i, j, k + 1] - p[i, j, k - 1])


def pressure_poisson(
    u, v, w, p, T, obstacle_mask, b, omega_x, omega_y, omega_z, omega_magnitude,
    dt, point_divergence, delta, rho, expansion_rate, t_reference,
    max_iter=10, threadsperblock_3d=None, relaxation_factor=1
):
    """
    Host-side pressure Poisson solve that launches CUDA kernels for the RHS,
    red-black SOR iterations and Neumann boundary conditions.

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
        relaxation_factor (float): SOR relaxation factor used on the GPU
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
    _remove_rhs_mean(b, threadsperblock_3d)
    p_old = p

    for _ in range(max_iter):
        _pressure_poisson_red_black_sor_step[blockspergrid_3d, threadsperblock_3d](
            p_old, b, delta, 0, relaxation_factor
        )
        BC._pressure_poisson_apply_neumann_bcs[blockspergrid_3d, threadsperblock_3d](p_old)
        _pressure_poisson_red_black_sor_step[blockspergrid_3d, threadsperblock_3d](
            p_old, b, delta, 1, relaxation_factor
        )
        BC._pressure_poisson_apply_neumann_bcs[blockspergrid_3d, threadsperblock_3d](p_old)
        
    BC._pressure_poisson_apply_neumann_bcs[blockspergrid_3d, threadsperblock_3d](p_old)
    return p_old


def project_velocity(u, v, w, p, obstacle_mask, dt, delta, rho, threadsperblock_3d=None):
    """Project one intermediate velocity field with the solved pressure."""
    if threadsperblock_3d is None:
        threadsperblock_3d = kernel_config.THREADS_PER_BLOCK_3D
    blockspergrid_3d = kernel_config.volume_blocks_per_grid(u.shape, threadsperblock_3d)

    _project_velocity_kernel[blockspergrid_3d, threadsperblock_3d](
        u, v, w, p, obstacle_mask, dt, delta, rho
    )
    return u, v, w
