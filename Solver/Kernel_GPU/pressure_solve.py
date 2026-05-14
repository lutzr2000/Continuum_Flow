import math

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
def _pressure_poisson_fix_reference_pressure(p, ref_i, ref_j, ref_k):
    """Pin one interior pressure cell to zero to remove the Neumann nullspace."""
    if cuda.grid(1) != 0:
        return
    p[ref_i, ref_j, ref_k] = 0.0


def pressure_poisson(
    u, v, w, p, T, obstacle_mask, b, omega_x, omega_y, omega_z, omega_magnitude,
    dt, point_divergence, delta, rho, expansion_rate, t_reference,
    max_iter=10, threadsperblock_3d=None, relaxation_factor=1.7
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
    nx, ny, nz = p.shape
    ref_i = min(max(nx // 2, 1), nx - 2)
    ref_j = min(max(ny // 2, 1), ny - 2)
    ref_k = min(max(nz // 2, 1), nz - 2)

    pressure_equation_right_side[blockspergrid_3d, threadsperblock_3d](
        u, v, w, T, obstacle_mask, b, omega_x, omega_y, omega_z, omega_magnitude,
        dt, point_divergence, delta, rho, expansion_rate, t_reference
    )
    p_old = p

    for _ in range(max_iter):
        _pressure_poisson_red_black_sor_step[blockspergrid_3d, threadsperblock_3d](
            p_old, b, delta, 0, relaxation_factor
        )
        BC._pressure_poisson_apply_neumann_bcs[blockspergrid_3d, threadsperblock_3d](p_old)
        _pressure_poisson_fix_reference_pressure[1, 1](p_old, ref_i, ref_j, ref_k)
        _pressure_poisson_red_black_sor_step[blockspergrid_3d, threadsperblock_3d](
            p_old, b, delta, 1, relaxation_factor
        )
        BC._pressure_poisson_apply_neumann_bcs[blockspergrid_3d, threadsperblock_3d](p_old)
        _pressure_poisson_fix_reference_pressure[1, 1](p_old, ref_i, ref_j, ref_k)
        
    BC._pressure_poisson_apply_neumann_bcs[blockspergrid_3d, threadsperblock_3d](p_old)
    _pressure_poisson_fix_reference_pressure[1, 1](p_old, ref_i, ref_j, ref_k)
    return p_old
