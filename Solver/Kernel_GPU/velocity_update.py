from typing import Any

from numba import cuda

import Solver.Kernel_GPU.sparse_managment as sparse_managment
import Solver.Kernel_GPU.advection_schemes as advection_schemes
import Solver.Kernel_GPU.forces as forces
from Solver.Kernel_GPU.vorticity import apply_vorticity_confinement


@cuda.jit(cache=True)
def advect_velocity_semi_lagrangian(
    u: Any,
    v: Any,
    w: Any,
    advected_u: Any,
    advected_v: Any,
    advected_w: Any,
    dt: float,
    delta: float,
    n_substeps: int,
    tile_map: Any,
    u_initial: float,
    v_initial: float,
    w_initial: float,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    r"""
    Build the semi-Lagrangian predictor for all three velocity components.

    .. math::

        \mathbf{u}^{*}(\mathbf{x}) =
        \mathbf{u}^{n}\!\left(
            \mathbf{x} - \Delta t\,\mathbf{u}^{n}(\mathbf{x})
        \right).
    """
    (
        tile_i,
        tile_j,
        tile_k,
        local_i,
        local_j,
        local_k,
        i,
        j,
        k,
    ) = sparse_managment.tile_to_index()

    tile_index = tile_map[tile_i, tile_j, tile_k]
    if tile_index == -1:
        return

    x_depart, y_depart, z_depart = advection_schemes._backtrace_position_sparse(
        u,
        v,
        w,
        tile_map,
        float(i),
        float(j),
        float(k),
        dt / delta,
        n_substeps,
        nx,
        ny,
        nz,
        u_initial,
        v_initial,
        w_initial,
    )

    sampled_u, sampled_v, sampled_w = advection_schemes._sample_trilinear_vec3_sparse(
        u,
        v,
        w,
        tile_map,
        x_depart,
        y_depart,
        z_depart,
        nx,
        ny,
        nz,
        u_initial,
        v_initial,
        w_initial,
    )

    advected_u[tile_index, local_i, local_j, local_k] = sampled_u
    advected_v[tile_index, local_i, local_j, local_k] = sampled_v
    advected_w[tile_index, local_i, local_j, local_k] = sampled_w


@cuda.jit(cache=True)
def update_velocity_maccormack(
    u: Any,
    v: Any,
    w: Any,
    obstacle_mask: Any,
    predictor_u: Any,
    predictor_v: Any,
    predictor_w: Any,
    dt: float,
    un: Any,
    vn: Any,
    wn: Any,
    delta: float,
    rho: float,
    n_substeps: int,
    nu: float,
    vorticity_magnitude: Any,
    vorticity_strength: float,
    temperature: Any,
    buoyancy_factor: float,
    t_reference: float,
    gravity_x: float,
    gravity_y: float,
    gravity_z: float,
    tile_map: Any,
    fx_const: Any,
    fy_const: Any,
    fz_const: Any,
    has_swirl_nodes: bool,
    swirl_config: Any,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    has_turbulence_nodes: bool,
    turbulence_config: Any,
    t: float,
    u_initial: float,
    v_initial: float,
    w_initial: float,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    r"""
    CUDA kernel that updates velocity with a MacCormack-corrected
    semi-Lagrangian advection step on sparse velocity pools.

    Advection:
        MacCormack-corrected semi-Lagrangian.

    Forces:
        Vorticity confinement, swirl, turbulence, constant forces
        and buoyancy.

    Diffusion:
        Uses one local backward-Euler/Jacobi-style update

            (I - nu * dt * Laplacian) u_new = rhs

        approximated using the velocity field from the beginning of
        the timestep for the neighbouring values.

        This removes the explicit diffusion CFL restriction, but is
        not equivalent to a fully converged implicit diffusion solve.
    """

    (
        tile_i,
        tile_j,
        tile_k,
        local_i,
        local_j,
        local_k,
        i,
        j,
        k,
    ) = sparse_managment.tile_to_index()

    tile_index = tile_map[tile_i, tile_j, tile_k]

    if tile_index == -1:
        return

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    dt_over_delta = dt / delta
    diffusion_alpha = nu * dt / (delta * delta)
    diffusion_inv_diag = 1.0 / (1.0 + 6.0 * diffusion_alpha)
    force_coeff = dt / rho

    u_center = u[tile_index, local_i, local_j, local_k]
    v_center = v[tile_index, local_i, local_j, local_k]
    w_center = w[tile_index, local_i, local_j, local_k]

    x_depart, y_depart, z_depart = advection_schemes._backtrace_position_sparse(
        u,
        v,
        w,
        tile_map,
        float(i),
        float(j),
        float(k),
        dt_over_delta,
        n_substeps,
        nx,
        ny,
        nz,
        u_initial,
        v_initial,
        w_initial,
    )

    x_forward, y_forward, z_forward = advection_schemes._forward_trace_position_sparse(
        u,
        v,
        w,
        tile_map,
        x_depart,
        y_depart,
        z_depart,
        dt_over_delta,
        n_substeps,
        nx,
        ny,
        nz,
        u_initial,
        v_initial,
        w_initial,
    )

    advected_u = predictor_u[
        tile_index,
        local_i,
        local_j,
        local_k,
    ]

    advected_v = predictor_v[
        tile_index,
        local_i,
        local_j,
        local_k,
    ]

    advected_w = predictor_w[
        tile_index,
        local_i,
        local_j,
        local_k,
    ]

    reverse_u, reverse_v, reverse_w = advection_schemes._sample_trilinear_vec3_sparse(
        predictor_u,
        predictor_v,
        predictor_w,
        tile_map,
        x_forward,
        y_forward,
        z_forward,
        nx,
        ny,
        nz,
        u_initial,
        v_initial,
        w_initial,
    )

    corrected_u = advected_u + 0.5 * (u_center - reverse_u)
    corrected_v = advected_v + 0.5 * (v_center - reverse_v)
    corrected_w = advected_w + 0.5 * (w_center - reverse_w)

    (
        x0,
        y0,
        z0,
        x1,
        y1,
        z1,
        _,
        _,
        _,
    ) = advection_schemes._prepare_trilinear_coords(
        x_depart,
        y_depart,
        z_depart,
        nx,
        ny,
        nz,
    )

    u_lower, u_upper = advection_schemes._sample_cell_extrema_inner_sparse(
        u,
        tile_map,
        x0,
        y0,
        z0,
        x1,
        y1,
        z1,
        u_initial,
    )

    v_lower, v_upper = advection_schemes._sample_cell_extrema_inner_sparse(
        v,
        tile_map,
        x0,
        y0,
        z0,
        x1,
        y1,
        z1,
        v_initial,
    )

    w_lower, w_upper = advection_schemes._sample_cell_extrema_inner_sparse(
        w,
        tile_map,
        x0,
        y0,
        z0,
        x1,
        y1,
        z1,
        w_initial,
    )

    corrected_u = advection_schemes._clamp(
        corrected_u,
        u_lower,
        u_upper,
    )

    corrected_v = advection_schemes._clamp(
        corrected_v,
        v_lower,
        v_upper,
    )

    corrected_w = advection_schemes._clamp(
        corrected_w,
        w_lower,
        w_upper,
    )

    Fx = 0.0
    Fy = 0.0
    Fz = 0.0

    if vorticity_strength > 0.0:
        Fx, Fy, Fz = apply_vorticity_confinement(
            u,
            v,
            w,
            obstacle_mask,
            vorticity_magnitude,
            i,
            j,
            k,
            delta,
            vorticity_strength,
            tile_map,
            u_initial,
            v_initial,
            w_initial,
            nx,
            ny,
            nz,
        )

    if has_swirl_nodes:
        swirl_fx, swirl_fy, swirl_fz = forces.apply_swirl_forces(
            swirl_config,
            i,
            j,
            k,
            delta,
            origin_x,
            origin_y,
            origin_z,
        )

        Fx += swirl_fx
        Fy += swirl_fy
        Fz += swirl_fz

    if has_turbulence_nodes:
        turb_fx, turb_fy, turb_fz = forces.apply_turbulence_forces(
            turbulence_config,
            i,
            j,
            k,
            delta,
            origin_x,
            origin_y,
            origin_z,
            t,
        )

        Fx += turb_fx
        Fy += turb_fy
        Fz += turb_fz

    Fx += fx_const * 0.1
    Fy += fy_const * 0.1
    Fz += fz_const * 0.1

    buoyancy = forces.buoyancy_approximation(
        temperature,
        tile_map,
        i,
        j,
        k,
        buoyancy_factor,
        t_reference,
    )
    Fx += gravity_x * buoyancy
    Fy += gravity_y * buoyancy
    Fz += gravity_z * buoyancy

    rhs_u = corrected_u + force_coeff * Fx
    rhs_v = corrected_v + force_coeff * Fy
    rhs_w = corrected_w + force_coeff * Fz

    u_xp = sparse_managment.get_pool_value(
        u,
        tile_map,
        i + 1,
        j,
        k,
        u_initial,
    )

    u_xm = sparse_managment.get_pool_value(
        u,
        tile_map,
        i - 1,
        j,
        k,
        u_initial,
    )

    u_yp = sparse_managment.get_pool_value(
        u,
        tile_map,
        i,
        j + 1,
        k,
        u_initial,
    )

    u_ym = sparse_managment.get_pool_value(
        u,
        tile_map,
        i,
        j - 1,
        k,
        u_initial,
    )

    u_zp = sparse_managment.get_pool_value(
        u,
        tile_map,
        i,
        j,
        k + 1,
        u_initial,
    )

    u_zm = sparse_managment.get_pool_value(
        u,
        tile_map,
        i,
        j,
        k - 1,
        u_initial,
    )

    v_xp = sparse_managment.get_pool_value(
        v,
        tile_map,
        i + 1,
        j,
        k,
        v_initial,
    )

    v_xm = sparse_managment.get_pool_value(
        v,
        tile_map,
        i - 1,
        j,
        k,
        v_initial,
    )

    v_yp = sparse_managment.get_pool_value(
        v,
        tile_map,
        i,
        j + 1,
        k,
        v_initial,
    )

    v_ym = sparse_managment.get_pool_value(
        v,
        tile_map,
        i,
        j - 1,
        k,
        v_initial,
    )

    v_zp = sparse_managment.get_pool_value(
        v,
        tile_map,
        i,
        j,
        k + 1,
        v_initial,
    )

    v_zm = sparse_managment.get_pool_value(
        v,
        tile_map,
        i,
        j,
        k - 1,
        v_initial,
    )

    w_xp = sparse_managment.get_pool_value(
        w,
        tile_map,
        i + 1,
        j,
        k,
        w_initial,
    )

    w_xm = sparse_managment.get_pool_value(
        w,
        tile_map,
        i - 1,
        j,
        k,
        w_initial,
    )

    w_yp = sparse_managment.get_pool_value(
        w,
        tile_map,
        i,
        j + 1,
        k,
        w_initial,
    )

    w_ym = sparse_managment.get_pool_value(
        w,
        tile_map,
        i,
        j - 1,
        k,
        w_initial,
    )

    w_zp = sparse_managment.get_pool_value(
        w,
        tile_map,
        i,
        j,
        k + 1,
        w_initial,
    )

    w_zm = sparse_managment.get_pool_value(
        w,
        tile_map,
        i,
        j,
        k - 1,
        w_initial,
    )

    u_neighbor_sum = u_xp + u_xm + u_yp + u_ym + u_zp + u_zm
    v_neighbor_sum = v_xp + v_xm + v_yp + v_ym + v_zp + v_zm
    w_neighbor_sum = w_xp + w_xm + w_yp + w_ym + w_zp + w_zm

    u_raw = (rhs_u + diffusion_alpha * u_neighbor_sum) * diffusion_inv_diag
    v_raw = (rhs_v + diffusion_alpha * v_neighbor_sum) * diffusion_inv_diag
    w_raw = (rhs_w + diffusion_alpha * w_neighbor_sum) * diffusion_inv_diag

    un[
        tile_index,
        local_i,
        local_j,
        local_k,
    ] = u_raw

    vn[
        tile_index,
        local_i,
        local_j,
        local_k,
    ] = v_raw

    wn[
        tile_index,
        local_i,
        local_j,
        local_k,
    ] = w_raw
