from numba import cuda

import Solver.Kernel_GPU.sparse_managment as sparse_managment
import Solver.Kernel_GPU.advection_schemes as advection_schemes
import Solver.Kernel_GPU.forces as forces
from Solver.Kernel_GPU.vorticity import apply_vorticity_confinement


@cuda.jit(cache=True)
def advect_velocity_semi_lagrangian(
    u,
    v,
    w,
    advected_u,
    advected_v,
    advected_w,
    dt,
    delta,
    tile_map,
    u_initial,
    v_initial,
    w_initial,
    nx,
    ny,
    nz,
):
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
    u,
    v,
    w,
    obstacle_mask,
    predictor_u,
    predictor_v,
    predictor_w,
    dt,
    un,
    vn,
    wn,
    delta,
    rho,
    nu,
    vorticity_magnitude,
    vorticity_strength,
    temperature,
    buoyancy_factor,
    t_reference,
    tile_map,
    fx_const,
    fy_const,
    fz_const,
    has_swirl_nodes,
    swirl_config,
    origin_x,
    origin_y,
    origin_z,
    has_turbulence_nodes,
    turbulence_config,
    t,
    u_initial,
    v_initial,
    w_initial,
    nx,
    ny,
    nz,
):
    """
    CUDA kernel that updates velocity with a MacCormack-corrected
    semi-Lagrangian advection step on sparse velocity pools.
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

    Fx = 0.0
    Fy = 0.0
    Fz = 0.0

    dt_over_delta = dt / delta
    diffusion_coeff = nu * dt / (delta * delta)
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
        nx,
        ny,
        nz,
        u_initial,
        v_initial,
        w_initial,
    )

    advected_u = predictor_u[tile_index, local_i, local_j, local_k]
    advected_v = predictor_v[tile_index, local_i, local_j, local_k]
    advected_w = predictor_w[tile_index, local_i, local_j, local_k]

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

    x0, y0, z0, x1, y1, z1, _, _, _ = advection_schemes._prepare_trilinear_coords(
        x_depart, y_depart, z_depart, nx, ny, nz
    )

    u_lower, u_upper = advection_schemes._sample_cell_extrema_inner_sparse(
        u, tile_map, x0, y0, z0, x1, y1, z1, u_initial
    )
    v_lower, v_upper = advection_schemes._sample_cell_extrema_inner_sparse(
        v, tile_map, x0, y0, z0, x1, y1, z1, v_initial
    )
    w_lower, w_upper = advection_schemes._sample_cell_extrema_inner_sparse(
        w, tile_map, x0, y0, z0, x1, y1, z1, w_initial
    )

    corrected_u = advection_schemes._clamp(corrected_u, u_lower, u_upper)
    corrected_v = advection_schemes._clamp(corrected_v, v_lower, v_upper)
    corrected_w = advection_schemes._clamp(corrected_w, w_lower, w_upper)

    diffusion_x = diffusion_coeff * (
        (
            sparse_managment._sample_sparse_cell(u, tile_map, i + 1, j, k, u_initial)
            - 2.0 * u_center
            + sparse_managment._sample_sparse_cell(u, tile_map, i - 1, j, k, u_initial)
        )
        + (
            sparse_managment._sample_sparse_cell(u, tile_map, i, j + 1, k, u_initial)
            - 2.0 * u_center
            + sparse_managment._sample_sparse_cell(u, tile_map, i, j - 1, k, u_initial)
        )
        + (
            sparse_managment._sample_sparse_cell(u, tile_map, i, j, k + 1, u_initial)
            - 2.0 * u_center
            + sparse_managment._sample_sparse_cell(u, tile_map, i, j, k - 1, u_initial)
        )
    )
    diffusion_y = diffusion_coeff * (
        (
            sparse_managment._sample_sparse_cell(v, tile_map, i + 1, j, k, v_initial)
            - 2.0 * v_center
            + sparse_managment._sample_sparse_cell(v, tile_map, i - 1, j, k, v_initial)
        )
        + (
            sparse_managment._sample_sparse_cell(v, tile_map, i, j + 1, k, v_initial)
            - 2.0 * v_center
            + sparse_managment._sample_sparse_cell(v, tile_map, i, j - 1, k, v_initial)
        )
        + (
            sparse_managment._sample_sparse_cell(v, tile_map, i, j, k + 1, v_initial)
            - 2.0 * v_center
            + sparse_managment._sample_sparse_cell(v, tile_map, i, j, k - 1, v_initial)
        )
    )
    diffusion_z = diffusion_coeff * (
        (
            sparse_managment._sample_sparse_cell(w, tile_map, i + 1, j, k, w_initial)
            - 2.0 * w_center
            + sparse_managment._sample_sparse_cell(w, tile_map, i - 1, j, k, w_initial)
        )
        + (
            sparse_managment._sample_sparse_cell(w, tile_map, i, j + 1, k, w_initial)
            - 2.0 * w_center
            + sparse_managment._sample_sparse_cell(w, tile_map, i, j - 1, k, w_initial)
        )
        + (
            sparse_managment._sample_sparse_cell(w, tile_map, i, j, k + 1, w_initial)
            - 2.0 * w_center
            + sparse_managment._sample_sparse_cell(w, tile_map, i, j, k - 1, w_initial)
        )
    )

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

    Fz += forces.buoyancy_approximation(
        temperature,
        tile_map,
        i,
        j,
        k,
        buoyancy_factor,
        t_reference,
    )

    u_raw = corrected_u + diffusion_x + force_coeff * Fx
    v_raw = corrected_v + diffusion_y + force_coeff * Fy
    w_raw = corrected_w + diffusion_z + force_coeff * Fz

    un[tile_index, local_i, local_j, local_k] = u_raw
    vn[tile_index, local_i, local_j, local_k] = v_raw
    wn[tile_index, local_i, local_j, local_k] = w_raw
