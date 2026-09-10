from typing import Any

from numba import njit, prange

import Solver.Kernel_CPU.sparse_managment as sparse_managment
import Solver.Kernel_CPU.advection_schemes as advection_schemes
import Solver.Kernel_CPU.forces as forces
import Solver.Kernel_CPU.kernel_config as kernel_config
from Solver.Kernel_CPU.vorticity import apply_vorticity_confinement


@njit(cache=True, parallel=True)
def advect_velocity_semi_lagrangian(
    active_tile_coords,
    active_tile_slots,
    active_tile_count,
    u: Any,
    v: Any,
    w: Any,
    advected_u: Any,
    advected_v: Any,
    advected_w: Any,
    dt: float,
    delta: float,
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
    total_tiles = active_tile_count

    for active_tile_n in prange(total_tiles):
        tile_i = active_tile_coords[active_tile_n, 0]
        tile_j = active_tile_coords[active_tile_n, 1]
        tile_k = active_tile_coords[active_tile_n, 2]
        tile_flat = (tile_i * tile_map.shape[1] + tile_j) * tile_map.shape[2] + tile_k
        for local_i in range(kernel_config.TILE_SIZE):
            for local_j in range(kernel_config.TILE_SIZE):
                for local_k in range(kernel_config.TILE_SIZE):
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
                    ) = sparse_managment.tile_to_index(
                        tile_flat,
                        local_i,
                        local_j,
                        local_k,
                        tile_map.shape[0],
                        tile_map.shape[1],
                        tile_map.shape[2],
                    )

                    tile_index = active_tile_slots[active_tile_n]
                    if tile_index == -1:
                        continue

                    x_depart, y_depart, z_depart = (
                        advection_schemes._backtrace_position_sparse(
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
                    )

                    sampled_u, sampled_v, sampled_w = (
                        advection_schemes._sample_trilinear_vec3_sparse(
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
                    )

                    advected_u[tile_index, local_i, local_j, local_k] = sampled_u
                    advected_v[tile_index, local_i, local_j, local_k] = sampled_v
                    advected_w[tile_index, local_i, local_j, local_k] = sampled_w


@njit(cache=True, parallel=True)
def update_velocity_maccormack(
    active_tile_coords,
    active_tile_slots,
    active_tile_count,
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
    nu: float,
    vorticity_magnitude: Any,
    vorticity_strength: float,
    temperature: Any,
    buoyancy_factor: float,
    t_reference: float,
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
    CPU kernel that updates velocity with a MacCormack-corrected
    semi-Lagrangian advection step on sparse velocity pools.

    The correction uses a reverse trace,

    .. math::

        \mathbf{u}^{n+1} = \mathbf{u}^{*}
        + \frac{1}{2}\left(\mathbf{u}^{n} - \widehat{\mathbf{u}}^{n}\right),

    after which viscosity, buoyancy, vorticity confinement, and configured
    external forces are accumulated explicitly.
    """
    total_tiles = active_tile_count

    for active_tile_n in prange(total_tiles):
        tile_i = active_tile_coords[active_tile_n, 0]
        tile_j = active_tile_coords[active_tile_n, 1]
        tile_k = active_tile_coords[active_tile_n, 2]
        tile_flat = (tile_i * tile_map.shape[1] + tile_j) * tile_map.shape[2] + tile_k
        for local_i in range(kernel_config.TILE_SIZE):
            for local_j in range(kernel_config.TILE_SIZE):
                for local_k in range(kernel_config.TILE_SIZE):
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
                    ) = sparse_managment.tile_to_index(
                        tile_flat,
                        local_i,
                        local_j,
                        local_k,
                        tile_map.shape[0],
                        tile_map.shape[1],
                        tile_map.shape[2],
                    )

                    tile_index = active_tile_slots[active_tile_n]
                    if tile_index == -1:
                        continue

                    if (
                        i < 1
                        or j < 1
                        or k < 1
                        or i >= nx - 1
                        or j >= ny - 1
                        or k >= nz - 1
                    ):
                        continue

                    Fx = 0.0
                    Fy = 0.0
                    Fz = 0.0

                    dt_over_delta = dt / delta
                    diffusion_coeff = nu * dt / (delta * delta)
                    force_coeff = dt / rho

                    u_center = u[tile_index, local_i, local_j, local_k]
                    v_center = v[tile_index, local_i, local_j, local_k]
                    w_center = w[tile_index, local_i, local_j, local_k]

                    x_depart, y_depart, z_depart = (
                        advection_schemes._backtrace_position_sparse(
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
                    )

                    x_forward, y_forward, z_forward = (
                        advection_schemes._forward_trace_position_sparse(
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
                    )

                    advected_u = predictor_u[tile_index, local_i, local_j, local_k]
                    advected_v = predictor_v[tile_index, local_i, local_j, local_k]
                    advected_w = predictor_w[tile_index, local_i, local_j, local_k]

                    reverse_u, reverse_v, reverse_w = (
                        advection_schemes._sample_trilinear_vec3_sparse(
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
                    )

                    corrected_u = advected_u + 0.5 * (u_center - reverse_u)
                    corrected_v = advected_v + 0.5 * (v_center - reverse_v)
                    corrected_w = advected_w + 0.5 * (w_center - reverse_w)

                    x0, y0, z0, x1, y1, z1, _, _, _ = (
                        advection_schemes._prepare_trilinear_coords(
                            x_depart, y_depart, z_depart, nx, ny, nz
                        )
                    )

                    u_lower, u_upper = (
                        advection_schemes._sample_cell_extrema_inner_sparse(
                            u, tile_map, x0, y0, z0, x1, y1, z1, u_initial
                        )
                    )
                    v_lower, v_upper = (
                        advection_schemes._sample_cell_extrema_inner_sparse(
                            v, tile_map, x0, y0, z0, x1, y1, z1, v_initial
                        )
                    )
                    w_lower, w_upper = (
                        advection_schemes._sample_cell_extrema_inner_sparse(
                            w, tile_map, x0, y0, z0, x1, y1, z1, w_initial
                        )
                    )

                    corrected_u = advection_schemes._clamp(
                        corrected_u, u_lower, u_upper
                    )
                    corrected_v = advection_schemes._clamp(
                        corrected_v, v_lower, v_upper
                    )
                    corrected_w = advection_schemes._clamp(
                        corrected_w, w_lower, w_upper
                    )

                    diffusion_x = diffusion_coeff * (
                        (
                            sparse_managment.get_pool_value(
                                u, tile_map, i + 1, j, k, u_initial
                            )
                            - 2.0 * u_center
                            + sparse_managment.get_pool_value(
                                u, tile_map, i - 1, j, k, u_initial
                            )
                        )
                        + (
                            sparse_managment.get_pool_value(
                                u, tile_map, i, j + 1, k, u_initial
                            )
                            - 2.0 * u_center
                            + sparse_managment.get_pool_value(
                                u, tile_map, i, j - 1, k, u_initial
                            )
                        )
                        + (
                            sparse_managment.get_pool_value(
                                u, tile_map, i, j, k + 1, u_initial
                            )
                            - 2.0 * u_center
                            + sparse_managment.get_pool_value(
                                u, tile_map, i, j, k - 1, u_initial
                            )
                        )
                    )
                    diffusion_y = diffusion_coeff * (
                        (
                            sparse_managment.get_pool_value(
                                v, tile_map, i + 1, j, k, v_initial
                            )
                            - 2.0 * v_center
                            + sparse_managment.get_pool_value(
                                v, tile_map, i - 1, j, k, v_initial
                            )
                        )
                        + (
                            sparse_managment.get_pool_value(
                                v, tile_map, i, j + 1, k, v_initial
                            )
                            - 2.0 * v_center
                            + sparse_managment.get_pool_value(
                                v, tile_map, i, j - 1, k, v_initial
                            )
                        )
                        + (
                            sparse_managment.get_pool_value(
                                v, tile_map, i, j, k + 1, v_initial
                            )
                            - 2.0 * v_center
                            + sparse_managment.get_pool_value(
                                v, tile_map, i, j, k - 1, v_initial
                            )
                        )
                    )
                    diffusion_z = diffusion_coeff * (
                        (
                            sparse_managment.get_pool_value(
                                w, tile_map, i + 1, j, k, w_initial
                            )
                            - 2.0 * w_center
                            + sparse_managment.get_pool_value(
                                w, tile_map, i - 1, j, k, w_initial
                            )
                        )
                        + (
                            sparse_managment.get_pool_value(
                                w, tile_map, i, j + 1, k, w_initial
                            )
                            - 2.0 * w_center
                            + sparse_managment.get_pool_value(
                                w, tile_map, i, j - 1, k, w_initial
                            )
                        )
                        + (
                            sparse_managment.get_pool_value(
                                w, tile_map, i, j, k + 1, w_initial
                            )
                            - 2.0 * w_center
                            + sparse_managment.get_pool_value(
                                w, tile_map, i, j, k - 1, w_initial
                            )
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
