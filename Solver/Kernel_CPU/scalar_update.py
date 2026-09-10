from typing import Any

from numba import njit, prange

import Solver.Kernel_CPU.advection_schemes as advection_schemes
import Solver.Kernel_CPU.sparse_managment as sparse_managment
import Solver.Kernel_CPU.noise as noise
import Solver.Kernel_CPU.kernel_config as kernel_config


@njit(cache=True, parallel=True)
def predict_scalar_fields_semi_lagrangian(
    active_tile_coords,
    active_tile_slots,
    active_tile_count,
    T: Any,
    smoke: Any,
    fuel: Any,
    u: Any,
    v: Any,
    w: Any,
    dt: float,
    predictor_T: Any,
    predictor_smoke: Any,
    predictor_fuel: Any,
    delta: float,
    t_reference: float,
    tile_map: Any,
    u_initial: float,
    v_initial: float,
    w_initial: float,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    r"""
    Build the semi-Lagrangian predictor state for the scalar update.

    Each scalar :math:`\phi` is sampled at the backtraced departure point:

    .. math::

        \phi^{*}(\mathbf{x}) =
        \phi^n\!\left(\mathbf{x} - \Delta t\,\mathbf{u}(\mathbf{x})\right).
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

                    sampled_T, sampled_smoke, sampled_fuel = (
                        advection_schemes._sample_trilinear_vec3_sparse(
                            T,
                            smoke,
                            fuel,
                            tile_map,
                            x_depart,
                            y_depart,
                            z_depart,
                            nx,
                            ny,
                            nz,
                            t_reference,
                            0.0,
                            0.0,
                        )
                    )

                    predictor_T[tile_index, local_i, local_j, local_k] = sampled_T
                    predictor_smoke[tile_index, local_i, local_j, local_k] = (
                        sampled_smoke
                    )
                    predictor_fuel[tile_index, local_i, local_j, local_k] = sampled_fuel


@njit(cache=True, parallel=True)
def update_scalar_fields_maccormack(
    active_tile_coords,
    active_tile_slots,
    active_tile_count,
    T: Any,
    smoke: Any,
    fuel: Any,
    predictor_T: Any,
    predictor_smoke: Any,
    predictor_fuel: Any,
    u: Any,
    v: Any,
    w: Any,
    dt: float,
    T_out: Any,
    smoke_out: Any,
    fuel_out: Any,
    flame_out: Any,
    delta: float,
    temperature_dissipation_rate: float,
    temperature_production_rate: float,
    smoke_dissipation_rate: float,
    smoke_production_rate: float,
    fuel_dissipation_rate: float,
    fuel_burn_rate: float,
    fuel_ignition_temperature: Any,
    burn_noise_scale: float,
    burn_noise_amplitude: Any,
    t_reference: float,
    tile_map: Any,
    u_initial: float,
    v_initial: float,
    w_initial: float,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    r"""
    Update scalars with a MacCormack-corrected semi-Lagrangian advection step.

    A reverse advection estimates the predictor error and applies

    .. math::

        \phi^{n+1} = \phi^{*}
        + \frac{1}{2}\left(\phi^n - \widehat{\phi}^n\right).

    The corrected value is clamped to local source extrema before dissipation,
    production, and combustion terms are applied.
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

                    dt_over_delta = dt / delta

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

                    T_advected = predictor_T[tile_index, local_i, local_j, local_k]
                    smoke_advected = predictor_smoke[
                        tile_index, local_i, local_j, local_k
                    ]
                    fuel_advected = predictor_fuel[
                        tile_index, local_i, local_j, local_k
                    ]

                    T_reverse, smoke_reverse, fuel_reverse = (
                        advection_schemes._sample_trilinear_vec3_sparse(
                            predictor_T,
                            predictor_smoke,
                            predictor_fuel,
                            tile_map,
                            x_forward,
                            y_forward,
                            z_forward,
                            nx,
                            ny,
                            nz,
                            t_reference,
                            0.0,
                            0.0,
                        )
                    )

                    T_corrected = T_advected + 0.5 * (
                        T[tile_index, local_i, local_j, local_k] - T_reverse
                    )
                    smoke_corrected = smoke_advected + 0.5 * (
                        smoke[tile_index, local_i, local_j, local_k] - smoke_reverse
                    )
                    fuel_corrected = fuel_advected + 0.5 * (
                        fuel[tile_index, local_i, local_j, local_k] - fuel_reverse
                    )

                    x0, y0, z0, x1, y1, z1, _, _, _ = (
                        advection_schemes._prepare_trilinear_coords(
                            x_depart, y_depart, z_depart, nx, ny, nz
                        )
                    )

                    T_lower, T_upper = (
                        advection_schemes._sample_cell_extrema_inner_sparse(
                            T, tile_map, x0, y0, z0, x1, y1, z1, t_reference
                        )
                    )
                    smoke_lower, smoke_upper = (
                        advection_schemes._sample_cell_extrema_inner_sparse(
                            smoke, tile_map, x0, y0, z0, x1, y1, z1, 0.0
                        )
                    )
                    fuel_lower, fuel_upper = (
                        advection_schemes._sample_cell_extrema_inner_sparse(
                            fuel, tile_map, x0, y0, z0, x1, y1, z1, 0.0
                        )
                    )

                    T_corrected = advection_schemes._clamp(
                        T_corrected, T_lower, T_upper
                    )
                    smoke_corrected = advection_schemes._clamp(
                        smoke_corrected, smoke_lower, smoke_upper
                    )
                    fuel_corrected = advection_schemes._clamp(
                        fuel_corrected, fuel_lower, fuel_upper
                    )

                    oxygen_center = max(
                        0.0, min(1.0, (100.0 - smoke_corrected) / 100.0)
                    )

                    if T_corrected > fuel_ignition_temperature and fuel_corrected > 0.0:
                        fuel_xp = sparse_managment.get_pool_value(
                            fuel, tile_map, i + 1, j, k, 0.0
                        )
                        fuel_xm = sparse_managment.get_pool_value(
                            fuel, tile_map, i - 1, j, k, 0.0
                        )
                        fuel_yp = sparse_managment.get_pool_value(
                            fuel, tile_map, i, j + 1, k, 0.0
                        )
                        fuel_ym = sparse_managment.get_pool_value(
                            fuel, tile_map, i, j - 1, k, 0.0
                        )
                        fuel_zp = sparse_managment.get_pool_value(
                            fuel, tile_map, i, j, k + 1, 0.0
                        )
                        fuel_zm = sparse_managment.get_pool_value(
                            fuel, tile_map, i, j, k - 1, 0.0
                        )

                        T_xp = sparse_managment.get_pool_value(
                            T, tile_map, i + 1, j, k, t_reference
                        )
                        T_xm = sparse_managment.get_pool_value(
                            T, tile_map, i - 1, j, k, t_reference
                        )
                        T_yp = sparse_managment.get_pool_value(
                            T, tile_map, i, j + 1, k, t_reference
                        )
                        T_ym = sparse_managment.get_pool_value(
                            T, tile_map, i, j - 1, k, t_reference
                        )
                        T_zp = sparse_managment.get_pool_value(
                            T, tile_map, i, j, k + 1, t_reference
                        )
                        T_zm = sparse_managment.get_pool_value(
                            T, tile_map, i, j, k - 1, t_reference
                        )

                        fuel_front = (
                            abs(fuel_xp - fuel_xm)
                            + abs(fuel_yp - fuel_ym)
                            + abs(fuel_zp - fuel_zm)
                        ) / 100.0

                        temperature_front = (
                            abs(T_xp - T_xm) + abs(T_yp - T_ym) + abs(T_zp - T_zm)
                        ) / max(fuel_ignition_temperature, 1.0)

                        front_factor = min(
                            max(0.5 * (fuel_front + temperature_front), 0.0), 1.0
                        )

                        n = noise._value_noise_3d(
                            float(i) * burn_noise_scale,
                            float(j) * burn_noise_scale,
                            float(k) * burn_noise_scale,
                            0,
                        )

                        burn_noise = 1.0 + burn_noise_amplitude * n
                        burn_noise = max(0.0, min(burn_noise, 2.0))

                        burn_front_weight = front_factor * front_factor

                        fuel_burn_source = (
                            -fuel_burn_rate
                            * fuel_corrected
                            * oxygen_center
                            * burn_noise
                            * burn_front_weight
                        )
                        temperature_burn_source = (
                            temperature_production_rate * -fuel_burn_source
                        )
                        smoke_burn_source = smoke_production_rate * -fuel_burn_source
                    else:
                        temperature_burn_source = 0.0
                        smoke_burn_source = 0.0
                        fuel_burn_source = 0.0

                    dT = T_corrected - t_reference
                    cool_factor = abs(dT) / (abs(dT) + 200)

                    temperature_dissipation = (
                        -temperature_dissipation_rate * dT * cool_factor
                    )
                    smoke_dissipation = -smoke_dissipation_rate * smoke_corrected
                    fuel_dissipation = -fuel_dissipation_rate * fuel_corrected

                    T_updated = (
                        T_corrected
                        + dt * temperature_burn_source
                        + dt * temperature_dissipation
                    )
                    smoke_updated = (
                        smoke_corrected
                        + dt * smoke_burn_source
                        + dt * smoke_dissipation
                    )
                    fuel_updated = (
                        fuel_corrected + dt * fuel_burn_source + dt * fuel_dissipation
                    )

                    T_out[tile_index, local_i, local_j, local_k] = max(T_updated, 0.0)
                    smoke_out[tile_index, local_i, local_j, local_k] = min(
                        max(smoke_updated, 0.0), 100.0
                    )
                    fuel_out[tile_index, local_i, local_j, local_k] = min(
                        max(fuel_updated, 0.0), 100.0
                    )
                    flame_out[tile_index, local_i, local_j, local_k] = max(
                        -fuel_burn_source, 0.0
                    )
