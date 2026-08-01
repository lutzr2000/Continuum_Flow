from numba import cuda

import Solver.Kernel_GPU_sparse.advection_schemes as advection_schemes
import Solver.Kernel_GPU_sparse.sparse_managment as sparse_managment  


@cuda.jit(cache=True)
def predict_scalar_fields_semi_lagrangian(
    T,
    smoke,
    fuel,
    u,
    v,
    w,
    dt,
    predictor_T,
    predictor_smoke,
    predictor_fuel,
    delta,
    tile_map,
):
    """
    Build the semi-Lagrangian predictor state for the scalar update.
    """
    tile_i, tile_j, tile_k, i, j, k, nx, ny, nz = sparse_managment.tile_to_index(
        u.shape
    )

    tile_index = tile_map[tile_i, tile_j, tile_k]

    if tile_index == -1:
        return
    
    if i >= nx or j >= ny or k >= nz:
        return

    x_depart, y_depart, z_depart = advection_schemes._backtrace_position(
        u,
        v,
        w,
        float(i),
        float(j),
        float(k),
        dt / delta,
        nx,
        ny,
        nz,
    )

    predictor_T[i, j, k], predictor_smoke[i, j, k], predictor_fuel[i, j, k] = (
        advection_schemes._sample_trilinear_vec3(
            T,
            smoke,
            fuel,
            x_depart,
            y_depart,
            z_depart,
            nx,
            ny,
            nz,
        )
    )


@cuda.jit(cache=True)
def update_scalar_fields_maccormack(
    T,
    smoke,
    fuel,
    predictor_T,
    predictor_smoke,
    predictor_fuel,
    u,
    v,
    w,
    dt,
    T_out,
    smoke_out,
    fuel_out,
    flame_out,
    delta,
    temperature_dissipation_rate,
    temperature_production_rate,
    smoke_dissipation_rate,
    smoke_production_rate,
    fuel_dissipation_rate,
    fuel_burn_rate,
    fuel_ignition_temperature,
    burn_noise_scale,
    burn_noise_amplitude,
    t_reference,
    tile_map,
):
    """
    Update scalars with a MacCormack-corrected semi-Lagrangian advection step.

    The forward predictor arrays contain the first semi-Lagrangian pass. The
    corrector reverses the predictor, applies the MacCormack correction, clamps
    to the local departure-cell extrema and then evaluates combustion and
    dissipation source terms from the corrected state.
    """
    tile_i, tile_j, tile_k, i, j, k, nx, ny, nz = sparse_managment.tile_to_index(
        u.shape
    )

    tile_index = tile_map[tile_i, tile_j, tile_k]

    if tile_index == -1:
        return
    
    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    dt_over_delta = dt / delta

    x_depart, y_depart, z_depart = advection_schemes._backtrace_position(
        u,
        v,
        w,
        float(i),
        float(j),
        float(k),
        dt / delta,
        nx,
        ny,
        nz,
    )

    # Forward trace from the departure point:
    # approximately:
    # x_forward = x_depart + dt * u(x_depart)
    x_forward, y_forward, z_forward = advection_schemes._forward_trace_position(
        u,
        v,
        w,
        x_depart,
        y_depart,
        z_depart,
        dt_over_delta,
        nx,
        ny,
        nz,
    )

    T_advected = predictor_T[i, j, k]
    smoke_advected = predictor_smoke[i, j, k]
    fuel_advected = predictor_fuel[i, j, k]

    # find depart scalar values
    T_reverse, smoke_reverse, fuel_reverse = advection_schemes._sample_trilinear_vec3(
        predictor_T,
        predictor_smoke,
        predictor_fuel,
        x_forward,
        y_forward,
        z_forward,
        nx,
        ny,
        nz,
    )

    T_corrected = T_advected + 0.5 * (T[i, j, k] - T_reverse)
    smoke_corrected = smoke_advected + 0.5 * (
        smoke[i, j, k] - smoke_reverse
    )
    fuel_corrected = fuel_advected + 0.5 * (fuel[i, j, k] - fuel_reverse)

    x0, y0, z0, x1, y1, z1, _, _, _ = advection_schemes._prepare_trilinear_coords(
        x_depart, y_depart, z_depart, nx, ny, nz
    )

    # find the scalars upper and lower bounds of neighbour cells at backtrace positions
    T_lower, T_upper = advection_schemes._sample_cell_extrema_inner(
        T, x0, y0, z0, x1, y1, z1
    )
    smoke_lower, smoke_upper = advection_schemes._sample_cell_extrema_inner(
        smoke, x0, y0, z0, x1, y1, z1
    )
    fuel_lower, fuel_upper = advection_schemes._sample_cell_extrema_inner(
        fuel, x0, y0, z0, x1, y1, z1
    )

    # clamping to bounds
    T_corrected = advection_schemes._clamp(T_corrected, T_lower, T_upper)
    smoke_corrected = advection_schemes._clamp(
        smoke_corrected, smoke_lower, smoke_upper
    )
    fuel_corrected = advection_schemes._clamp(fuel_corrected, fuel_lower, fuel_upper)

    # oxygen
    oxygen_center = max(0.0, min(1.0, (100.0 - smoke_corrected) / 100.0))

    # burn logic
    if (
        T_corrected > fuel_ignition_temperature
        and fuel_corrected > 0.0
    ):
        n = advection_schemes._value_noise_3d(
            float(i) * burn_noise_scale,
            float(j) * burn_noise_scale,
            float(k) * burn_noise_scale,
            0,
        )

        burn_noise = 1.0 + burn_noise_amplitude * n
        burn_noise = max(0.0, min(burn_noise, 2.0))

        fuel_burn_source = -fuel_burn_rate * fuel_corrected * oxygen_center * burn_noise
        temperature_burn_source = temperature_production_rate * -fuel_burn_source 
        smoke_burn_source = smoke_production_rate * -fuel_burn_source
    else:
        temperature_burn_source = 0.0
        smoke_burn_source = 0.0
        fuel_burn_source = 0.0

    # dissipation
    dT = T_corrected - t_reference

    cool_factor = abs(dT) / (abs(dT) + 200)

    temperature_dissipation = (
        -temperature_dissipation_rate
        * dT
        * cool_factor
    )
    smoke_dissipation = - smoke_dissipation_rate * smoke_corrected
    fuel_dissipation = -fuel_dissipation_rate * fuel_corrected

    T_updated = T_corrected + dt * temperature_burn_source + dt * temperature_dissipation
    smoke_updated = smoke_corrected + dt * smoke_burn_source + dt * smoke_dissipation
    fuel_updated = fuel_corrected + dt * fuel_burn_source + dt * fuel_dissipation

    # ensure physically reasonable bounds
    T_out[i, j, k] = max(T_updated, 0.0)
    smoke_out[i, j, k] = min(max(smoke_updated, 0.0), 100.0)
    fuel_out[i, j, k] = min(max(fuel_updated, 0.0), 100.0)
    flame_out[i, j, k] = max(-fuel_burn_source, 0.0)
