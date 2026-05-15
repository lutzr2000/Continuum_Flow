from numba import cuda

import Solver.Kernel_GPU.advection_schemes as advection_schemes


@cuda.jit(cache=True)
def predict_scalar_fields_maccormack(
    T, smoke, fuel, u, v, w, dt, predictor_T, predictor_smoke, predictor_fuel, delta
):
    """
    Build the forward predictor state for the MacCormack scalar update.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape

    if i >= nx or j >= ny or k >= nz:
        return

    x_depart, y_depart, z_depart = advection_schemes._backtrace_position(
        u, v, w,
        float(i), float(j), float(k),
        dt / delta,
        nx, ny, nz,
    )

    predictor_T[i, j, k], predictor_smoke[i, j, k], predictor_fuel[i, j, k] = advection_schemes._sample_trilinear_vec3(
        T, smoke, fuel,
        x_depart, y_depart, z_depart,
        nx, ny, nz,
    )


@cuda.jit(cache=True)
def update_scalar_fields_maccormack(
    T, smoke, fuel, predictor_T, predictor_smoke, predictor_fuel,
    u, v, w, dt, T_out, smoke_out, fuel_out, flame_out,
    delta, temperature_dissipation_rate, temperature_production_rate,
    smoke_dissipation_rate, smoke_production_rate,
    fuel_burn_rate, fuel_ignition_temperature, minimum_oxygen_concentration, t_reference,
    maccormack_factor
):
    """
    Update scalars with a MacCormack-corrected semi-Lagrangian advection step.

    The forward predictor arrays contain the first semi-Lagrangian pass. The
    corrector reverses the predictor, applies the MacCormack correction, clamps
    to the local departure-cell extrema and then evaluates combustion and
    dissipation source terms from the corrected state.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    dt_over_delta = dt / delta

    x_depart, y_depart, z_depart = advection_schemes._backtrace_position(
        u, v, w,
        float(i), float(j), float(k),
        dt_over_delta,
        nx, ny, nz,
    )
    x_forward, y_forward, z_forward = advection_schemes._forward_trace_position(
        u, v, w,
        float(i), float(j), float(k),
        dt_over_delta,
        nx, ny, nz,
    )

    T_advected = predictor_T[i, j, k]
    smoke_advected = predictor_smoke[i, j, k]
    fuel_advected = predictor_fuel[i, j, k]

    T_reverse, smoke_reverse, fuel_reverse = advection_schemes._sample_trilinear_vec3(
        predictor_T, predictor_smoke, predictor_fuel,
        x_forward, y_forward, z_forward,
        nx, ny, nz,
    )

    T_corrected = T_advected + maccormack_factor * (T[i, j, k] - T_reverse)
    smoke_corrected = smoke_advected + maccormack_factor * (smoke[i, j, k] - smoke_reverse)
    fuel_corrected = fuel_advected + maccormack_factor * (fuel[i, j, k] - fuel_reverse)

    x0, y0, z0, x1, y1, z1, _, _, _ = advection_schemes._prepare_trilinear_coords(
        x_depart, y_depart, z_depart, nx, ny, nz
    )
    T_lower, T_upper = advection_schemes._sample_cell_extrema_inner(T, x0, y0, z0, x1, y1, z1)
    smoke_lower, smoke_upper = advection_schemes._sample_cell_extrema_inner(smoke, x0, y0, z0, x1, y1, z1)
    fuel_lower, fuel_upper = advection_schemes._sample_cell_extrema_inner(fuel, x0, y0, z0, x1, y1, z1)

    T_corrected = advection_schemes._clamp(T_corrected, T_lower, T_upper)
    smoke_corrected = advection_schemes._clamp(smoke_corrected, smoke_lower, smoke_upper)
    fuel_corrected = advection_schemes._clamp(fuel_corrected, fuel_lower, fuel_upper)

    oxygen_center = 100.0 - smoke_corrected

    if (
        T_corrected > fuel_ignition_temperature and
        fuel_corrected > 0.0 and
        oxygen_center >= minimum_oxygen_concentration
    ):
        fuel_source = -fuel_burn_rate * fuel_corrected
    else:
        fuel_source = 0.0

    temperature_source = (
        -temperature_dissipation_rate * (T_corrected - t_reference) +
        temperature_production_rate * (-fuel_source)
    )
    smoke_source = smoke_production_rate * (-fuel_source) - smoke_dissipation_rate * smoke_corrected

    T_updated = T_corrected + dt * temperature_source
    smoke_updated = smoke_corrected + dt * smoke_source
    fuel_updated = fuel_corrected + dt * fuel_source

    T_out[i, j, k] = max(T_updated, 0.0)
    smoke_out[i, j, k] = min(max(smoke_updated, 0.0), 100.0)
    fuel_out[i, j, k] = min(max(fuel_updated, 0.0), 100.0)
    flame_out[i, j, k] = max(-fuel_source, 0.0)
