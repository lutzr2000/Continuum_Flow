from numba import cuda

import Solver.Kernel_GPU.advection_schemes as advection_schemes


@cuda.jit(cache=True)
def advect_scalar_fields_semi_lagrangian(
    T, smoke, fuel, u, v, w, dt, T_advected, smoke_advected, fuel_advected, delta
):
    """
    Backtrace all transported scalar fields once and store the advected predictor state.

    This helper is shared by the plain semi-Lagrangian scalar update and by the
    first predictor pass of the MacCormack scalar scheme.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape

    if i >= nx or j >= ny or k >= nz:
        return

    x_depart, y_depart, z_depart = advection_schemes._backtrace_position(
        u, v, w,
        float(i), float(j), float(k),
        dt / delta,
    )

    T_advected[i, j, k] = advection_schemes._sample_trilinear(T, x_depart, y_depart, z_depart)
    smoke_advected[i, j, k] = advection_schemes._sample_trilinear(smoke, x_depart, y_depart, z_depart)
    fuel_advected[i, j, k] = advection_schemes._sample_trilinear(fuel, x_depart, y_depart, z_depart)


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


@cuda.jit(cache=True)
def update_scalar_fields_semi_lagrangian(
    T, smoke, fuel, u, v, w, dt, T_out, smoke_out, fuel_out, flame_out,
    delta, temperature_dissipation_rate, temperature_production_rate,
    smoke_dissipation_rate, smoke_production_rate,
    fuel_burn_rate, fuel_ignition_temperature, minimum_oxygen_concentration, t_reference
):
    """
    Update temperature, smoke and fuel with semi-Lagrangian advection on the GPU.

    Scalars are backtraced through the current velocity field and reconstructed
    with trilinear interpolation before the combustion and dissipation source
    terms are applied.
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
    )

    T_advected = advection_schemes._sample_trilinear(T, x_depart, y_depart, z_depart)
    smoke_advected = advection_schemes._sample_trilinear(smoke, x_depart, y_depart, z_depart)
    fuel_advected = advection_schemes._sample_trilinear(fuel, x_depart, y_depart, z_depart)

    oxygen_center = 100.0 - smoke_advected

    if (
        T_advected > fuel_ignition_temperature and
        fuel_advected > 0.0 and
        oxygen_center >= minimum_oxygen_concentration
    ):
        fuel_source = -fuel_burn_rate * fuel_advected
    else:
        fuel_source = 0.0

    temperature_source = (
        -temperature_dissipation_rate * (T_advected - t_reference) +
        temperature_production_rate * (-fuel_source)
    )
    smoke_source = smoke_production_rate * (-fuel_source) - smoke_dissipation_rate * smoke_advected

    T_updated = T_advected + dt * temperature_source
    smoke_updated = smoke_advected + dt * smoke_source
    fuel_updated = fuel_advected + dt * fuel_source

    T_out[i, j, k] = max(T_updated, 0.0)
    smoke_out[i, j, k] = min(max(smoke_updated, 0.0), 100.0)
    fuel_out[i, j, k] = min(max(fuel_updated, 0.0), 100.0)
    flame_out[i, j, k] = max(-fuel_source, 0.0)


@cuda.jit(cache=True)
def update_scalar_fields_maccormack(
    T, smoke, fuel, predictor_T, predictor_smoke, predictor_fuel,
    u, v, w, dt, T_out, smoke_out, fuel_out, flame_out,
    delta, temperature_dissipation_rate, temperature_production_rate,
    smoke_dissipation_rate, smoke_production_rate,
    fuel_burn_rate, fuel_ignition_temperature, minimum_oxygen_concentration, t_reference
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
    )
    x_forward, y_forward, z_forward = advection_schemes._forward_trace_position(
        u, v, w,
        float(i), float(j), float(k),
        dt_over_delta,
    )

    T_advected = predictor_T[i, j, k]
    smoke_advected = predictor_smoke[i, j, k]
    fuel_advected = predictor_fuel[i, j, k]

    T_reverse = advection_schemes._sample_trilinear(predictor_T, x_forward, y_forward, z_forward)
    smoke_reverse = advection_schemes._sample_trilinear(predictor_smoke, x_forward, y_forward, z_forward)
    fuel_reverse = advection_schemes._sample_trilinear(predictor_fuel, x_forward, y_forward, z_forward)

    T_corrected = T_advected + 0.25 * (T[i, j, k] - T_reverse)
    smoke_corrected = smoke_advected + 0.25 * (smoke[i, j, k] - smoke_reverse)
    fuel_corrected = fuel_advected + 0.25 * (fuel[i, j, k] - fuel_reverse)

    T_lower, T_upper = advection_schemes._sample_cell_extrema(T, x_depart, y_depart, z_depart)
    smoke_lower, smoke_upper = advection_schemes._sample_cell_extrema(smoke, x_depart, y_depart, z_depart)
    fuel_lower, fuel_upper = advection_schemes._sample_cell_extrema(fuel, x_depart, y_depart, z_depart)

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
