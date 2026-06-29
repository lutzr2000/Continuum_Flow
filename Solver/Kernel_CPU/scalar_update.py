from numba import njit, prange

import Solver.Kernel_CPU.advection_schemes as advection_schemes
import Solver.Kernel_CPU.kernel_config as kernel_config


@njit(cache=True, inline="always")
def _active_tile_cell_indices(flat_idx, field_shape):
    """
    Map one flattened full-volume loop index to cell and active-tile indices.
    """
    ny = field_shape[1]
    nz = field_shape[2]

    i = flat_idx // (ny * nz)
    rem = flat_idx - i * ny * nz
    j = rem // nz
    k = rem - j * nz

    tile_i = i // kernel_config.ACTIVE_TILE_SIZE
    tile_j = j // kernel_config.ACTIVE_TILE_SIZE
    tile_k = k // kernel_config.ACTIVE_TILE_SIZE
    nx = field_shape[0]
    return tile_i, tile_j, tile_k, i, j, k, nx, ny, nz


@njit(cache=True, parallel=True)
def build_active_scalar_tile_mask(
    T,
    smoke,
    fuel,
    flame,
    tile_mask,
    t_reference,
    activity_threshold,
):
    """
    Mark 4x4x4 tiles that currently contain meaningful scalar activity.
    """
    tiles_x, tiles_y, tiles_z = tile_mask.shape
    nx, ny, nz = T.shape

    for tile_i in prange(tiles_x):
        for tile_j in range(tiles_y):
            for tile_k in range(tiles_z):
                cell_i_start = tile_i * kernel_config.ACTIVE_TILE_SIZE
                cell_j_start = tile_j * kernel_config.ACTIVE_TILE_SIZE
                cell_k_start = tile_k * kernel_config.ACTIVE_TILE_SIZE

                active = False
                for local_i in range(kernel_config.ACTIVE_TILE_SIZE):
                    i = cell_i_start + local_i
                    if i >= nx:
                        break
                    for local_j in range(kernel_config.ACTIVE_TILE_SIZE):
                        j = cell_j_start + local_j
                        if j >= ny:
                            break
                        for local_k in range(kernel_config.ACTIVE_TILE_SIZE):
                            k = cell_k_start + local_k
                            if k >= nz:
                                break
                            if (
                                smoke[i, j, k] > activity_threshold
                                or fuel[i, j, k] > activity_threshold
                                or flame[i, j, k] > activity_threshold
                                or abs(T[i, j, k] - t_reference) > activity_threshold
                            ):
                                active = True
                                break
                        if active:
                            break
                    if active:
                        break

                tile_mask[tile_i, tile_j, tile_k] = active

@njit(cache=True, parallel=True)
def dilate_active_scalar_tile_mask(tile_mask_in, tile_mask_out, padding_tiles):
    """
    Expand active scalar tiles by a tile-radius buffer.
    """
    tiles_x, tiles_y, tiles_z = tile_mask_in.shape

    for tile_i in prange(tiles_x):
        for tile_j in range(tiles_y):
            for tile_k in range(tiles_z):
                i_start = max(0, tile_i - padding_tiles)
                i_stop = min(tiles_x, tile_i + padding_tiles + 1)
                j_start = max(0, tile_j - padding_tiles)
                j_stop = min(tiles_y, tile_j + padding_tiles + 1)
                k_start = max(0, tile_k - padding_tiles)
                k_stop = min(tiles_z, tile_k + padding_tiles + 1)

                active = False
                for source_i in range(i_start, i_stop):
                    for source_j in range(j_start, j_stop):
                        for source_k in range(k_start, k_stop):
                            if tile_mask_in[source_i, source_j, source_k] != 0:
                                active = True
                                break
                        if active:
                            break
                    if active:
                        break

                tile_mask_out[tile_i, tile_j, tile_k] = active

@njit(cache=True, parallel=True)
def fill_active_scalar_tile_mask(tile_mask, value):
    """
    Write one uniform activity value into the whole tile mask.
    """
    tiles_x, tiles_y, tiles_z = tile_mask.shape

    for tile_i in prange(tiles_x):
        for tile_j in range(tiles_y):
            for tile_k in range(tiles_z):
                tile_mask[tile_i, tile_j, tile_k] = value


@njit(cache=True, parallel=True)
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
    active_tile_mask,
):
    """
    Build the semi-Lagrangian predictor state for the scalar update.
    """
    nx, ny, nz = u.shape
    tile_size = kernel_config.ACTIVE_TILE_SIZE

    for i in prange(nx):
        tile_i = i // tile_size
        for j in range(ny):
            tile_j = j // tile_size
            for k in range(nz):
                tile_k = k // tile_size
                if (
                    tile_i >= active_tile_mask.shape[0]
                    or tile_j >= active_tile_mask.shape[1]
                    or tile_k >= active_tile_mask.shape[2]
                ):
                    continue
                if active_tile_mask[tile_i, tile_j, tile_k] == 0:
                    continue

                x_depart, y_depart, z_depart = advection_schemes._backtrace_position(
                    u, v, w, float(i), float(j), float(k), dt / delta, nx, ny, nz,
                )
                pred_t, pred_smoke, pred_fuel = advection_schemes._sample_trilinear_vec3(
                    T, smoke, fuel, x_depart, y_depart, z_depart, nx, ny, nz,
                )
                predictor_T[i, j, k] = pred_t
                predictor_smoke[i, j, k] = pred_smoke
                predictor_fuel[i, j, k] = pred_fuel

@njit(cache=True, parallel=True)
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
    active_tile_mask,
):
    """
    Update scalars with a MacCormack-corrected semi-Lagrangian advection step.

    The forward predictor arrays contain the first semi-Lagrangian pass. The
    corrector reverses the predictor, applies the MacCormack correction, clamps
    to the local departure-cell extrema and then evaluates combustion and
    dissipation source terms from the corrected state.
    """
    nx, ny, nz = u.shape
    tile_size = kernel_config.ACTIVE_TILE_SIZE

    for i in prange(nx):
        tile_i = i // tile_size
        for j in range(ny):
            tile_j = j // tile_size
            for k in range(nz):
                tile_k = k // tile_size

                if (
                    tile_i >= active_tile_mask.shape[0]
                    or tile_j >= active_tile_mask.shape[1]
                    or tile_k >= active_tile_mask.shape[2]
                ):
                    continue
                if active_tile_mask[tile_i, tile_j, tile_k] == 0:
                    continue
                if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
                    continue

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
                smoke_corrected = smoke_advected + 0.5 * (smoke[i, j, k] - smoke_reverse)
                fuel_corrected = fuel_advected + 0.5 * (fuel[i, j, k] - fuel_reverse)

                x0, y0, z0, x1, y1, z1, _, _, _ = advection_schemes._prepare_trilinear_coords(
                    x_depart, y_depart, z_depart, nx, ny, nz
                )

                T_lower, T_upper = advection_schemes._sample_cell_extrema_inner(
                    T, x0, y0, z0, x1, y1, z1
                )
                smoke_lower, smoke_upper = advection_schemes._sample_cell_extrema_inner(
                    smoke, x0, y0, z0, x1, y1, z1
                )
                fuel_lower, fuel_upper = advection_schemes._sample_cell_extrema_inner(
                    fuel, x0, y0, z0, x1, y1, z1
                )

                T_corrected = advection_schemes._clamp(T_corrected, T_lower, T_upper)
                smoke_corrected = advection_schemes._clamp(
                    smoke_corrected, smoke_lower, smoke_upper
                )
                fuel_corrected = advection_schemes._clamp(fuel_corrected, fuel_lower, fuel_upper)

                oxygen_center = max(0.0, min(1.0, (100.0 - smoke_corrected) / 100.0))

                if T_corrected > fuel_ignition_temperature and fuel_corrected > 0.0:
                    n = advection_schemes._value_noise_3d(
                        float(i) * burn_noise_scale,
                        float(j) * burn_noise_scale,
                        float(k) * burn_noise_scale,
                        0,
                    )
                    burn_noise = 1.0 + burn_noise_amplitude * n
                    burn_noise = max(0.0, min(burn_noise, 2.0))

                    fuel_burn_source = (
                        -fuel_burn_rate * fuel_corrected * oxygen_center * burn_noise
                    )
                    temperature_burn_source = temperature_production_rate * -fuel_burn_source
                    smoke_burn_source = smoke_production_rate * -fuel_burn_source
                else:
                    temperature_burn_source = 0.0
                    smoke_burn_source = 0.0
                    fuel_burn_source = 0.0

                dT = T_corrected - t_reference

                cool_factor = abs(dT) / (abs(dT) + 200)

                temperature_dissipation = (
                    -temperature_dissipation_rate
                    * dT
                    * cool_factor
                )
                smoke_dissipation = -smoke_dissipation_rate * smoke_corrected
                fuel_dissipation = -fuel_dissipation_rate * fuel_corrected

                T_updated = T_corrected + dt * temperature_burn_source + dt * temperature_dissipation
                smoke_updated = smoke_corrected + dt * smoke_burn_source + dt * smoke_dissipation
                fuel_updated = fuel_corrected + dt * fuel_burn_source + dt * fuel_dissipation

                T_out[i, j, k] = max(T_updated, 0.0)
                smoke_out[i, j, k] = min(max(smoke_updated, 0.0), 100.0)
                fuel_out[i, j, k] = min(max(fuel_updated, 0.0), 100.0)
                flame_out[i, j, k] = max(-fuel_burn_source, 0.0)
