from numba import cuda

import Solver.Kernel_GPU.advection_schemes as advection_schemes
import Solver.Kernel_GPU.kernel_config as kernel_config


@cuda.jit(cache=True)
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
    tile_i, tile_j, tile_k = cuda.grid(3)
    tiles_x, tiles_y, tiles_z = tile_mask.shape

    if tile_i >= tiles_x or tile_j >= tiles_y or tile_k >= tiles_z:
        return

    nx, ny, nz = T.shape
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


@cuda.jit(cache=True)
def dilate_active_scalar_tile_mask(tile_mask_in, tile_mask_out, padding_tiles):
    """
    Expand active scalar tiles by a tile-radius buffer.
    """
    tile_i, tile_j, tile_k = cuda.grid(3)
    tiles_x, tiles_y, tiles_z = tile_mask_in.shape

    if tile_i >= tiles_x or tile_j >= tiles_y or tile_k >= tiles_z:
        return

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


@cuda.jit(cache=True)
def fill_active_scalar_tile_mask(tile_mask, value):
    """
    Write one uniform activity value into the whole tile mask.
    """
    tile_i, tile_j, tile_k = cuda.grid(3)
    tiles_x, tiles_y, tiles_z = tile_mask.shape

    if tile_i >= tiles_x or tile_j >= tiles_y or tile_k >= tiles_z:
        return

    tile_mask[tile_i, tile_j, tile_k] = value


@cuda.jit(device=True, inline=True, cache=True)
def _active_tile_cell_indices(field_shape):
    """
    Map one active-tile launch to one cell index.
    """
    tile_i = cuda.blockIdx.x
    tile_j = cuda.blockIdx.y
    tile_k = cuda.blockIdx.z
    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y
    local_k = cuda.threadIdx.z

    i = tile_i * kernel_config.ACTIVE_TILE_SIZE + local_i
    j = tile_j * kernel_config.ACTIVE_TILE_SIZE + local_j
    k = tile_k * kernel_config.ACTIVE_TILE_SIZE + local_k
    nx, ny, nz = field_shape
    return tile_i, tile_j, tile_k, i, j, k, nx, ny, nz


@cuda.jit(cache=True)
def preserve_inactive_scalar_tiles(
    T, smoke, fuel, flame, T_out, smoke_out, fuel_out, flame_out, active_tile_mask
):
    """
    Copy current scalar values into output buffers for inactive tiles.
    """
    tile_i, tile_j, tile_k, i, j, k, nx, ny, nz = _active_tile_cell_indices(T.shape)

    if (
        tile_i >= active_tile_mask.shape[0]
        or tile_j >= active_tile_mask.shape[1]
        or tile_k >= active_tile_mask.shape[2]
    ):
        return
    if i >= nx or j >= ny or k >= nz:
        return
    if active_tile_mask[tile_i, tile_j, tile_k] != 0:
        return

    T_out[i, j, k] = T[i, j, k]
    smoke_out[i, j, k] = smoke[i, j, k]
    fuel_out[i, j, k] = fuel[i, j, k]
    flame_out[i, j, k] = flame[i, j, k]


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
    depart_x,
    depart_y,
    depart_z,
    delta,
    active_tile_mask,
):
    """
    Build the semi-Lagrangian predictor state for the scalar update.
    """
    tile_i, tile_j, tile_k, i, j, k, nx, ny, nz = _active_tile_cell_indices(u.shape)

    if (
        tile_i >= active_tile_mask.shape[0]
        or tile_j >= active_tile_mask.shape[1]
        or tile_k >= active_tile_mask.shape[2]
    ):
        return
    if active_tile_mask[tile_i, tile_j, tile_k] == 0:
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
    depart_x[i, j, k] = x_depart
    depart_y[i, j, k] = y_depart
    depart_z[i, j, k] = z_depart

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
    depart_x,
    depart_y,
    depart_z,
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
    fuel_burn_rate,
    fuel_ignition_temperature,
    minimum_oxygen_concentration,
    t_reference,
    maccormack_factor,
    active_tile_mask,
):
    """
    Update scalars with a MacCormack-corrected semi-Lagrangian advection step.

    The forward predictor arrays contain the first semi-Lagrangian pass. The
    corrector reverses the predictor, applies the MacCormack correction, clamps
    to the local departure-cell extrema and then evaluates combustion and
    dissipation source terms from the corrected state.
    """
    tile_i, tile_j, tile_k, i, j, k, nx, ny, nz = _active_tile_cell_indices(u.shape)

    if (
        tile_i >= active_tile_mask.shape[0]
        or tile_j >= active_tile_mask.shape[1]
        or tile_k >= active_tile_mask.shape[2]
    ):
        return
    if active_tile_mask[tile_i, tile_j, tile_k] == 0:
        return
    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    dt_over_delta = dt / delta

    x_depart = depart_x[i, j, k]
    y_depart = depart_y[i, j, k]
    z_depart = depart_z[i, j, k]
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

    T_corrected = T_advected + maccormack_factor * (T[i, j, k] - T_reverse)
    smoke_corrected = smoke_advected + maccormack_factor * (
        smoke[i, j, k] - smoke_reverse
    )
    fuel_corrected = fuel_advected + maccormack_factor * (fuel[i, j, k] - fuel_reverse)

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

    oxygen_center = 100.0 - smoke_corrected

    if (
        T_corrected > fuel_ignition_temperature
        and fuel_corrected > 0.0
        and oxygen_center >= minimum_oxygen_concentration
    ):
        fuel_source = -fuel_burn_rate * fuel_corrected
    else:
        fuel_source = 0.0

    temperature_source = -temperature_dissipation_rate * (
        T_corrected - t_reference
    ) + temperature_production_rate * (-fuel_source)
    smoke_source = (
        smoke_production_rate * (-fuel_source)
        - smoke_dissipation_rate * smoke_corrected
    )

    T_updated = T_corrected + dt * temperature_source
    smoke_updated = smoke_corrected + dt * smoke_source
    fuel_updated = fuel_corrected + dt * fuel_source

    T_out[i, j, k] = max(T_updated, 0.0)
    smoke_out[i, j, k] = min(max(smoke_updated, 0.0), 100.0)
    fuel_out[i, j, k] = min(max(fuel_updated, 0.0), 100.0)
    flame_out[i, j, k] = max(-fuel_source, 0.0)
