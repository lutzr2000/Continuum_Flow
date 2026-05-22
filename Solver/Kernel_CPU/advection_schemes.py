from numba import njit, prange

import Solver.Kernel_CPU.kernel_config as kernel_config


@njit(cache=True, inline="always")
def _clamp(value, lower, upper):
    """
    Clamp one scalar value to the inclusive `[lower, upper]` interval.
    """
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


@njit(cache=True, inline="always")
def _prepare_trilinear_coords(x, y, z, nx, ny, nz):
    """
    Clamp one sample position and derive the surrounding cell coordinates.
    """
    if x < 0.0:
        x = 0.0
    elif x > nx - 1:
        x = nx - 1.0

    if y < 0.0:
        y = 0.0
    elif y > ny - 1:
        y = ny - 1.0

    if z < 0.0:
        z = 0.0
    elif z > nz - 1:
        z = nz - 1.0

    x0 = int(x)
    y0 = int(y)
    z0 = int(z)

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    if x1 >= nx:
        x1 = nx - 1
    if y1 >= ny:
        y1 = ny - 1
    if z1 >= nz:
        z1 = nz - 1

    tx = x - x0
    ty = y - y0
    tz = z - z0

    return x0, y0, z0, x1, y1, z1, tx, ty, tz


@njit(cache=True, inline="always")
def _sample_trilinear_inner(field, x0, y0, z0, x1, y1, z1, tx, ty, tz):
    """
    Blend one scalar field from precomputed trilinear coordinates.
    """
    c000 = field[x0, y0, z0]
    c100 = field[x1, y0, z0]
    c010 = field[x0, y1, z0]
    c110 = field[x1, y1, z0]
    c001 = field[x0, y0, z1]
    c101 = field[x1, y0, z1]
    c011 = field[x0, y1, z1]
    c111 = field[x1, y1, z1]

    c00 = c000 + tx * (c100 - c000)
    c10 = c010 + tx * (c110 - c010)
    c01 = c001 + tx * (c101 - c001)
    c11 = c011 + tx * (c111 - c011)

    c0 = c00 + ty * (c10 - c00)
    c1 = c01 + ty * (c11 - c01)
    return c0 + tz * (c1 - c0)


@njit(cache=True, inline="always")
def _sample_cell_extrema_inner(field, x0, y0, z0, x1, y1, z1):
    """
    Return the corner-value range for one cell from precomputed coordinates.
    """
    c000 = field[x0, y0, z0]
    c100 = field[x1, y0, z0]
    c010 = field[x0, y1, z0]
    c110 = field[x1, y1, z0]
    c001 = field[x0, y0, z1]
    c101 = field[x1, y0, z1]
    c011 = field[x0, y1, z1]
    c111 = field[x1, y1, z1]

    lower = min(
        min(min(c000, c100), min(c010, c110)), min(min(c001, c101), min(c011, c111))
    )
    upper = max(
        max(max(c000, c100), max(c010, c110)), max(max(c001, c101), max(c011, c111))
    )
    return lower, upper


@njit(cache=True, inline="always")
def _sample_trilinear_vec3(field_x, field_y, field_z, x, y, z, nx, ny, nz):
    """
    Sample three scalar fields at one position while reusing the same coordinates.
    """
    x0, y0, z0, x1, y1, z1, tx, ty, tz = _prepare_trilinear_coords(x, y, z, nx, ny, nz)

    sample_x = _sample_trilinear_inner(field_x, x0, y0, z0, x1, y1, z1, tx, ty, tz)
    sample_y = _sample_trilinear_inner(field_y, x0, y0, z0, x1, y1, z1, tx, ty, tz)
    sample_z = _sample_trilinear_inner(field_z, x0, y0, z0, x1, y1, z1, tx, ty, tz)
    return sample_x, sample_y, sample_z


@njit(cache=True, inline="always")
def _backtrace_position(u, v, w, x_start, y_start, z_start, dt_over_delta, nx, ny, nz):
    """
    Backtrace one particle position through the velocity field with two substeps.

    The helper is used by the semi-Lagrangian and MacCormack advection passes
    to estimate the departure point in grid coordinates.
    """
    n_substeps = 3
    substep_dt = dt_over_delta * 1 / n_substeps
    x_pos = x_start
    y_pos = y_start
    z_pos = z_start

    for _ in range(n_substeps):
        u_sample, v_sample, w_sample = _sample_trilinear_vec3(
            u, v, w, x_pos, y_pos, z_pos, nx, ny, nz
        )

        x_pos -= substep_dt * u_sample
        y_pos -= substep_dt * v_sample
        z_pos -= substep_dt * w_sample

    return x_pos, y_pos, z_pos


@njit(cache=True, inline="always")
def _forward_trace_position(
    u, v, w, x_start, y_start, z_start, dt_over_delta, nx, ny, nz
):
    """
    Forward-trace one position by reusing the backtracer with a negated timestep.
    """
    return _backtrace_position(
        u, v, w, x_start, y_start, z_start, -dt_over_delta, nx, ny, nz
    )


@njit(cache=True, parallel=True, fastmath=True)
def preserve_inactive_velocity_tiles(u, v, w, u_out, v_out, w_out, active_tile_mask):
    """
    Copy unchanged velocity values for inactive tiles.
    """
    nx, ny, nz = u.shape
    tile_size = kernel_config.ACTIVE_TILE_SIZE
    for i in prange(nx):
        tile_i = i // tile_size
        for j in range(ny):
            tile_j = j // tile_size
            for k in range(nz):
                tile_k = k // tile_size
                if active_tile_mask[tile_i, tile_j, tile_k]:
                    continue
                u_out[i, j, k] = u[i, j, k]
                v_out[i, j, k] = v[i, j, k]
                w_out[i, j, k] = w[i, j, k]


@njit(cache=True, parallel=True, fastmath=True)
def advect_velocity_semi_lagrangian(
    u,
    v,
    w,
    advected_u,
    advected_v,
    advected_w,
    depart_x,
    depart_y,
    depart_z,
    dt,
    delta,
    active_tile_mask,
):
    """
    Backtrace the velocity field once and store the purely advected values.
    """
    nx, ny, nz = u.shape
    tile_size = kernel_config.ACTIVE_TILE_SIZE
    for i in prange(nx):
        tile_i = i // tile_size
        for j in range(ny):
            tile_j = j // tile_size
            for k in range(nz):
                tile_k = k // tile_size
                if not active_tile_mask[tile_i, tile_j, tile_k]:
                    continue

                x_depart, y_depart, z_depart = _backtrace_position(
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

                advected_u[i, j, k], advected_v[i, j, k], advected_w[i, j, k] = (
                    _sample_trilinear_vec3(
                        u,
                        v,
                        w,
                        x_depart,
                        y_depart,
                        z_depart,
                        nx,
                        ny,
                        nz,
                    )
                )


@njit(cache=True, parallel=True, fastmath=True)
def update_velocity_maccormack(
    u,
    v,
    w,
    predictor_u,
    predictor_v,
    predictor_w,
    depart_x,
    depart_y,
    depart_z,
    dt,
    Fx,
    Fy,
    Fz,
    un,
    vn,
    wn,
    delta,
    rho,
    nu,
    maccormack_factor,
    active_tile_mask,
):
    """
    Update velocity with a MacCormack-corrected semi-Lagrangian advection step.

    A forward semi-Lagrangian predictor is supplied in `predictor_*`. The
    corrector reverses that predictor, applies the standard MacCormack
    compensation term, clamps the result to the departure cell range, and then
    adds pressure, diffusion and external forces explicitly.
    """
    nx, ny, nz = u.shape
    tile_size = kernel_config.ACTIVE_TILE_SIZE

    for i in prange(1, nx - 1):
        tile_i = i // tile_size
        for j in range(1, ny - 1):
            tile_j = j // tile_size
            for k in range(1, nz - 1):
                tile_k = k // tile_size
                if not active_tile_mask[tile_i, tile_j, tile_k]:
                    continue

                dt_over_delta = dt / delta
                diffusion_coeff = nu * dt / (delta * delta)
                force_coeff = dt / rho

                u_center = u[i, j, k]
                v_center = v[i, j, k]
                w_center = w[i, j, k]

                x_depart = depart_x[i, j, k]
                y_depart = depart_y[i, j, k]
                z_depart = depart_z[i, j, k]
                x_forward, y_forward, z_forward = _forward_trace_position(
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

                advected_u = predictor_u[i, j, k]
                advected_v = predictor_v[i, j, k]
                advected_w = predictor_w[i, j, k]

                reverse_u, reverse_v, reverse_w = _sample_trilinear_vec3(
                    predictor_u,
                    predictor_v,
                    predictor_w,
                    x_forward,
                    y_forward,
                    z_forward,
                    nx,
                    ny,
                    nz,
                )

                corrected_u = advected_u + maccormack_factor * (u_center - reverse_u)
                corrected_v = advected_v + maccormack_factor * (v_center - reverse_v)
                corrected_w = advected_w + maccormack_factor * (w_center - reverse_w)

                x0, y0, z0, x1, y1, z1, _, _, _ = _prepare_trilinear_coords(
                    x_depart, y_depart, z_depart, nx, ny, nz
                )
                u_lower, u_upper = _sample_cell_extrema_inner(u, x0, y0, z0, x1, y1, z1)
                v_lower, v_upper = _sample_cell_extrema_inner(v, x0, y0, z0, x1, y1, z1)
                w_lower, w_upper = _sample_cell_extrema_inner(w, x0, y0, z0, x1, y1, z1)

                corrected_u = _clamp(corrected_u, u_lower, u_upper)
                corrected_v = _clamp(corrected_v, v_lower, v_upper)
                corrected_w = _clamp(corrected_w, w_lower, w_upper)

                diffusion_x = diffusion_coeff * (
                    (u[i + 1, j, k] - 2.0 * u_center + u[i - 1, j, k])
                    + (u[i, j + 1, k] - 2.0 * u_center + u[i, j - 1, k])
                    + (u[i, j, k + 1] - 2.0 * u_center + u[i, j, k - 1])
                )
                diffusion_y = diffusion_coeff * (
                    (v[i + 1, j, k] - 2.0 * v_center + v[i - 1, j, k])
                    + (v[i, j + 1, k] - 2.0 * v_center + v[i, j - 1, k])
                    + (v[i, j, k + 1] - 2.0 * v_center + v[i, j, k - 1])
                )
                diffusion_z = diffusion_coeff * (
                    (w[i + 1, j, k] - 2.0 * w_center + w[i - 1, j, k])
                    + (w[i, j + 1, k] - 2.0 * w_center + w[i, j - 1, k])
                    + (w[i, j, k + 1] - 2.0 * w_center + w[i, j, k - 1])
                )

                u_raw = corrected_u + diffusion_x + force_coeff * Fx[i, j, k]
                v_raw = corrected_v + diffusion_y + force_coeff * Fy[i, j, k]
                w_raw = corrected_w + diffusion_z + force_coeff * Fz[i, j, k]

                un[i, j, k] = u_raw
                vn[i, j, k] = v_raw
                wn[i, j, k] = w_raw
