from numba import cuda

import Solver.Kernel_GPU.kernel_config as kernel_config


@cuda.jit(device=True, inline=True, cache=True)
def _clamp(value, lower, upper):
    """
    Clamp one scalar value to the inclusive `[lower, upper]` interval.
    """
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


@cuda.jit(device=True, inline=True, cache=True)
def _prepare_trilinear_coords(x, y, z, nx, ny, nz):
    """
    Clamp one sample position to the domain and derive the surrounding cell coordinates.
    Lastly computes interpolation weights tx, ty, tz.
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


@cuda.jit(device=True, inline=True, cache=True)
def _sample_trilinear_inner(field, x0, y0, z0, x1, y1, z1, tx, ty, tz):
    """
    Blend one scalar field from precomputed trilinear coordinates in _prepare_trilinear_coords()
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


@cuda.jit(device=True, inline=True, cache=True)
def _sample_cell_extrema_inner(field, x0, y0, z0, x1, y1, z1):
    """
    Return the corner-value range for one cell from precomputed coordinates.
    Later used to limit the interpolated sample velocity to the surrounding values.
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


@cuda.jit(device=True, inline=True, cache=True)
def _sample_trilinear_vec3(field_x, field_y, field_z, x, y, z, nx, ny, nz):
    """
    Sample three scalar fields at one position while reusing the same coordinates.
    Can be used if a vector instead of a scalar value is backtraced.
    """
    x0, y0, z0, x1, y1, z1, tx, ty, tz = _prepare_trilinear_coords(x, y, z, nx, ny, nz)

    sample_x = _sample_trilinear_inner(field_x, x0, y0, z0, x1, y1, z1, tx, ty, tz)
    sample_y = _sample_trilinear_inner(field_y, x0, y0, z0, x1, y1, z1, tx, ty, tz)
    sample_z = _sample_trilinear_inner(field_z, x0, y0, z0, x1, y1, z1, tx, ty, tz)
    return sample_x, sample_y, sample_z


@cuda.jit(device=True, inline=True, cache=True)
def _backtrace_position(u, v, w, x_start, y_start, z_start, dt_over_delta, nx, ny, nz):
    """
    Backtrace one particles position through the velocity field with n_substepds (hardcoded:3).

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


@cuda.jit(device=True, inline=True, cache=True)
def _forward_trace_position(
    u, v, w, x_start, y_start, z_start, dt_over_delta, nx, ny, nz
):
    """
    Forward-trace one position by reusing the backtracer with a negated timestep.
    """
    return _backtrace_position(
        u, v, w, x_start, y_start, z_start, -dt_over_delta, nx, ny, nz
    )


@cuda.jit(device=True, inline=True, cache=True)
def _active_tile_cell_indices(field_shape):
    """
    Map one active-tile launch to one cell index. Used for itterating through active
    and inactive blocks.
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
def preserve_inactive_velocity_tiles(u, v, w, u_out, v_out, w_out, active_tile_mask):
    """
    Copy unchanged velocity values for inactive tiles.
    """
    tile_i, tile_j, tile_k, i, j, k, nx, ny, nz = _active_tile_cell_indices(u.shape)

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

    u_out[i, j, k] = u[i, j, k]
    v_out[i, j, k] = v[i, j, k]
    w_out[i, j, k] = w[i, j, k]


@cuda.jit(cache=True)
def advect_velocity_semi_lagrangian(
    u, v, w, advected_u, advected_v, advected_w, dt, delta, active_tile_mask
):
    """
    Backtrace the velocity field once and store the purely advected values.
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


@cuda.jit(cache=True)
def update_velocity_maccormack(
    u,
    v,
    w,
    predictor_u,
    predictor_v,
    predictor_w,
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
    max_velocity_increment_factor,
    maccormack_factor,
    active_tile_mask,
):
    """
    CUDA kernel that updates velocity with a MacCormack-corrected
    semi-Lagrangian advection step.

    A forward semi-Lagrangian predictor is supplied in `predictor_*`. The
    corrector reverses that predictor, applies the standard MacCormack
    compensation term, clamps the result to the departure cell range, and then
    adds pressure, diffusion and external forces explicitly.
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

    # constants
    dt_over_delta = dt / delta
    diffusion_coeff = nu * dt / (delta * delta)
    force_coeff = dt / rho

    # load center values only once
    u_center = u[i, j, k]
    v_center = v[i, j, k]
    w_center = w[i, j, k]

    # Backwards trace
    # compute:
    # x_depart = x - dt * u
    # _backtrace_position does this in 3 substeps to allow for curvature in the path
    x_depart, y_depart, z_depart = _backtrace_position(
        u,
        v,
        w,
        float(i),
        float(j),
        float(k),
        dt_over_delta,
        nx,
        ny,
        nz,
    )
    # Forward trace from the departure point:
    # approximately:
    # x_forward = x_depart + dt * u(x_depart)
    x_forward, y_forward, z_forward = _forward_trace_position(
        u,
        v,
        w,
        float(i),
        float(j),
        float(k),
        dt_over_delta,
        nx,
        ny,
        nz,
    )

    advected_u = predictor_u[i, j, k]
    advected_v = predictor_v[i, j, k]
    advected_w = predictor_w[i, j, k]

    # find depart velocity
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
    # MacCormack correction
    # u_n+1 = u* + alpha(u_n-u**)
    # u_n+1 -- new velocity
    # u*    -- semi lagrangian advected field
    # alpha -- MacCormack factor
    # u_n   -- original velocity
    # u**   -- reversely advected velocity
    #          (without numerical inaccuracies, this would equal u_n again)
    corrected_u = advected_u + maccormack_factor * (u_center - reverse_u)
    corrected_v = advected_v + maccormack_factor * (v_center - reverse_v)
    corrected_w = advected_w + maccormack_factor * (w_center - reverse_w)

    # find backtraced coords
    x0, y0, z0, x1, y1, z1, _, _, _ = _prepare_trilinear_coords(
        x_depart, y_depart, z_depart, nx, ny, nz
    )
    # find the velocity vectors upper and lower bounds of neighbour cells at backtrace positions
    u_lower, u_upper = _sample_cell_extrema_inner(u, x0, y0, z0, x1, y1, z1)
    v_lower, v_upper = _sample_cell_extrema_inner(v, x0, y0, z0, x1, y1, z1)
    w_lower, w_upper = _sample_cell_extrema_inner(w, x0, y0, z0, x1, y1, z1)

    # clamping to bounds
    corrected_u = _clamp(corrected_u, u_lower, u_upper)
    corrected_v = _clamp(corrected_v, v_lower, v_upper)
    corrected_w = _clamp(corrected_w, w_lower, w_upper)

    # D_x = nu * dt * (u[i+1,j,k] - 2*u[i,j,k] + u[i-1,j,k]) / delta^2
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

    # u_sword = u_n+1 - D_x + Fx
    u_raw = corrected_u - diffusion_x + force_coeff * Fx[i, j, k]
    v_raw = corrected_v - diffusion_y + force_coeff * Fy[i, j, k]
    w_raw = corrected_w - diffusion_z + force_coeff * Fz[i, j, k]

    # limiter
    max_increment = max_velocity_increment_factor * delta / dt
    du = min(max(u_raw - u_center, -max_increment), max_increment)
    dv = min(max(v_raw - v_center, -max_increment), max_increment)
    dw = min(max(w_raw - w_center, -max_increment), max_increment)

    un[i, j, k] = u_center + du
    vn[i, j, k] = v_center + dv
    wn[i, j, k] = w_center + dw
