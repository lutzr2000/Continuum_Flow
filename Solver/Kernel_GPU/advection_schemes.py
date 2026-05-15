from numba import cuda


@cuda.jit(device=True, inline=True, cache=True)
def _clamp(value, lower, upper):
    """Clamp one scalar value to the inclusive `[lower, upper]` interval."""
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


@cuda.jit(device=True, inline=True, cache=True)
def _sample_trilinear(field, x, y, z, nx, ny, nz):
    """
    Sample one scalar field at fractional grid coordinates with trilinear interpolation.

    The sample position is clamped to the valid array extent before the eight
    surrounding cell values are blended.
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
def _backtrace_position(u, v, w, x_start, y_start, z_start, dt_over_delta, nx, ny, nz):
    """
    Backtrace one particle position through the velocity field with two substeps.

    The helper is used by the semi-Lagrangian and MacCormack advection passes
    to estimate the departure point in grid coordinates.
    """
    substep_dt = dt_over_delta * 0.25
    x_pos = x_start
    y_pos = y_start
    z_pos = z_start

    for _ in range(3):
        u_sample = _sample_trilinear(u, x_pos, y_pos, z_pos, nx, ny, nz)
        v_sample = _sample_trilinear(v, x_pos, y_pos, z_pos, nx, ny, nz)
        w_sample = _sample_trilinear(w, x_pos, y_pos, z_pos, nx, ny, nz)

        x_pos -= substep_dt * u_sample
        y_pos -= substep_dt * v_sample
        z_pos -= substep_dt * w_sample

    return x_pos, y_pos, z_pos


@cuda.jit(device=True, inline=True, cache=True)
def _forward_trace_position(u, v, w, x_start, y_start, z_start, dt_over_delta, nx, ny, nz):
    """Forward-trace one position by reusing the backtracer with a negated timestep."""
    return _backtrace_position(u, v, w, x_start, y_start, z_start, -dt_over_delta, nx, ny, nz)


@cuda.jit(device=True, inline=True, cache=True)
def _sample_cell_extrema(field, x, y, z):
    """
    Return the minimum and maximum corner values of the cell surrounding a sample point.

    MacCormack uses these bounds as a limiter so the corrected value stays
    inside the local source range and avoids new extrema.
    """
    nx, ny, nz = field.shape

    x = _clamp(x, 0.0, float(nx - 1))
    y = _clamp(y, 0.0, float(ny - 1))
    z = _clamp(z, 0.0, float(nz - 1))

    x0 = int(x)
    y0 = int(y)
    z0 = int(z)

    x1 = min(x0 + 1, nx - 1)
    y1 = min(y0 + 1, ny - 1)
    z1 = min(z0 + 1, nz - 1)

    c000 = field[x0, y0, z0]
    c100 = field[x1, y0, z0]
    c010 = field[x0, y1, z0]
    c110 = field[x1, y1, z0]
    c001 = field[x0, y0, z1]
    c101 = field[x1, y0, z1]
    c011 = field[x0, y1, z1]
    c111 = field[x1, y1, z1]

    lower = min(min(min(c000, c100), min(c010, c110)), min(min(c001, c101), min(c011, c111)))
    upper = max(max(max(c000, c100), max(c010, c110)), max(max(c001, c101), max(c011, c111)))
    return lower, upper


@cuda.jit(cache=True)
def advect_velocity_semi_lagrangian(u, v, w, advected_u, advected_v, advected_w, dt, delta):
    """Backtrace the velocity field once and store the purely advected values."""
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape
    if i >= nx or j >= ny or k >= nz:
        return

    x_depart, y_depart, z_depart = _backtrace_position(
        u, v, w,
        float(i), float(j), float(k),
        dt / delta,
        nx, ny, nz,
    )

    advected_u[i, j, k] = _sample_trilinear(u, x_depart, y_depart, z_depart, nx, ny, nz)
    advected_v[i, j, k] = _sample_trilinear(v, x_depart, y_depart, z_depart, nx, ny, nz)
    advected_w[i, j, k] = _sample_trilinear(w, x_depart, y_depart, z_depart, nx, ny, nz)


@cuda.jit(cache=True)
def update_velocity_maccormack(
    u, v, w, predictor_u, predictor_v, predictor_w,
    p, dt, Fx, Fy, Fz, un, vn, wn, delta, rho, nu,
    max_velocity_increment_factor, pressure_scale, maccormack_factor
):
    """
    CUDA kernel that updates velocity with a MacCormack-corrected
    semi-Lagrangian advection step.

    A forward semi-Lagrangian predictor is supplied in `predictor_*`. The
    corrector reverses that predictor, applies the standard MacCormack
    compensation term, clamps the result to the departure cell range, and then
    adds pressure, diffusion and external forces explicitly.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape
    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    dt_over_delta = dt / delta
    pressure_coeff = dt / (2.0 * rho * delta)
    diffusion_coeff = nu * dt / (delta * delta)
    force_coeff = dt / rho

    u_center = u[i, j, k]
    v_center = v[i, j, k]
    w_center = w[i, j, k]

    x_depart, y_depart, z_depart = _backtrace_position(
        u, v, w,
        float(i), float(j), float(k),
        dt_over_delta,
        nx, ny, nz,
    )
    x_forward, y_forward, z_forward = _forward_trace_position(
        u, v, w,
        float(i), float(j), float(k),
        dt_over_delta,
        nx, ny, nz,
    )

    advected_u = predictor_u[i, j, k]
    advected_v = predictor_v[i, j, k]
    advected_w = predictor_w[i, j, k]

    reverse_u = _sample_trilinear(predictor_u, x_forward, y_forward, z_forward, nx, ny, nz)
    reverse_v = _sample_trilinear(predictor_v, x_forward, y_forward, z_forward, nx, ny, nz)
    reverse_w = _sample_trilinear(predictor_w, x_forward, y_forward, z_forward, nx, ny, nz)

    corrected_u = advected_u + maccormack_factor * (u_center - reverse_u)
    corrected_v = advected_v + maccormack_factor * (v_center - reverse_v)
    corrected_w = advected_w + maccormack_factor * (w_center - reverse_w)

    u_lower, u_upper = _sample_cell_extrema(u, x_depart, y_depart, z_depart)
    v_lower, v_upper = _sample_cell_extrema(v, x_depart, y_depart, z_depart)
    w_lower, w_upper = _sample_cell_extrema(w, x_depart, y_depart, z_depart)

    corrected_u = _clamp(corrected_u, u_lower, u_upper)
    corrected_v = _clamp(corrected_v, v_lower, v_upper)
    corrected_w = _clamp(corrected_w, w_lower, w_upper)

    diffusion_x = diffusion_coeff * (
        (u[i + 1, j, k] - 2.0 * u_center + u[i - 1, j, k]) +
        (u[i, j + 1, k] - 2.0 * u_center + u[i, j - 1, k]) +
        (u[i, j, k + 1] - 2.0 * u_center + u[i, j, k - 1])
    )
    diffusion_y = diffusion_coeff * (
        (v[i + 1, j, k] - 2.0 * v_center + v[i - 1, j, k]) +
        (v[i, j + 1, k] - 2.0 * v_center + v[i, j - 1, k]) +
        (v[i, j, k + 1] - 2.0 * v_center + v[i, j, k - 1])
    )
    diffusion_z = diffusion_coeff * (
        (w[i + 1, j, k] - 2.0 * w_center + w[i - 1, j, k]) +
        (w[i, j + 1, k] - 2.0 * w_center + w[i, j - 1, k]) +
        (w[i, j, k + 1] - 2.0 * w_center + w[i, j, k - 1])
    )

    pressure_gradient_x = pressure_scale * pressure_coeff * (p[i + 1, j, k] - p[i - 1, j, k])
    pressure_gradient_y = pressure_scale * pressure_coeff * (p[i, j + 1, k] - p[i, j - 1, k])
    pressure_gradient_z = pressure_scale * pressure_coeff * (p[i, j, k + 1] - p[i, j, k - 1])

    u_raw = corrected_u - pressure_gradient_x + diffusion_x + force_coeff * Fx[i, j, k]
    v_raw = corrected_v - pressure_gradient_y + diffusion_y + force_coeff * Fy[i, j, k]
    w_raw = corrected_w - pressure_gradient_z + diffusion_z + force_coeff * Fz[i, j, k]

    max_increment = max_velocity_increment_factor * delta / dt
    du = min(max(u_raw - u_center, -max_increment), max_increment)
    dv = min(max(v_raw - v_center, -max_increment), max_increment)
    dw = min(max(w_raw - w_center, -max_increment), max_increment)

    un[i, j, k] = u_center + du
    vn[i, j, k] = v_center + dv
    wn[i, j, k] = w_center + dw
