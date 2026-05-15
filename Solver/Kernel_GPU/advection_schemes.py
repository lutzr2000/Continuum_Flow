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
def _sample_trilinear(field, x, y, z):
    """
    Sample one scalar field at fractional grid coordinates with trilinear interpolation.

    The sample position is clamped to the valid array extent before the eight
    surrounding cell values are blended.
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
def _backtrace_position(u, v, w, x_start, y_start, z_start, dt_over_delta):
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
        u_sample = _sample_trilinear(u, x_pos, y_pos, z_pos)
        v_sample = _sample_trilinear(v, x_pos, y_pos, z_pos)
        w_sample = _sample_trilinear(w, x_pos, y_pos, z_pos)

        x_pos -= substep_dt * u_sample
        y_pos -= substep_dt * v_sample
        z_pos -= substep_dt * w_sample

    return x_pos, y_pos, z_pos


@cuda.jit(device=True, inline=True, cache=True)
def _forward_trace_position(u, v, w, x_start, y_start, z_start, dt_over_delta):
    """Forward-trace one position by reusing the backtracer with a negated timestep."""
    return _backtrace_position(u, v, w, x_start, y_start, z_start, -dt_over_delta)


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
    )

    advected_u[i, j, k] = _sample_trilinear(u, x_depart, y_depart, z_depart)
    advected_v[i, j, k] = _sample_trilinear(v, x_depart, y_depart, z_depart)
    advected_w[i, j, k] = _sample_trilinear(w, x_depart, y_depart, z_depart)


@cuda.jit(cache=True)
def update_velocity(
    u, v, w, p, dt, Fx, Fy, Fz, un, vn, wn, delta, rho, nu,
    max_velocity_increment_factor, pressure_scale
):
    """
    CUDA kernel that updates all three velocity components based on the
    momentum equation. Convection is done by first order upwind, diffusion with
    central differences.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        dt (float): timestep size
        Fx (device array): x-direction body force field
        Fy (device array): y-direction body force field
        Fz (device array): z-direction body force field
        un (device array): output array for updated x-velocity
        vn (device array): output array for updated y-velocity
        wn (device array): output array for updated z-velocity
        delta (float): grid spacing
        rho (float): density
        nu (float): kinematic viscosity
        max_velocity_increment_factor (float): maximum allowed per-step velocity
            change relative to delta / dt
        pressure_scale (float): scales the explicit pressure-gradient term
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

    #------------Upwinding-------------------
    if u_center >= 0.0:
        u_x_high = u_center
        u_x_low = u[i - 1, j, k]
        v_x_high = v_center
        v_x_low = v[i - 1, j, k]
        w_x_high = w_center
        w_x_low = w[i - 1, j, k]
    else:
        u_x_high = u[i + 1, j, k]
        u_x_low = u_center
        v_x_high = v[i + 1, j, k]
        v_x_low = v_center
        w_x_high = w[i + 1, j, k]
        w_x_low = w_center

    if v_center >= 0.0:
        u_y_high = u_center
        u_y_low = u[i, j - 1, k]
        v_y_high = v_center
        v_y_low = v[i, j - 1, k]
        w_y_high = w_center
        w_y_low = w[i, j - 1, k]

    else:
        u_y_high = u[i, j + 1, k]
        u_y_low = u_center
        v_y_high = v[i, j + 1, k]
        v_y_low = v_center
        w_y_high = w[i, j + 1, k]
        w_y_low = w_center

    if w_center >= 0.0:
        u_z_high = u_center
        u_z_low = u[i, j, k - 1]
        v_z_high = v_center
        v_z_low = v[i, j, k - 1]
        w_z_high = w_center
        w_z_low = w[i, j, k - 1]

    else:
        u_z_high = u[i, j, k + 1]
        u_z_low = u_center
        v_z_high = v[i, j, k + 1]
        v_z_low = v_center
        w_z_high = w[i, j, k + 1]
        w_z_low = w_center

    #------------Convection-------------------
    convection_x = dt_over_delta * (
        u_center * (u_x_high - u_x_low) +
        v_center * (u_y_high - u_y_low) +
        w_center * (u_z_high - u_z_low)
    )

    convection_y = dt_over_delta * (
        u_center * (v_x_high - v_x_low) +
        v_center * (v_y_high - v_y_low) +
        w_center * (v_z_high - v_z_low)
    )

    convection_z = dt_over_delta * (
        u_center * (w_x_high - w_x_low) +
        v_center * (w_y_high - w_y_low) +
        w_center * (w_z_high - w_z_low)
    )

    #------------Diffusion-------------------
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

    #------------Pressure-------------------
    pressure_gradient_x = pressure_scale * pressure_coeff * (p[i + 1, j, k] - p[i - 1, j, k])
    pressure_gradient_y = pressure_scale * pressure_coeff * (p[i, j + 1, k] - p[i, j - 1, k])
    pressure_gradient_z = pressure_scale * pressure_coeff * (p[i, j, k + 1] - p[i, j, k - 1])

    #------------Update-------------------
    u_raw = u_center - convection_x - pressure_gradient_x + diffusion_x + force_coeff * Fx[i, j, k]
    v_raw = v_center - convection_y - pressure_gradient_y + diffusion_y + force_coeff * Fy[i, j, k]
    w_raw = w_center - convection_z - pressure_gradient_z + diffusion_z + force_coeff * Fz[i, j, k]

    max_increment = max_velocity_increment_factor * delta / dt
    du = min(max(u_raw - u_center, -max_increment), max_increment)
    dv = min(max(v_raw - v_center, -max_increment), max_increment)
    dw = min(max(w_raw - w_center, -max_increment), max_increment)

    un[i, j, k] = u_center + du
    vn[i, j, k] = v_center + dv
    wn[i, j, k] = w_center + dw


@cuda.jit(cache=True)
def update_velocity_second_order_upwind(
    u, v, w, p, dt, Fx, Fy, Fz, un, vn, wn, delta, rho, nu,
    max_velocity_increment_factor, pressure_scale
):
    """
    CUDA kernel that updates all three velocity components based on the
    momentum equation. Convection is done by second order upwind where enough
    stencil points exist, otherwise first order upwind is used near boundaries.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        dt (float): timestep size
        Fx (device array): x-direction body force field
        Fy (device array): y-direction body force field
        Fz (device array): z-direction body force field
        un (device array): output array for updated x-velocity
        vn (device array): output array for updated y-velocity
        wn (device array): output array for updated z-velocity
        delta (float): grid spacing
        rho (float): density
        nu (float): kinematic viscosity
        max_velocity_increment_factor (float): maximum allowed per-step velocity
            change relative to delta / dt
        pressure_scale (float): scales the explicit pressure-gradient term
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

    has_im2 = i >= 2
    has_ip2 = i < nx - 2
    has_jm2 = j >= 2
    has_jp2 = j < ny - 2
    has_km2 = k >= 2
    has_kp2 = k < nz - 2

    # Use a second-order upwind stencil where available and fall back to the
    # original first-order stencil one cell away from the domain boundary.
    if u_center >= 0.0:
        if has_im2:
            du_dx_term = 0.5 * (3.0 * u_center - 4.0 * u[i - 1, j, k] + u[i - 2, j, k])
            dv_dx_term = 0.5 * (3.0 * v_center - 4.0 * v[i - 1, j, k] + v[i - 2, j, k])
            dw_dx_term = 0.5 * (3.0 * w_center - 4.0 * w[i - 1, j, k] + w[i - 2, j, k])
        else:
            du_dx_term = u_center - u[i - 1, j, k]
            dv_dx_term = v_center - v[i - 1, j, k]
            dw_dx_term = w_center - w[i - 1, j, k]
    else:
        if has_ip2:
            du_dx_term = 0.5 * (-3.0 * u_center + 4.0 * u[i + 1, j, k] - u[i + 2, j, k])
            dv_dx_term = 0.5 * (-3.0 * v_center + 4.0 * v[i + 1, j, k] - v[i + 2, j, k])
            dw_dx_term = 0.5 * (-3.0 * w_center + 4.0 * w[i + 1, j, k] - w[i + 2, j, k])
        else:
            du_dx_term = u[i + 1, j, k] - u_center
            dv_dx_term = v[i + 1, j, k] - v_center
            dw_dx_term = w[i + 1, j, k] - w_center

    if v_center >= 0.0:
        if has_jm2:
            du_dy_term = 0.5 * (3.0 * u_center - 4.0 * u[i, j - 1, k] + u[i, j - 2, k])
            dv_dy_term = 0.5 * (3.0 * v_center - 4.0 * v[i, j - 1, k] + v[i, j - 2, k])
            dw_dy_term = 0.5 * (3.0 * w_center - 4.0 * w[i, j - 1, k] + w[i, j - 2, k])
        else:
            du_dy_term = u_center - u[i, j - 1, k]
            dv_dy_term = v_center - v[i, j - 1, k]
            dw_dy_term = w_center - w[i, j - 1, k]
    else:
        if has_jp2:
            du_dy_term = 0.5 * (-3.0 * u_center + 4.0 * u[i, j + 1, k] - u[i, j + 2, k])
            dv_dy_term = 0.5 * (-3.0 * v_center + 4.0 * v[i, j + 1, k] - v[i, j + 2, k])
            dw_dy_term = 0.5 * (-3.0 * w_center + 4.0 * w[i, j + 1, k] - w[i, j + 2, k])
        else:
            du_dy_term = u[i, j + 1, k] - u_center
            dv_dy_term = v[i, j + 1, k] - v_center
            dw_dy_term = w[i, j + 1, k] - w_center

    if w_center >= 0.0:
        if has_km2:
            du_dz_term = 0.5 * (3.0 * u_center - 4.0 * u[i, j, k - 1] + u[i, j, k - 2])
            dv_dz_term = 0.5 * (3.0 * v_center - 4.0 * v[i, j, k - 1] + v[i, j, k - 2])
            dw_dz_term = 0.5 * (3.0 * w_center - 4.0 * w[i, j, k - 1] + w[i, j, k - 2])
        else:
            du_dz_term = u_center - u[i, j, k - 1]
            dv_dz_term = v_center - v[i, j, k - 1]
            dw_dz_term = w_center - w[i, j, k - 1]
    else:
        if has_kp2:
            du_dz_term = 0.5 * (-3.0 * u_center + 4.0 * u[i, j, k + 1] - u[i, j, k + 2])
            dv_dz_term = 0.5 * (-3.0 * v_center + 4.0 * v[i, j, k + 1] - v[i, j, k + 2])
            dw_dz_term = 0.5 * (-3.0 * w_center + 4.0 * w[i, j, k + 1] - w[i, j, k + 2])
        else:
            du_dz_term = u[i, j, k + 1] - u_center
            dv_dz_term = v[i, j, k + 1] - v_center
            dw_dz_term = w[i, j, k + 1] - w_center

    convection_x = dt_over_delta * (
        u_center * du_dx_term +
        v_center * du_dy_term +
        w_center * du_dz_term
    )

    convection_y = dt_over_delta * (
        u_center * dv_dx_term +
        v_center * dv_dy_term +
        w_center * dv_dz_term
    )

    convection_z = dt_over_delta * (
        u_center * dw_dx_term +
        v_center * dw_dy_term +
        w_center * dw_dz_term
    )

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

    u_raw = u_center - convection_x - pressure_gradient_x + diffusion_x + force_coeff * Fx[i, j, k]
    v_raw = v_center - convection_y - pressure_gradient_y + diffusion_y + force_coeff * Fy[i, j, k]
    w_raw = w_center - convection_z - pressure_gradient_z + diffusion_z + force_coeff * Fz[i, j, k]

    max_increment = max_velocity_increment_factor * delta / dt
    du = min(max(u_raw - u_center, -max_increment), max_increment)
    dv = min(max(v_raw - v_center, -max_increment), max_increment)
    dw = min(max(w_raw - w_center, -max_increment), max_increment)

    un[i, j, k] = u_center + du
    vn[i, j, k] = v_center + dv
    wn[i, j, k] = w_center + dw


@cuda.jit(cache=True)
def update_velocity_semi_lagrangian(
    u, v, w, p, dt, Fx, Fy, Fz, un, vn, wn, delta, rho, nu,
    max_velocity_increment_factor, pressure_scale
):
    """
    CUDA kernel that updates velocity with semi-Lagrangian advection.

    The velocity at each cell is backtraced through the old velocity field and
    reconstructed with trilinear interpolation. Pressure, diffusion and body
    forces are then applied explicitly at the current cell.
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
    )

    advected_u = _sample_trilinear(u, x_depart, y_depart, z_depart)
    advected_v = _sample_trilinear(v, x_depart, y_depart, z_depart)
    advected_w = _sample_trilinear(w, x_depart, y_depart, z_depart)

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

    u_raw = advected_u - pressure_gradient_x + diffusion_x + force_coeff * Fx[i, j, k]
    v_raw = advected_v - pressure_gradient_y + diffusion_y + force_coeff * Fy[i, j, k]
    w_raw = advected_w - pressure_gradient_z + diffusion_z + force_coeff * Fz[i, j, k]

    max_increment = max_velocity_increment_factor * delta / dt
    du = min(max(u_raw - u_center, -max_increment), max_increment)
    dv = min(max(v_raw - v_center, -max_increment), max_increment)
    dw = min(max(w_raw - w_center, -max_increment), max_increment)

    un[i, j, k] = u_center + du
    vn[i, j, k] = v_center + dv
    wn[i, j, k] = w_center + dw


@cuda.jit(cache=True)
def update_velocity_maccormack(
    u, v, w, predictor_u, predictor_v, predictor_w,
    p, dt, Fx, Fy, Fz, un, vn, wn, delta, rho, nu,
    max_velocity_increment_factor, pressure_scale
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
    )
    x_forward, y_forward, z_forward = _forward_trace_position(
        u, v, w,
        float(i), float(j), float(k),
        dt_over_delta,
    )

    advected_u = predictor_u[i, j, k]
    advected_v = predictor_v[i, j, k]
    advected_w = predictor_w[i, j, k]

    reverse_u = _sample_trilinear(predictor_u, x_forward, y_forward, z_forward)
    reverse_v = _sample_trilinear(predictor_v, x_forward, y_forward, z_forward)
    reverse_w = _sample_trilinear(predictor_w, x_forward, y_forward, z_forward)

    corrected_u = advected_u + 0.25 * (u_center - reverse_u)
    corrected_v = advected_v + 0.25 * (v_center - reverse_v)
    corrected_w = advected_w + 0.25 * (w_center - reverse_w)

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
