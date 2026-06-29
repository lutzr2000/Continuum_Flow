import math
from numba import njit, prange

import Solver.Kernel_CPU.kernel_config as kernel_config


@njit(cache=True, inline="always")
def buoyancy_approximation(
    T,
    i,
    j,
    k,
    buoyancy_factor,
    t_reference,
):
    """
    Compute the buoyancy force in the z direction.
    """
    g = 9.81
    return g * buoyancy_factor * (T[i, j, k] - t_reference)


@njit(cache=True, inline="always")
def apply_swirl_forces(
    swirl_config,
    i,
    j,
    k,
    delta,
    origin_x,
    origin_y,
    origin_z,
):
    Fx = 0.0
    Fy = 0.0
    Fz = 0.0

    px = origin_x + float(i) * delta
    py = origin_y + float(j) * delta
    pz = origin_z + float(k) * delta

    for swirl_idx in range(swirl_config.shape[0]):
        strength = swirl_config[swirl_idx, 0]

        ox = swirl_config[swirl_idx, 1]
        oy = swirl_config[swirl_idx, 2]
        oz = swirl_config[swirl_idx, 3]

        ax = swirl_config[swirl_idx, 4]
        ay = swirl_config[swirl_idx, 5]
        az = swirl_config[swirl_idx, 6]

        radius = swirl_config[swirl_idx, 7]

        if radius <= 0.0 or strength == 0.0:
            continue

        axis_len = math.sqrt(ax * ax + ay * ay + az * az)
        if axis_len <= 1e-8:
            continue

        ax /= axis_len
        ay /= axis_len
        az /= axis_len

        rx = px - ox
        ry = py - oy
        rz = pz - oz

        projection = rx * ax + ry * ay + rz * az

        closest_x = ox + projection * ax
        closest_y = oy + projection * ay
        closest_z = oz + projection * az

        radial_x = px - closest_x
        radial_y = py - closest_y
        radial_z = pz - closest_z

        dist_sq = radial_x * radial_x + radial_y * radial_y + radial_z * radial_z
        radius_sq = radius * radius

        if dist_sq > radius_sq or dist_sq <= 1e-12:
            continue

        tx = ay * radial_z - az * radial_y
        ty = az * radial_x - ax * radial_z
        tz = ax * radial_y - ay * radial_x

        t_len = math.sqrt(tx * tx + ty * ty + tz * tz)
        if t_len <= 1e-8:
            continue

        tx /= t_len
        ty /= t_len
        tz /= t_len

        dist = math.sqrt(dist_sq)
        falloff = 1.0 - dist / radius

        Fx += strength * falloff * tx
        Fy += strength * falloff * ty
        Fz += strength * falloff * tz

    return Fx, Fy, Fz


@njit(cache=True, inline="always")
def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True, inline="always")
def _lerp(a, b, t):
    return a + t * (b - a)


@njit(cache=True, inline="always")
def _fast_floor(x):
    i = int(x)
    if x < float(i):
        return i - 1
    return i


@njit(cache=True, inline="always")
def _hash_noise_3d(ix, iy, iz, seed):
    n = ix * 15731 + iy * 789221 + iz * 1376312589 + seed * 1013
    n = (n << 13) ^ n
    nn = n * (n * n * 15731 + 789221) + 1376312589
    nn = nn & 0x7FFFFFFF
    return float(nn) / 1073741824.0 - 1.0


@njit(cache=True, inline="always")
def _value_noise_3d(x, y, z, seed):
    x0 = _fast_floor(x)
    y0 = _fast_floor(y)
    z0 = _fast_floor(z)

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    tx = _smoothstep(x - float(x0))
    ty = _smoothstep(y - float(y0))
    tz = _smoothstep(z - float(z0))

    c000 = _hash_noise_3d(x0, y0, z0, seed)
    c100 = _hash_noise_3d(x1, y0, z0, seed)
    c010 = _hash_noise_3d(x0, y1, z0, seed)
    c110 = _hash_noise_3d(x1, y1, z0, seed)

    c001 = _hash_noise_3d(x0, y0, z1, seed)
    c101 = _hash_noise_3d(x1, y0, z1, seed)
    c011 = _hash_noise_3d(x0, y1, z1, seed)
    c111 = _hash_noise_3d(x1, y1, z1, seed)

    x00 = _lerp(c000, c100, tx)
    x10 = _lerp(c010, c110, tx)
    x01 = _lerp(c001, c101, tx)
    x11 = _lerp(c011, c111, tx)

    y0v = _lerp(x00, x10, ty)
    y1v = _lerp(x01, x11, ty)

    return _lerp(y0v, y1v, tz)


@njit(cache=True, inline="always")
def apply_turbulence_forces(
    turbulence_config,
    i,
    j,
    k,
    delta,
    origin_x,
    origin_y,
    origin_z,
    t,
):
    Fx = 0.0
    Fy = 0.0
    Fz = 0.0

    px = origin_x + float(i) * delta
    py = origin_y + float(j) * delta
    pz = origin_z + float(k) * delta

    for turb_idx in range(turbulence_config.shape[0]):
        amplitude = turbulence_config[turb_idx, 0]
        scale = turbulence_config[turb_idx, 1]
        frequency = turbulence_config[turb_idx, 2]
        seed = int(turbulence_config[turb_idx, 3])

        if amplitude == 0.0 or scale <= 1e-8:
            continue

        inv_scale = 1.0 / scale
        time_offset = t * frequency

        x = px * inv_scale
        y = py * inv_scale
        z = pz * inv_scale + time_offset

        noise = _value_noise_3d(x, y, z, seed)

        Fx += amplitude * noise
        Fy += amplitude * noise
        Fz += amplitude * noise

    return Fx, Fy, Fz


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


@njit(cache=True, inline="always")
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


@njit(cache=True, inline="always")
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


@njit(cache=True, inline="always")
def _sample_trilinear_vec3(field_x, field_y, field_z, x, y, z, nx, ny, nz):
    """
    Sample three fields at one position with shared trilinear coordinates.
    """
    x0, y0, z0, x1, y1, z1, tx, ty, tz = _prepare_trilinear_coords(x, y, z, nx, ny, nz)

    sample_x = _sample_trilinear_inner(field_x, x0, y0, z0, x1, y1, z1, tx, ty, tz)
    sample_y = _sample_trilinear_inner(field_y, x0, y0, z0, x1, y1, z1, tx, ty, tz)
    sample_z = _sample_trilinear_inner(field_z, x0, y0, z0, x1, y1, z1, tx, ty, tz)
    return sample_x, sample_y, sample_z


@njit(cache=True, inline="always")
def _backtrace_position(u, v, w, x_start, y_start, z_start, dt_over_delta, nx, ny, nz):
    """
    Backtrace one particles position through the velocity field with n_substepds (hardcoded:3).

    The helper is used by the semi-Lagrangian and MacCormack advection passes
    to estimate the departure point in grid coordinates.
    """
    n_substeps = 1
    substep_dt = dt_over_delta * 1.0 / n_substeps
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


@njit(cache=True, inline="always")
def apply_vorticity_confinement(
    u,
    v,
    w,
    obstacle_mask,
    omega_magnitude,
    i,
    j,
    k,
    nx,
    ny,
    nz,
    delta,
    vorticity_strength,
):
    if (
        i < 2
        or j < 2
        or k < 2
        or i >= nx - 2
        or j >= ny - 2
        or k >= nz - 2
        or obstacle_mask[i, j, k]
    ):
        return 0.0, 0.0, 0.0

    half_inv_delta = 0.5 / delta

    grad_x = (omega_magnitude[i + 1, j, k] - omega_magnitude[i - 1, j, k]) * half_inv_delta
    grad_y = (omega_magnitude[i, j + 1, k] - omega_magnitude[i, j - 1, k]) * half_inv_delta
    grad_z = (omega_magnitude[i, j, k + 1] - omega_magnitude[i, j, k - 1]) * half_inv_delta

    grad_length = math.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z)

    if grad_length <= 1.0e-12:
        return 0.0, 0.0, 0.0

    nx_dir = grad_x / grad_length
    ny_dir = grad_y / grad_length
    nz_dir = grad_z / grad_length

    du_dy = (u[i, j + 1, k] - u[i, j - 1, k]) * half_inv_delta
    du_dz = (u[i, j, k + 1] - u[i, j, k - 1]) * half_inv_delta

    dv_dx = (v[i + 1, j, k] - v[i - 1, j, k]) * half_inv_delta
    dv_dz = (v[i, j, k + 1] - v[i, j, k - 1]) * half_inv_delta

    dw_dx = (w[i + 1, j, k] - w[i - 1, j, k]) * half_inv_delta
    dw_dy = (w[i, j + 1, k] - w[i, j - 1, k]) * half_inv_delta

    wx = dw_dy - dv_dz
    wy = du_dz - dw_dx
    wz = dv_dx - du_dy

    fx = vorticity_strength * (ny_dir * wz - nz_dir * wy)
    fy = vorticity_strength * (nz_dir * wx - nx_dir * wz)
    fz = vorticity_strength * (nx_dir * wy - ny_dir * wx)

    return fx, fy, fz


@njit(cache=True, inline="always")
def _is_active_tile(active_tile_mask, i, j, k):
    tile_i = i // kernel_config.ACTIVE_TILE_SIZE
    tile_j = j // kernel_config.ACTIVE_TILE_SIZE
    tile_k = k // kernel_config.ACTIVE_TILE_SIZE

    if (
        tile_i >= active_tile_mask.shape[0]
        or tile_j >= active_tile_mask.shape[1]
        or tile_k >= active_tile_mask.shape[2]
    ):
        return False

    return active_tile_mask[tile_i, tile_j, tile_k] != 0


@njit(cache=True, parallel=True)
def advect_velocity_semi_lagrangian(
    u,
    v,
    w,
    advected_u,
    advected_v,
    advected_w,
    dt,
    delta,
    active_tile_mask,
):
    """
    Backtrace the velocity field once and store the purely advected values.
    """
    nx, ny, nz = u.shape

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if not _is_active_tile(active_tile_mask, i, j, k):
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

                adv_u, adv_v, adv_w = _sample_trilinear_vec3(
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
                advected_u[i, j, k] = adv_u
                advected_v[i, j, k] = adv_v
                advected_w[i, j, k] = adv_w


@njit(cache=True, parallel=True)
def update_velocity_maccormack(
    u,
    v,
    w,
    obstacle_mask,
    predictor_u,
    predictor_v,
    predictor_w,
    dt,
    un,
    vn,
    wn,
    delta,
    rho,
    nu,
    vorticity_magnitude,
    vorticity_strength,
    temperature,
    buoyancy_factor,
    t_reference,
    active_tile_mask,
    fx_const,
    fy_const,
    fz_const,
    has_swirl_nodes,
    swirl_config,
    origin_x,
    origin_y,
    origin_z,
    has_turbulence_nodes,
    turbulence_config,
    t,
):
    """
    Update velocity with a MacCormack-corrected semi-Lagrangian step.
    """
    nx, ny, nz = u.shape

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if not _is_active_tile(active_tile_mask, i, j, k):
                    continue
                if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
                    continue

                Fx = 0.0
                Fy = 0.0
                Fz = 0.0

                dt_over_delta = dt / delta
                diffusion_coeff = nu * dt / (delta * delta)
                force_coeff = dt / rho

                u_center = u[i, j, k]
                v_center = v[i, j, k]
                w_center = w[i, j, k]

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

                corrected_u = advected_u + 0.5 * (u_center - reverse_u)
                corrected_v = advected_v + 0.5 * (v_center - reverse_v)
                corrected_w = advected_w + 0.5 * (w_center - reverse_w)

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

                if vorticity_strength > 0.0:
                    Fx, Fy, Fz = apply_vorticity_confinement(
                        u,
                        v,
                        w,
                        obstacle_mask,
                        vorticity_magnitude,
                        i,
                        j,
                        k,
                        nx,
                        ny,
                        nz,
                        delta,
                        vorticity_strength,
                    )

                if has_swirl_nodes:
                    swirl_fx, swirl_fy, swirl_fz = apply_swirl_forces(
                        swirl_config,
                        i,
                        j,
                        k,
                        delta,
                        origin_x,
                        origin_y,
                        origin_z,
                    )
                    Fx += swirl_fx
                    Fy += swirl_fy
                    Fz += swirl_fz

                if has_turbulence_nodes:
                    turb_fx, turb_fy, turb_fz = apply_turbulence_forces(
                        turbulence_config,
                        i,
                        j,
                        k,
                        delta,
                        origin_x,
                        origin_y,
                        origin_z,
                        t,
                    )
                    Fx += turb_fx
                    Fy += turb_fy
                    Fz += turb_fz

                Fx += fx_const * 10.0
                Fy += fy_const * 10.0
                Fz += fz_const * 10.0
                Fz += buoyancy_approximation(
                    temperature,
                    i,
                    j,
                    k,
                    buoyancy_factor,
                    t_reference,
                )

                un[i, j, k] = corrected_u + diffusion_x + force_coeff * Fx
                vn[i, j, k] = corrected_v + diffusion_y + force_coeff * Fy
                wn[i, j, k] = corrected_w + diffusion_z + force_coeff * Fz
