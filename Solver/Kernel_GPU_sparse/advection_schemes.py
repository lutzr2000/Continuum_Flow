from numba import cuda

import Solver.Kernel_GPU_sparse.sparse_managment as sparse_managment
from Solver.Kernel_GPU_sparse.vorticity import apply_vorticity_confinement
import Solver.Kernel_GPU_sparse.kernel_config as kernel_config

tile_size = kernel_config.TILE_SIZE


@cuda.jit(device=True, inline=True, cache=True)
def buoyancy_approximation(
    T,
    tile_map,
    i,
    j,
    k,
    buoyancy_factor,
    t_reference,
):
    """
    computes the buoyancy force in z-direction with the Boussinesq approximation on the GPU.
    """
    g = 9.81

    temperature = _sample_sparse_cell(T, tile_map, i, j, k, t_reference)

    return g * buoyancy_factor * (temperature - t_reference)


@cuda.jit(device=True, inline=True, cache=True)
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

        axis_len = (ax * ax + ay * ay + az * az) ** 0.5
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

        t_len = (tx * tx + ty * ty + tz * tz) ** 0.5
        if t_len <= 1e-8:
            continue

        tx /= t_len
        ty /= t_len
        tz /= t_len

        dist = dist_sq**0.5
        falloff = 1.0 - dist / radius

        Fx += strength * falloff * tx
        Fy += strength * falloff * ty
        Fz += strength * falloff * tz

    return Fx, Fy, Fz


@cuda.jit(device=True, inline=True, cache=True)
def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


@cuda.jit(device=True, inline=True, cache=True)
def _lerp(a, b, t):
    return a + t * (b - a)


@cuda.jit(device=True, inline=True, cache=True)
def _fast_floor(x):
    i = int(x)
    if x < float(i):
        return i - 1
    return i


@cuda.jit(device=True, inline=True, cache=True)
def _hash_noise_3d(ix, iy, iz, seed):
    n = ix * 15731 + iy * 789221 + iz * 1376312589 + seed * 1013
    n = (n << 13) ^ n
    nn = n * (n * n * 15731 + 789221) + 1376312589
    nn = nn & 0x7FFFFFFF
    return float(nn) / 1073741824.0 - 1.0  # [-1, 1]


@cuda.jit(device=True, inline=True, cache=True)
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


@cuda.jit(device=True, inline=True, cache=True)
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


@cuda.jit(device=True, inline=True, cache=True)
def _sample_sparse_cell(field, tile_map, i, j, k, default_value):
    tile_i = i // tile_size
    tile_j = j // tile_size
    tile_k = k // tile_size

    tile_index = tile_map[tile_i, tile_j, tile_k]
    if tile_index == -1:
        return default_value

    local_i = i - tile_i * tile_size
    local_j = j - tile_j * tile_size
    local_k = k - tile_k * tile_size

    return field[tile_index, local_i, local_j, local_k]


@cuda.jit(device=True, inline=True, cache=True)
def _sample_trilinear_inner_sparse(
    field, tile_map, x0, y0, z0, x1, y1, z1, tx, ty, tz, default_value
):
    c000 = _sample_sparse_cell(field, tile_map, x0, y0, z0, default_value)
    c100 = _sample_sparse_cell(field, tile_map, x1, y0, z0, default_value)
    c010 = _sample_sparse_cell(field, tile_map, x0, y1, z0, default_value)
    c110 = _sample_sparse_cell(field, tile_map, x1, y1, z0, default_value)
    c001 = _sample_sparse_cell(field, tile_map, x0, y0, z1, default_value)
    c101 = _sample_sparse_cell(field, tile_map, x1, y0, z1, default_value)
    c011 = _sample_sparse_cell(field, tile_map, x0, y1, z1, default_value)
    c111 = _sample_sparse_cell(field, tile_map, x1, y1, z1, default_value)

    c00 = c000 + tx * (c100 - c000)
    c10 = c010 + tx * (c110 - c010)
    c01 = c001 + tx * (c101 - c001)
    c11 = c011 + tx * (c111 - c011)

    c0 = c00 + ty * (c10 - c00)
    c1 = c01 + ty * (c11 - c01)
    return c0 + tz * (c1 - c0)


@cuda.jit(device=True, inline=True, cache=True)
def _sample_cell_extrema_inner_sparse(
    field, tile_map, x0, y0, z0, x1, y1, z1, default_value
):
    c000 = _sample_sparse_cell(field, tile_map, x0, y0, z0, default_value)
    c100 = _sample_sparse_cell(field, tile_map, x1, y0, z0, default_value)
    c010 = _sample_sparse_cell(field, tile_map, x0, y1, z0, default_value)
    c110 = _sample_sparse_cell(field, tile_map, x1, y1, z0, default_value)
    c001 = _sample_sparse_cell(field, tile_map, x0, y0, z1, default_value)
    c101 = _sample_sparse_cell(field, tile_map, x1, y0, z1, default_value)
    c011 = _sample_sparse_cell(field, tile_map, x0, y1, z1, default_value)
    c111 = _sample_sparse_cell(field, tile_map, x1, y1, z1, default_value)

    lower = min(
        min(min(c000, c100), min(c010, c110)),
        min(min(c001, c101), min(c011, c111)),
    )
    upper = max(
        max(max(c000, c100), max(c010, c110)),
        max(max(c001, c101), max(c011, c111)),
    )
    return lower, upper


@cuda.jit(device=True, inline=True, cache=True)
def _sample_trilinear_vec3_sparse(
    field_x,
    field_y,
    field_z,
    tile_map,
    x,
    y,
    z,
    nx,
    ny,
    nz,
    default_x,
    default_y,
    default_z,
):
    x0, y0, z0, x1, y1, z1, tx, ty, tz = _prepare_trilinear_coords(x, y, z, nx, ny, nz)

    sample_x = _sample_trilinear_inner_sparse(
        field_x, tile_map, x0, y0, z0, x1, y1, z1, tx, ty, tz, default_x
    )
    sample_y = _sample_trilinear_inner_sparse(
        field_y, tile_map, x0, y0, z0, x1, y1, z1, tx, ty, tz, default_y
    )
    sample_z = _sample_trilinear_inner_sparse(
        field_z, tile_map, x0, y0, z0, x1, y1, z1, tx, ty, tz, default_z
    )
    return sample_x, sample_y, sample_z


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
def _backtrace_position_sparse(
    u, v, w, tile_map, x_start, y_start, z_start, dt_over_delta, nx, ny, nz
):
    n_substeps = 1
    substep_dt = dt_over_delta / n_substeps
    x_pos = x_start
    y_pos = y_start
    z_pos = z_start

    for _ in range(n_substeps):
        u_sample, v_sample, w_sample = _sample_trilinear_vec3_sparse(
            u, v, w, tile_map, x_pos, y_pos, z_pos, nx, ny, nz, 0.0, 0.0, 0.0
        )
        x_pos -= substep_dt * u_sample
        y_pos -= substep_dt * v_sample
        z_pos -= substep_dt * w_sample

    return x_pos, y_pos, z_pos


@cuda.jit(device=True, inline=True, cache=True)
def _forward_trace_position_sparse(
    u, v, w, tile_map, x_start, y_start, z_start, dt_over_delta, nx, ny, nz
):
    return _backtrace_position_sparse(
        u, v, w, tile_map, x_start, y_start, z_start, -dt_over_delta, nx, ny, nz
    )


@cuda.jit(cache=True)
def advect_velocity_semi_lagrangian(
    u,
    v,
    w,
    advected_u,
    advected_v,
    advected_w,
    dt,
    delta,
    tile_map,
    nx,
    ny,
    nz,
):
    (
        tile_i,
        tile_j,
        tile_k,
        local_i,
        local_j,
        local_k,
        i,
        j,
        k,
        _,
        _,
        _,
    ) = sparse_managment.tile_to_index((nx, ny, nz))

    if i >= nx or j >= ny or k >= nz:
        return

    tile_index = tile_map[tile_i, tile_j, tile_k]
    if tile_index == -1:
        return

    x_depart, y_depart, z_depart = _backtrace_position_sparse(
        u, v, w, tile_map, float(i), float(j), float(k), dt / delta, nx, ny, nz
    )

    sampled_u, sampled_v, sampled_w = _sample_trilinear_vec3_sparse(
        u, v, w, tile_map, x_depart, y_depart, z_depart, nx, ny, nz, 0.0, 0.0, 0.0
    )

    advected_u[tile_index, local_i, local_j, local_k] = sampled_u
    advected_v[tile_index, local_i, local_j, local_k] = sampled_v
    advected_w[tile_index, local_i, local_j, local_k] = sampled_w


@cuda.jit(cache=True)
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
    tile_map,
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
    nx,
    ny,
    nz,
):
    """
    CUDA kernel that updates velocity with a MacCormack-corrected
    semi-Lagrangian advection step on sparse velocity pools.
    """
    (
        tile_i,
        tile_j,
        tile_k,
        local_i,
        local_j,
        local_k,
        i,
        j,
        k,
        _,
        _,
        _,
    ) = sparse_managment.tile_to_index((nx, ny, nz))

    if i >= nx or j >= ny or k >= nz:
        return

    tile_index = tile_map[tile_i, tile_j, tile_k]
    if tile_index == -1:
        return

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    Fx = 0.0
    Fy = 0.0
    Fz = 0.0

    dt_over_delta = dt / delta
    diffusion_coeff = nu * dt / (delta * delta)
    force_coeff = dt / rho

    u_center = u[tile_index, local_i, local_j, local_k]
    v_center = v[tile_index, local_i, local_j, local_k]
    w_center = w[tile_index, local_i, local_j, local_k]

    x_depart, y_depart, z_depart = _backtrace_position_sparse(
        u,
        v,
        w,
        tile_map,
        float(i),
        float(j),
        float(k),
        dt_over_delta,
        nx,
        ny,
        nz,
    )

    x_forward, y_forward, z_forward = _forward_trace_position_sparse(
        u,
        v,
        w,
        tile_map,
        x_depart,
        y_depart,
        z_depart,
        dt_over_delta,
        nx,
        ny,
        nz,
    )

    advected_u = predictor_u[tile_index, local_i, local_j, local_k]
    advected_v = predictor_v[tile_index, local_i, local_j, local_k]
    advected_w = predictor_w[tile_index, local_i, local_j, local_k]

    reverse_u, reverse_v, reverse_w = _sample_trilinear_vec3_sparse(
        predictor_u,
        predictor_v,
        predictor_w,
        tile_map,
        x_forward,
        y_forward,
        z_forward,
        nx,
        ny,
        nz,
        0.0,
        0.0,
        0.0,
    )

    corrected_u = advected_u + 0.5 * (u_center - reverse_u)
    corrected_v = advected_v + 0.5 * (v_center - reverse_v)
    corrected_w = advected_w + 0.5 * (w_center - reverse_w)

    x0, y0, z0, x1, y1, z1, _, _, _ = _prepare_trilinear_coords(
        x_depart, y_depart, z_depart, nx, ny, nz
    )

    u_lower, u_upper = _sample_cell_extrema_inner_sparse(
        u, tile_map, x0, y0, z0, x1, y1, z1, 0.0
    )
    v_lower, v_upper = _sample_cell_extrema_inner_sparse(
        v, tile_map, x0, y0, z0, x1, y1, z1, 0.0
    )
    w_lower, w_upper = _sample_cell_extrema_inner_sparse(
        w, tile_map, x0, y0, z0, x1, y1, z1, 0.0
    )

    corrected_u = _clamp(corrected_u, u_lower, u_upper)
    corrected_v = _clamp(corrected_v, v_lower, v_upper)
    corrected_w = _clamp(corrected_w, w_lower, w_upper)

    diffusion_x = diffusion_coeff * (
        (_sample_sparse_cell(u, tile_map, i + 1, j, k, 0.0) - 2.0 * u_center + _sample_sparse_cell(u, tile_map, i - 1, j, k, 0.0))
        + (_sample_sparse_cell(u, tile_map, i, j + 1, k, 0.0) - 2.0 * u_center + _sample_sparse_cell(u, tile_map, i, j - 1, k, 0.0))
        + (_sample_sparse_cell(u, tile_map, i, j, k + 1, 0.0) - 2.0 * u_center + _sample_sparse_cell(u, tile_map, i, j, k - 1, 0.0))
    )
    diffusion_y = diffusion_coeff * (
        (_sample_sparse_cell(v, tile_map, i + 1, j, k, 0.0) - 2.0 * v_center + _sample_sparse_cell(v, tile_map, i - 1, j, k, 0.0))
        + (_sample_sparse_cell(v, tile_map, i, j + 1, k, 0.0) - 2.0 * v_center + _sample_sparse_cell(v, tile_map, i, j - 1, k, 0.0))
        + (_sample_sparse_cell(v, tile_map, i, j, k + 1, 0.0) - 2.0 * v_center + _sample_sparse_cell(v, tile_map, i, j, k - 1, 0.0))
    )
    diffusion_z = diffusion_coeff * (
        (_sample_sparse_cell(w, tile_map, i + 1, j, k, 0.0) - 2.0 * w_center + _sample_sparse_cell(w, tile_map, i - 1, j, k, 0.0))
        + (_sample_sparse_cell(w, tile_map, i, j + 1, k, 0.0) - 2.0 * w_center + _sample_sparse_cell(w, tile_map, i, j - 1, k, 0.0))
        + (_sample_sparse_cell(w, tile_map, i, j, k + 1, 0.0) - 2.0 * w_center + _sample_sparse_cell(w, tile_map, i, j, k - 1, 0.0))
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
            delta,
            vorticity_strength,
            tile_map,
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

    Fx += fx_const * 0.1
    Fy += fy_const * 0.1
    Fz += fz_const * 0.1

    Fz += buoyancy_approximation(
        temperature,
        tile_map,
        i,
        j,
        k,
        buoyancy_factor,
        t_reference,
    )

    u_raw = corrected_u + diffusion_x + force_coeff * Fx
    v_raw = corrected_v + diffusion_y + force_coeff * Fy
    w_raw = corrected_w + diffusion_z + force_coeff * Fz

    un[tile_index, local_i, local_j, local_k] = u_raw
    vn[tile_index, local_i, local_j, local_k] = v_raw
    wn[tile_index, local_i, local_j, local_k] = w_raw
