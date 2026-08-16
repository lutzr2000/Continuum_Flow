from numba import cuda

import Solver.Kernel_GPU.sparse_managment as sparse_managment

@cuda.jit(device=True, inline=True, cache=True)
def _sample_trilinear_inner_sparse(
    field, tile_map, x0, y0, z0, x1, y1, z1, tx, ty, tz, default_value
):
    c000 = sparse_managment.get_pool_value(field, tile_map, x0, y0, z0, default_value)
    c100 = sparse_managment.get_pool_value(field, tile_map, x1, y0, z0, default_value)
    c010 = sparse_managment.get_pool_value(field, tile_map, x0, y1, z0, default_value)
    c110 = sparse_managment.get_pool_value(field, tile_map, x1, y1, z0, default_value)
    c001 = sparse_managment.get_pool_value(field, tile_map, x0, y0, z1, default_value)
    c101 = sparse_managment.get_pool_value(field, tile_map, x1, y0, z1, default_value)
    c011 = sparse_managment.get_pool_value(field, tile_map, x0, y1, z1, default_value)
    c111 = sparse_managment.get_pool_value(field, tile_map, x1, y1, z1, default_value)

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
    c000 = sparse_managment.get_pool_value(field, tile_map, x0, y0, z0, default_value)
    c100 = sparse_managment.get_pool_value(field, tile_map, x1, y0, z0, default_value)
    c010 = sparse_managment.get_pool_value(field, tile_map, x0, y1, z0, default_value)
    c110 = sparse_managment.get_pool_value(field, tile_map, x1, y1, z0, default_value)
    c001 = sparse_managment.get_pool_value(field, tile_map, x0, y0, z1, default_value)
    c101 = sparse_managment.get_pool_value(field, tile_map, x1, y0, z1, default_value)
    c011 = sparse_managment.get_pool_value(field, tile_map, x0, y1, z1, default_value)
    c111 = sparse_managment.get_pool_value(field, tile_map, x1, y1, z1, default_value)

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
    u,
    v,
    w,
    tile_map,
    x_start,
    y_start,
    z_start,
    dt_over_delta,
    nx,
    ny,
    nz,
    u_initial,
    v_initial,
    w_initial,
):
    n_substeps = 1
    substep_dt = dt_over_delta / n_substeps
    x_pos = x_start
    y_pos = y_start
    z_pos = z_start

    for _ in range(n_substeps):
        u_sample, v_sample, w_sample = _sample_trilinear_vec3_sparse(
            u,
            v,
            w,
            tile_map,
            x_pos,
            y_pos,
            z_pos,
            nx,
            ny,
            nz,
            u_initial,
            v_initial,
            w_initial,
        )
        x_pos -= substep_dt * u_sample
        y_pos -= substep_dt * v_sample
        z_pos -= substep_dt * w_sample

    return x_pos, y_pos, z_pos


@cuda.jit(device=True, inline=True, cache=True)
def _forward_trace_position_sparse(
    u,
    v,
    w,
    tile_map,
    x_start,
    y_start,
    z_start,
    dt_over_delta,
    nx,
    ny,
    nz,
    u_initial,
    v_initial,
    w_initial,
):
    return _backtrace_position_sparse(
        u,
        v,
        w,
        tile_map,
        x_start,
        y_start,
        z_start,
        -dt_over_delta,
        nx,
        ny,
        nz,
        u_initial,
        v_initial,
        w_initial,
    )
