from numba import cuda
import numpy as np
import Solver.Kernel_GPU.kernel_config as kernel_config
import Solver.Kernel_GPU.sparse_managment as sparse_managment
import Solver.Kernel_GPU.Boundary_Conditions.domain_bc as BC

GPU_FIELD_DTYPE = kernel_config.GPU_FIELD_DTYPE

def create_multigrid_levels(shape, delta, min_size=8):
    p_levels = []
    b_levels = []
    delta_levels = []
    zero_levels = []

    nx, ny, nz = shape
    level = 0

    while nx >= min_size and ny >= min_size and nz >= min_size:
        level_shape = (nx, ny, nz)

        p_levels.append(cuda.device_array(level_shape, dtype=GPU_FIELD_DTYPE))
        b_levels.append(cuda.device_array(level_shape, dtype=GPU_FIELD_DTYPE))
        zero_levels.append(cuda.to_device(np.zeros(level_shape, dtype=GPU_FIELD_DTYPE)))
        delta_levels.append(delta * (2**level))

        nx = (nx + 1) // 2
        ny = (ny + 1) // 2
        nz = (nz + 1) // 2
        level += 1

    return p_levels, b_levels, delta_levels, zero_levels


@cuda.jit(cache=True)
def mg_restrict_residual_8cell(p, b, coarse_b, delta, nx, ny, nz):
    I, J, K = cuda.grid(3)

    cnx, cny, cnz = coarse_b.shape

    if I >= cnx or J >= cny or K >= cnz:
        return

    inv_delta2 = 1.0 / (delta * delta)

    i0 = 2 * I
    j0 = 2 * J
    k0 = 2 * K

    s = 0.0
    count = 0.0

    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i = i0 + di
                j = j0 + dj
                k = k0 + dk

                if (
                    i >= 1
                    and j >= 1
                    and k >= 1
                    and i < nx - 1
                    and j < ny - 1
                    and k < nz - 1
                ):
                    lap = (
                        p[i + 1, j, k]
                        + p[i - 1, j, k]
                        + p[i, j + 1, k]
                        + p[i, j - 1, k]
                        + p[i, j, k + 1]
                        + p[i, j, k - 1]
                        - 6.0 * p[i, j, k]
                    ) * inv_delta2

                    residual = b[i, j, k] - lap

                    s += residual
                    count += 1.0

    if count > 0.0:
        coarse_b[I, J, K] = s / count
    else:
        coarse_b[I, J, K] = 0.0


@cuda.jit(cache=True)
def mg_prolongate_add_nearest_sparse_level0(coarse_e, fine_p, tile_map, field_shape):
    I, J, K = cuda.grid(3)
    cnx, cny, cnz = coarse_e.shape
    fnx, fny, fnz = field_shape

    if I >= cnx or J >= cny or K >= cnz:
        return

    e = 0.25 * coarse_e[I, J, K]

    i0 = 2 * I
    j0 = 2 * J
    k0 = 2 * K

    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i = i0 + di
                j = j0 + dj
                k = k0 + dk

                if i < fnx and j < fny and k < fnz:
                    tile_i = i // kernel_config.TILE_SIZE
                    tile_j = j // kernel_config.TILE_SIZE
                    tile_k = k // kernel_config.TILE_SIZE
                    tile_index = tile_map[tile_i, tile_j, tile_k]
                    if tile_index == -1:
                        continue
                    local_i = i - tile_i * kernel_config.TILE_SIZE
                    local_j = j - tile_j * kernel_config.TILE_SIZE
                    local_k = k - tile_k * kernel_config.TILE_SIZE
                    fine_p[tile_index, local_i, local_j, local_k] += e


@cuda.jit(cache=True)
def mg_restrict_8cell(fine_r, coarse_b):
    I, J, K = cuda.grid(3)
    cnx, cny, cnz = coarse_b.shape
    fnx, fny, fnz = fine_r.shape

    if I >= cnx or J >= cny or K >= cnz:
        return

    i0 = 2 * I
    j0 = 2 * J
    k0 = 2 * K

    s = 0.0
    count = 0.0

    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i = i0 + di
                j = j0 + dj
                k = k0 + dk

                if i < fnx and j < fny and k < fnz:
                    s += fine_r[i, j, k]
                    count += 1.0

    if count > 0.0:
        coarse_b[I, J, K] = s / count
    else:
        coarse_b[I, J, K] = 0.0


@cuda.jit(cache=True)
def mg_restrict_residual_8cell_sparse_level0(p, b, coarse_b, delta, tile_map, nx, ny, nz):
    I, J, K = cuda.grid(3)

    cnx, cny, cnz = coarse_b.shape

    if I >= cnx or J >= cny or K >= cnz:
        return

    inv_delta2 = 1.0 / (delta * delta)

    i0 = 2 * I
    j0 = 2 * J
    k0 = 2 * K

    s = 0.0
    count = 0.0

    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i = i0 + di
                j = j0 + dj
                k = k0 + dk

                if (
                    i >= 1
                    and j >= 1
                    and k >= 1
                    and i < nx - 1
                    and j < ny - 1
                    and k < nz - 1
                ):
                    tile_i = i // kernel_config.TILE_SIZE
                    tile_j = j // kernel_config.TILE_SIZE
                    tile_k = k // kernel_config.TILE_SIZE
                    tile_index = tile_map[tile_i, tile_j, tile_k]

                    if tile_index == -1:
                        continue

                    local_i = i - tile_i * kernel_config.TILE_SIZE
                    local_j = j - tile_j * kernel_config.TILE_SIZE
                    local_k = k - tile_k * kernel_config.TILE_SIZE

                    p_center = p[tile_index, local_i, local_j, local_k]

                    lap = (
                        sparse_managment._sample_sparse_cell(p, tile_map, i + 1, j, k, 0.0)
                        + sparse_managment._sample_sparse_cell(p, tile_map, i - 1, j, k, 0.0)
                        + sparse_managment._sample_sparse_cell(p, tile_map, i, j + 1, k, 0.0)
                        + sparse_managment._sample_sparse_cell(p, tile_map, i, j - 1, k, 0.0)
                        + sparse_managment._sample_sparse_cell(p, tile_map, i, j, k + 1, 0.0)
                        + sparse_managment._sample_sparse_cell(p, tile_map, i, j, k - 1, 0.0)
                        - 6.0 * p_center
                    ) * inv_delta2

                    rhs = b[tile_index, local_i, local_j, local_k]

                    s += rhs - lap
                    count += 1.0

    coarse_b[I, J, K] = s / count if count > 0.0 else 0.0


@cuda.jit(cache=True)
def mg_prolongate_add_nearest(coarse_e, fine_p):
    I, J, K = cuda.grid(3)
    cnx, cny, cnz = coarse_e.shape
    fnx, fny, fnz = fine_p.shape

    if I >= cnx or J >= cny or K >= cnz:
        return

    e = 0.25 * coarse_e[I, J, K]

    i0 = 2 * I
    j0 = 2 * J
    k0 = 2 * K

    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i = i0 + di
                j = j0 + dj
                k = k0 + dk

                if i < fnx and j < fny and k < fnz:
                    fine_p[i, j, k] += e


@cuda.jit(cache=True)
def mg_rbgs_step(p, b, delta, parity):
    i, j, k = cuda.grid(3)

    nx, ny, nz = p.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    if ((i + j + k) & 1) != parity:
        return

    delta2 = delta * delta

    p[i, j, k] = (
        p[i + 1, j, k]
        + p[i - 1, j, k]
        + p[i, j + 1, k]
        + p[i, j - 1, k]
        + p[i, j, k + 1]
        + p[i, j, k - 1]
        - delta2 * b[i, j, k]
    ) / 6.0


@cuda.jit(cache=True)
def mg_rbgs_step_sparse_level0(p, b, delta, parity, tile_map, nx, ny, nz):
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
    ) = sparse_managment.tile_to_index()

    tile_index = tile_map[tile_i, tile_j, tile_k]

    if tile_index == -1:
        return

    if (
        i < 1
        or j < 1
        or k < 1
        or i >= nx - 1
        or j >= ny - 1
        or k >= nz - 1
        or ((i + j + k) & 1) != parity
    ):
        return

    delta2 = delta * delta

    center = (
        sparse_managment._sample_sparse_cell(p, tile_map, i + 1, j, k, 0.0)
        + sparse_managment._sample_sparse_cell(p, tile_map, i - 1, j, k, 0.0)
        + sparse_managment._sample_sparse_cell(p, tile_map, i, j + 1, k, 0.0)
        + sparse_managment._sample_sparse_cell(p, tile_map, i, j - 1, k, 0.0)
        + sparse_managment._sample_sparse_cell(p, tile_map, i, j, k + 1, 0.0)
        + sparse_managment._sample_sparse_cell(p, tile_map, i, j, k - 1, 0.0)
        - delta2 * b[tile_index, local_i, local_j, local_k]
    ) / 6.0

    p[tile_index, local_i, local_j, local_k] = center


def multigrid_smooth(
    p,
    b,
    delta,
    iterations,
    level=0,
    tile_map=None,
    nx=None,
    ny=None,
    nz=None,
):
    if level == 0:
        blocks = kernel_config.volume_blocks_per_grid(
            (nx, ny, nz),
            kernel_config.THREADS_PER_BLOCK_3D,
        )
    else:
        blocks = kernel_config.volume_blocks_per_grid(
            p.shape,
            kernel_config.THREADS_PER_BLOCK_3D,
        )

    for _ in range(iterations):
        if level == 0:
            mg_rbgs_step_sparse_level0[
                blocks, kernel_config.THREADS_PER_BLOCK_3D
            ](p, b, delta, 0, tile_map, nx, ny, nz)
            mg_rbgs_step_sparse_level0[
                blocks, kernel_config.THREADS_PER_BLOCK_3D
            ](p, b, delta, 1, tile_map, nx, ny, nz)
        else:
            mg_rbgs_step[blocks, kernel_config.THREADS_PER_BLOCK_3D](
                p, b, delta, 0
            )
            mg_rbgs_step[blocks, kernel_config.THREADS_PER_BLOCK_3D](
                p, b, delta, 1
            )

    if level == 0:
        BC.pressure_poisson_apply_neumann_bcs[
            blocks, kernel_config.THREADS_PER_BLOCK_3D
        ](p, tile_map, nx, ny, nz)
    else:
        BC.pressure_poisson_apply_neumann_bcs_dense[
            blocks, kernel_config.THREADS_PER_BLOCK_3D
        ](p)


def multigrid_vcycle(
    level,
    p_levels,
    b_levels,
    p_level0,
    b_level0,
    zero_levels,
    delta_levels,
    pre_smooth,
    post_smooth,
    coarse_smooth,
    nx,
    ny,
    nz,
    tile_map=None,
):
    p = p_level0 if level == 0 else p_levels[level]
    b = b_level0 if level == 0 else b_levels[level]
    delta = delta_levels[level]

    multigrid_smooth(
        p,
        b,
        delta,
        pre_smooth,
        level=level,
        tile_map=tile_map,
        nx=nx,
        ny=ny,
        nz=nz,
    )

    last_level = len(p_levels) - 1

    if level == last_level:
        multigrid_smooth(
            p,
            b,
            delta,
            coarse_smooth,
            level=level,
            tile_map=tile_map,
            nx=nx,
            ny=ny,
            nz=nz,
        )
        return

    coarse_p = p_levels[level + 1]
    coarse_b = b_levels[level + 1]

    coarse_blocks = kernel_config.volume_blocks_per_grid(
        coarse_p.shape,
        kernel_config.THREADS_PER_BLOCK_3D,
    )
    coarse_p.copy_to_device(zero_levels[level + 1])

    if level == 0 and tile_map is not None:
        mg_restrict_residual_8cell_sparse_level0[
            coarse_blocks,
            kernel_config.THREADS_PER_BLOCK_3D,
        ](
            p,
            b,
            coarse_b,
            delta,
            tile_map,
            nx,
            ny,
            nz,
        )
    else:
        mg_restrict_residual_8cell[
            coarse_blocks,
            kernel_config.THREADS_PER_BLOCK_3D,
        ](
            p,
            b,
            coarse_b,
            delta,
            nx,
            ny,
            nz,
        )

    multigrid_vcycle(
        level + 1,
        p_levels,
        b_levels,
        p_level0,
        b_level0,
        zero_levels,
        delta_levels,
        pre_smooth,
        post_smooth,
        coarse_smooth,
        nx,
        ny,
        nz,
        tile_map=None,
    )

    if level == 0 and tile_map is not None:
        mg_prolongate_add_nearest_sparse_level0[
            coarse_blocks,
            kernel_config.THREADS_PER_BLOCK_3D,
        ](coarse_p, p, tile_map, (nx, ny, nz))
    else:
        mg_prolongate_add_nearest[
            coarse_blocks,
            kernel_config.THREADS_PER_BLOCK_3D,
        ](
            coarse_p,
            p,
        )

    multigrid_smooth(
        p,
        b,
        delta,
        post_smooth,
        level=level,
        tile_map=tile_map,
        nx=nx,
        ny=ny,
        nz=nz,
    )
