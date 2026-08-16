from numba import cuda
import numpy as np
from typing import Any
import Solver.Kernel_GPU.kernel_config as kernel_config
import Solver.Kernel_GPU.sparse_managment as sparse_managment
import Solver.Kernel_GPU.Boundary_Conditions.domain_bc as BC

GPU_FIELD_DTYPE = kernel_config.GPU_FIELD_DTYPE

def create_multigrid_levels(shape, delta, min_size=8):
    p_levels = []
    b_levels = []
    delta_levels = []
    zero_levels = []

    nx = (shape[0] + 1) // 2
    ny = (shape[1] + 1) // 2
    nz = (shape[2] + 1) // 2
    level = 1

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


@cuda.jit(device=True, inline=True, cache=True)
def residual(p, b, delta, i, j, k):
    inv_delta2 = 1.0 / (delta * delta)

    laplace = (
        p[i + 1, j, k]
        + p[i - 1, j, k]
        + p[i, j + 1, k]
        + p[i, j - 1, k]
        + p[i, j, k + 1]
        + p[i, j, k - 1]
        - 6.0 * p[i, j, k]
    ) * inv_delta2

    return b[i, j, k] - laplace


@cuda.jit(device=True, inline=True, cache=True)
def residual_level_0(
    p,
    b,
    inv_delta2,
    tile_map,
    i,
    j,
    k,
):
    tile_i = i // kernel_config.TILE_SIZE
    tile_j = j // kernel_config.TILE_SIZE
    tile_k = k // kernel_config.TILE_SIZE

    tile_index = tile_map[tile_i, tile_j, tile_k]

    if tile_index == -1:
        return 0.0, False

    local_i = i - tile_i * kernel_config.TILE_SIZE
    local_j = j - tile_j * kernel_config.TILE_SIZE
    local_k = k - tile_k * kernel_config.TILE_SIZE

    laplace = (
        sparse_managment.get_pool_value(p, tile_map, i + 1, j, k, 0.0)
        + sparse_managment.get_pool_value(p, tile_map, i - 1, j, k, 0.0)
        + sparse_managment.get_pool_value(p, tile_map, i, j + 1, k, 0.0)
        + sparse_managment.get_pool_value(p, tile_map, i, j - 1, k, 0.0)
        + sparse_managment.get_pool_value(p, tile_map, i, j, k + 1, 0.0)
        + sparse_managment.get_pool_value(p, tile_map, i, j, k - 1, 0.0)
        - 6.0 * sparse_managment.get_pool_value(
            p, tile_map, i, j, k, 0.0
        )
    ) * inv_delta2

    rhs = b[tile_index, local_i, local_j, local_k]

    return rhs - laplace, True


@cuda.jit(cache=True)
def restrict_residual(p, b, coarse_b, delta, nx, ny, nz):
    I, J, K = cuda.grid(3)

    cnx, cny, cnz = coarse_b.shape

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

                if (
                    i >= 1
                    and j >= 1
                    and k >= 1
                    and i < nx - 1
                    and j < ny - 1
                    and k < nz - 1
                ):
                    r = residual(
                        p, b, delta, i, j, k
                    )

                    s += r
                    count += 1.0

    coarse_b[I, J, K] = s / count if count > 0.0 else 0.0


@cuda.jit(cache=True)
def restrict_residual_level_0(
    p,
    b,
    coarse_b,
    delta,
    tile_map,
    nx,
    ny,
    nz,
):
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
                    r, valid = residual_level_0(
                        p,
                        b,
                        inv_delta2,
                        tile_map,
                        i,
                        j,
                        k,
                    )

                    if not valid:
                        continue

                    s += r
                    count += 1.0

    coarse_b[I, J, K] = (
        s / count if count > 0.0 else 0.0
    )


@cuda.jit(cache=True)
def prolongate_add_nearest_level_0(coarse_e, fine_p, tile_map, field_shape):
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
def prolongate_add_nearest(coarse_e, fine_p):
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
def rbgs_step(p, b, delta, parity):
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
def rbgs_step_level_0(p, b, delta, parity, tile_map, nx, ny, nz):
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
        sparse_managment.get_pool_value(p, tile_map, i + 1, j, k, 0.0)
        + sparse_managment.get_pool_value(p, tile_map, i - 1, j, k, 0.0)
        + sparse_managment.get_pool_value(p, tile_map, i, j + 1, k, 0.0)
        + sparse_managment.get_pool_value(p, tile_map, i, j - 1, k, 0.0)
        + sparse_managment.get_pool_value(p, tile_map, i, j, k + 1, 0.0)
        + sparse_managment.get_pool_value(p, tile_map, i, j, k - 1, 0.0)
        - delta2 * sparse_managment.get_pool_value(b, tile_map, i, j, k, 0.0)
    ) / 6.0

    p[tile_index, local_i, local_j, local_k] = center


def smooth(
    p: Any,
    b: Any,
    delta: float,
    iterations: int,
    level: int = 0,
    tile_map: Any = None,
    nx: int | None = None,
    ny: int | None = None,
    nz: int | None = None,
) -> None:
    r"""
    Apply smoothing iterations to one multigrid pressure level.

    Level ``0`` is the finest pooled level and uses ``tile_map`` to access the
    sparse pressure storage. All higher levels are dense coarse grids. The
    smoother uses red-black Gauss-Seidel and reapplies Neumann boundary
    conditions after the requested number of iterations.

    Parameters
    ----------
    p
        Pressure buffer updated in-place for the current multigrid level.
    b
        Right-hand-side buffer for the current multigrid level.
    delta
        Cell size of the current level.
    iterations
        Number of smoothing iterations to perform.
    level
        Current multigrid level. ``0`` is the finest level.
    tile_map
        Sparse tile lookup map for level 0. Dense levels ignore it.
    nx, ny, nz
        Finest-level simulation resolution in cells. Required for level 0 and
        ignored for dense levels.

    Returns
    -------
    None
        The pressure buffer ``p`` is modified in-place.
    """
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
            rbgs_step_level_0[
                blocks, kernel_config.THREADS_PER_BLOCK_3D
            ](p, b, delta, 0, tile_map, nx, ny, nz)
            rbgs_step_level_0[
                blocks, kernel_config.THREADS_PER_BLOCK_3D
            ](p, b, delta, 1, tile_map, nx, ny, nz)
        else:
            rbgs_step[blocks, kernel_config.THREADS_PER_BLOCK_3D](
                p, b, delta, 0
            )
            rbgs_step[blocks, kernel_config.THREADS_PER_BLOCK_3D](
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


def v_cycle(
    level: int,
    p_levels: list[Any],
    b_levels: list[Any],
    p_level0: Any,
    b_level0: Any,
    zero_levels: list[Any],
    base_delta: float,
    delta_levels: list[float],
    pre_smooth: int,
    post_smooth: int,
    coarse_smooth: int,
    nx: int,
    ny: int,
    nz: int,
    tile_map: Any = None,
) -> None:
    r"""
    Run one multigrid V-cycle for the pressure solve.

    Level ``0`` is the finest level stored in ``p_level0`` and
    ``b_level0``. All entries in ``p_levels`` and ``b_levels`` represent
    coarser dense levels starting at half resolution. The cycle performs
    pre-smoothing, residual restriction to the next coarser level, recursive
    coarse-grid correction, prolongation back to the current level, and
    post-smoothing.

    Mathematically, the cycle operates on the linear system

    .. math::

        A p = b.

    First, smoothing reduces high-frequency error on the current level. Then
    the residual is computed and restricted to the next coarser grid:

    .. math::

        r = b - A p.

    On the coarse grid, the error equation is solved approximately:

    .. math::

        A e = r.

    The resulting coarse-grid error estimate is prolongated back to the finer
    level and added to the current pressure iterate:

    .. math::

        p \leftarrow p + e.

    A final post-smoothing step then damps the remaining high-frequency error.

    Level 0 uses ``tile_map`` to access the pooled tile storage,
    while all coarser levels operate on dense arrays.

    Parameters
    ----------
    level
        Current multigrid level. ``0`` is the finest level.
    p_levels
        Dense pressure buffers for all coarser multigrid levels above level 0.
    b_levels
        Dense right-hand-side buffers matching ``p_levels``.
    p_level0
        Sparse pressure buffer for the finest simulation level.
    b_level0
        Sparse pressure right-hand-side buffer for the finest simulation level.
    zero_levels
        Zero-filled dense arrays used to reset coarse pressure buffers before
        each coarse solve.
    base_delta
        Cell size of the finest simulation level.
    delta_levels
        Cell sizes for the dense coarse multigrid levels in ``p_levels``.
    pre_smooth
        Number of smoothing iterations before restriction.
    post_smooth
        Number of smoothing iterations after prolongation.
    coarse_smooth
        Number of smoothing iterations on the coarsest level.
    nx, ny, nz
        Finest-level simulation resolution in cells.
    tile_map
        Sparse tile lookup map for level 0. Coarser dense levels ignore it.

    Returns
    -------
    None
        The pressure buffers are updated in-place across the current V-cycle.
    """
    if level == 0:
        p = p_level0
        b = b_level0
        delta = base_delta
    else:
        dense_level = level - 1
        p = p_levels[dense_level]
        b = b_levels[dense_level]
        delta = delta_levels[dense_level]

    smooth(
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

    last_level = len(p_levels)

    if level == last_level:
        smooth(
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

    coarse_p = p_levels[level]
    coarse_b = b_levels[level]

    coarse_blocks = kernel_config.volume_blocks_per_grid(
        coarse_p.shape,
        kernel_config.THREADS_PER_BLOCK_3D,
    )
    coarse_p.copy_to_device(zero_levels[level])

    if level == 0 and tile_map is not None:
        restrict_residual_level_0[
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
        restrict_residual[
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

    v_cycle(
        level + 1,
        p_levels,
        b_levels,
        p_level0,
        b_level0,
        zero_levels,
        base_delta,
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
        prolongate_add_nearest_level_0[
            coarse_blocks,
            kernel_config.THREADS_PER_BLOCK_3D,
        ](coarse_p, p, tile_map, (nx, ny, nz))
    else:
        prolongate_add_nearest[
            coarse_blocks,
            kernel_config.THREADS_PER_BLOCK_3D,
        ](
            coarse_p,
            p,
        )

    smooth(
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
