from numba import cuda
import numpy as np
from typing import Any
import Solver.Kernel_GPU.kernel_config as kernel_config
import Solver.Kernel_GPU.sparse_managment as sparse_managment
import Solver.Kernel_GPU.Boundary_Conditions.domain_bc as BC

GPU_FIELD_DTYPE = kernel_config.GPU_FIELD_DTYPE


@cuda.jit(cache=True)
def build_coarse_tile_level(
    fine_tile_map: Any,
    coarse_tile_map: Any,
    coarse_active_tiles: Any,
    coarse_active_tile_count: Any,
) -> None:
    """
    Activate a coarse tile when at least one corresponding fine tile is active.

    One coarse tile covers a 2x2x2 group of fine tiles.
    """
    coarse_i, coarse_j, coarse_k = cuda.grid(3)

    if (
        coarse_i >= coarse_tile_map.shape[0]
        or coarse_j >= coarse_tile_map.shape[1]
        or coarse_k >= coarse_tile_map.shape[2]
    ):
        return

    fine_i_start = coarse_i * 2
    fine_j_start = coarse_j * 2
    fine_k_start = coarse_k * 2

    is_active = False

    for offset_i in range(2):
        fine_i = fine_i_start + offset_i

        if fine_i >= fine_tile_map.shape[0]:
            continue

        for offset_j in range(2):
            fine_j = fine_j_start + offset_j

            if fine_j >= fine_tile_map.shape[1]:
                continue

            for offset_k in range(2):
                fine_k = fine_k_start + offset_k

                if fine_k >= fine_tile_map.shape[2]:
                    continue

                if fine_tile_map[fine_i, fine_j, fine_k] != -1:
                    is_active = True
                    break

            if is_active:
                break

        if is_active:
            break

    if not is_active:
        return

    pool_index = cuda.atomic.add(
        coarse_active_tile_count,
        0,
        1,
    )

    coarse_tile_map[
        coarse_i,
        coarse_j,
        coarse_k,
    ] = pool_index

    coarse_active_tiles[pool_index, 0] = coarse_i
    coarse_active_tiles[pool_index, 1] = coarse_j
    coarse_active_tiles[pool_index, 2] = coarse_k


@cuda.jit(cache=True)
def clear_tile_map(tile_map: Any) -> None:
    """
    Reset every entry of a tile map to inactive.
    """
    tile_i, tile_j, tile_k = cuda.grid(3)

    if (
        tile_i >= tile_map.shape[0]
        or tile_j >= tile_map.shape[1]
        or tile_k >= tile_map.shape[2]
    ):
        return

    tile_map[tile_i, tile_j, tile_k] = -1


def build_coarse_tile_hierarchy(
    level_0_tile_map: Any,
    tile_maps: list[Any],
    active_tiles: list[Any],
    active_tile_counts: list[Any],
) -> list[int]:
    """
    Rebuild all coarse tile maps from the current level-0 tile map.

    Return the number of active tiles for every coarse level.
    """
    fine_tile_map = level_0_tile_map
    zero_counter = np.zeros(1, dtype=np.int32)

    for level in range(len(tile_maps)):
        coarse_tile_map = tile_maps[level]
        coarse_active_tiles = active_tiles[level]
        coarse_active_tile_count = active_tile_counts[level]

        coarse_active_tile_count.copy_to_device(zero_counter)

        blocks = kernel_config.volume_blocks_per_grid(
            coarse_tile_map.shape,
            kernel_config.THREADS_PER_BLOCK_3D,
        )

        clear_tile_map[
            blocks,
            kernel_config.THREADS_PER_BLOCK_3D,
        ](
            coarse_tile_map,
        )

        build_coarse_tile_level[
            blocks,
            kernel_config.THREADS_PER_BLOCK_3D,
        ](
            fine_tile_map,
            coarse_tile_map,
            coarse_active_tiles,
            coarse_active_tile_count,
        )

        fine_tile_map = coarse_tile_map

    active_tile_counts_host = []

    for active_tile_count in active_tile_counts:
        count = int(active_tile_count.copy_to_host()[0])
        active_tile_counts_host.append(count)

    return active_tile_counts_host


def create_multigrid_levels(
    shape: tuple[int, int, int],
    delta: float,
    level_0_pool_capacity: int,
    min_size: int = 8,
) -> Any:
    """
    Allocate the sparse coarse levels used below the sparse simulation grid.

    Every level halves each dimension with upward rounding and doubles the
    physical cell spacing. Pressure, right-hand-side, tile maps, active-tile
    lists, and reusable zero buffers are allocated until any dimension would
    fall below ``min_size``.
    """
    p_levels = []
    b_levels = []
    delta_levels = []
    zero_levels = []

    tile_maps = []
    active_tiles = []
    active_tile_counts = []
    level_shapes = []

    nx = (shape[0] + 1) // 2
    ny = (shape[1] + 1) // 2
    nz = (shape[2] + 1) // 2

    level = 1
    parent_pool_capacity = level_0_pool_capacity

    while nx >= min_size and ny >= min_size and nz >= min_size:
        level_shape = (nx, ny, nz)

        tile_shape = (
            (nx + kernel_config.TILE_SIZE - 1) // kernel_config.TILE_SIZE,
            (ny + kernel_config.TILE_SIZE - 1) // kernel_config.TILE_SIZE,
            (nz + kernel_config.TILE_SIZE - 1) // kernel_config.TILE_SIZE,
        )

        total_tile_count = tile_shape[0] * tile_shape[1] * tile_shape[2]

        pool_capacity = min(
            parent_pool_capacity,
            total_tile_count,
        )

        pool_shape = (
            pool_capacity,
            kernel_config.TILE_SIZE,
            kernel_config.TILE_SIZE,
            kernel_config.TILE_SIZE,
        )

        p_levels.append(
            cuda.device_array(
                pool_shape,
                dtype=GPU_FIELD_DTYPE,
            )
        )

        b_levels.append(
            cuda.device_array(
                pool_shape,
                dtype=GPU_FIELD_DTYPE,
            )
        )

        zero_levels.append(
            cuda.to_device(
                np.zeros(
                    pool_shape,
                    dtype=GPU_FIELD_DTYPE,
                )
            )
        )

        tile_maps.append(
            cuda.to_device(
                np.full(
                    tile_shape,
                    -1,
                    dtype=np.int32,
                )
            )
        )

        active_tiles.append(
            cuda.device_array(
                (pool_capacity, 3),
                dtype=np.int32,
            )
        )

        active_tile_counts.append(cuda.to_device(np.zeros(1, dtype=np.int32)))

        level_shapes.append(level_shape)
        delta_levels.append(delta * (2**level))

        parent_pool_capacity = pool_capacity

        nx = (nx + 1) // 2
        ny = (ny + 1) // 2
        nz = (nz + 1) // 2
        level += 1

    return (
        p_levels,
        b_levels,
        delta_levels,
        zero_levels,
        tile_maps,
        active_tiles,
        active_tile_counts,
        level_shapes,
    )


@cuda.jit(cache=True)
def rbgs_step_sparse(
    p: Any,
    b: Any,
    delta: float,
    parity: int,
    tile_map: Any,
    active_tiles: Any,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """
    Perform one red or black Gauss-Seidel sweep over active sparse tiles.

    One CUDA block processes one active tile.
    """
    active_index = cuda.blockIdx.x

    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y
    local_k = cuda.threadIdx.z

    tile_i = active_tiles[active_index, 0]
    tile_j = active_tiles[active_index, 1]
    tile_k = active_tiles[active_index, 2]

    tile_index = tile_map[
        tile_i,
        tile_j,
        tile_k,
    ]

    if tile_index == -1:
        return

    i = tile_i * kernel_config.TILE_SIZE + local_i
    j = tile_j * kernel_config.TILE_SIZE + local_j
    k = tile_k * kernel_config.TILE_SIZE + local_k

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    if ((i + j + k) & 1) != parity:
        return

    delta2 = delta * delta

    p[
        tile_index,
        local_i,
        local_j,
        local_k,
    ] = (
        sparse_managment.get_pool_value(p, tile_map, i + 1, j, k, 0.0)
        + sparse_managment.get_pool_value(p, tile_map, i - 1, j, k, 0.0)
        + sparse_managment.get_pool_value(p, tile_map, i, j + 1, k, 0.0)
        + sparse_managment.get_pool_value(p, tile_map, i, j - 1, k, 0.0)
        + sparse_managment.get_pool_value(p, tile_map, i, j, k + 1, 0.0)
        + sparse_managment.get_pool_value(p, tile_map, i, j, k - 1, 0.0)
        - delta2
        * b[
            tile_index,
            local_i,
            local_j,
            local_k,
        ]
    ) / 6.0


@cuda.jit(cache=True)
def rbgs_step_level_0(
    p: Any, b: Any, delta: float, parity: int, tile_map: Any, nx: int, ny: int, nz: int
) -> None:
    """
    Perform one red or black Gauss-Seidel sweep on the sparse finest grid.
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


@cuda.jit(device=True, inline=True, cache=True)
def residual_sparse(
    p: Any,
    b: Any,
    inv_delta2: Any,
    tile_map: Any,
    i: int,
    j: int,
    k: int,
) -> Any:
    """
    Evaluate r = b - laplace(p) at one sparse-grid cell.
    """
    tile_i = i // kernel_config.TILE_SIZE
    tile_j = j // kernel_config.TILE_SIZE
    tile_k = k // kernel_config.TILE_SIZE

    tile_index = tile_map[
        tile_i,
        tile_j,
        tile_k,
    ]

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
        - 6.0
        * p[
            tile_index,
            local_i,
            local_j,
            local_k,
        ]
    ) * inv_delta2

    rhs = b[
        tile_index,
        local_i,
        local_j,
        local_k,
    ]

    return rhs - laplace, True


@cuda.jit(cache=True)
def restrict_residual_sparse(
    fine_p: Any,
    fine_b: Any,
    coarse_b: Any,
    fine_delta: float,
    fine_tile_map: Any,
    coarse_tile_map: Any,
    coarse_active_tiles: Any,
    fine_nx: int,
    fine_ny: int,
    fine_nz: int,
    coarse_nx: int,
    coarse_ny: int,
    coarse_nz: int,
) -> None:
    """
    Restrict the residual of one sparse level to the next sparse level.

    One CUDA block processes one active coarse tile. Threads correspond to
    cells inside that coarse tile.
    """
    coarse_pool_index = cuda.blockIdx.x

    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y
    local_k = cuda.threadIdx.z

    coarse_tile_i = coarse_active_tiles[coarse_pool_index, 0]
    coarse_tile_j = coarse_active_tiles[coarse_pool_index, 1]
    coarse_tile_k = coarse_active_tiles[coarse_pool_index, 2]

    I = coarse_tile_i * kernel_config.TILE_SIZE + local_i
    J = coarse_tile_j * kernel_config.TILE_SIZE + local_j
    K = coarse_tile_k * kernel_config.TILE_SIZE + local_k

    if I >= coarse_nx or J >= coarse_ny or K >= coarse_nz:
        return

    # Der Index sollte mit blockIdx.x übereinstimmen. Der Zugriff über
    # die Map macht die Beziehung aber explizit.
    coarse_tile_index = coarse_tile_map[
        coarse_tile_i,
        coarse_tile_j,
        coarse_tile_k,
    ]

    if coarse_tile_index == -1:
        return

    inv_delta2 = 1.0 / (fine_delta * fine_delta)

    fine_i_start = 2 * I
    fine_j_start = 2 * J
    fine_k_start = 2 * K

    residual_sum = 0.0
    residual_count = 0.0

    for offset_i in range(2):
        for offset_j in range(2):
            for offset_k in range(2):
                i = fine_i_start + offset_i
                j = fine_j_start + offset_j
                k = fine_k_start + offset_k

                if (
                    i < 1
                    or j < 1
                    or k < 1
                    or i >= fine_nx - 1
                    or j >= fine_ny - 1
                    or k >= fine_nz - 1
                ):
                    continue

                residual_value, valid = residual_sparse(
                    fine_p,
                    fine_b,
                    inv_delta2,
                    fine_tile_map,
                    i,
                    j,
                    k,
                )

                if not valid:
                    continue

                residual_sum += residual_value
                residual_count += 1.0

    if residual_count > 0.0:
        coarse_b[
            coarse_tile_index,
            local_i,
            local_j,
            local_k,
        ] = (
            residual_sum / residual_count
        )
    else:
        coarse_b[
            coarse_tile_index,
            local_i,
            local_j,
            local_k,
        ] = 0.0


@cuda.jit(cache=True)
def prolongate_add_nearest_sparse(
    coarse_e: Any,
    fine_p: Any,
    coarse_tile_map: Any,
    fine_tile_map: Any,
    coarse_active_tiles: Any,
    coarse_nx: int,
    coarse_ny: int,
    coarse_nz: int,
    fine_nx: int,
    fine_ny: int,
    fine_nz: int,
) -> None:
    """
    Prolongate sparse coarse-grid error by nearest-neighbor injection and add
    it to the sparse fine-grid pressure.

    One CUDA block processes one active coarse tile.
    """
    coarse_pool_index = cuda.blockIdx.x

    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y
    local_k = cuda.threadIdx.z

    coarse_tile_i = coarse_active_tiles[coarse_pool_index, 0]
    coarse_tile_j = coarse_active_tiles[coarse_pool_index, 1]
    coarse_tile_k = coarse_active_tiles[coarse_pool_index, 2]

    I = coarse_tile_i * kernel_config.TILE_SIZE + local_i
    J = coarse_tile_j * kernel_config.TILE_SIZE + local_j
    K = coarse_tile_k * kernel_config.TILE_SIZE + local_k

    if I >= coarse_nx or J >= coarse_ny or K >= coarse_nz:
        return

    coarse_tile_index = coarse_tile_map[
        coarse_tile_i,
        coarse_tile_j,
        coarse_tile_k,
    ]

    if coarse_tile_index == -1:
        return

    error = (
        0.25
        * coarse_e[
            coarse_tile_index,
            local_i,
            local_j,
            local_k,
        ]
    )

    fine_i_start = 2 * I
    fine_j_start = 2 * J
    fine_k_start = 2 * K

    for offset_i in range(2):
        for offset_j in range(2):
            for offset_k in range(2):
                i = fine_i_start + offset_i
                j = fine_j_start + offset_j
                k = fine_k_start + offset_k

                if i >= fine_nx or j >= fine_ny or k >= fine_nz:
                    continue

                fine_tile_i = i // kernel_config.TILE_SIZE
                fine_tile_j = j // kernel_config.TILE_SIZE
                fine_tile_k = k // kernel_config.TILE_SIZE

                fine_tile_index = fine_tile_map[
                    fine_tile_i,
                    fine_tile_j,
                    fine_tile_k,
                ]

                if fine_tile_index == -1:
                    continue

                fine_local_i = i - fine_tile_i * kernel_config.TILE_SIZE
                fine_local_j = j - fine_tile_j * kernel_config.TILE_SIZE
                fine_local_k = k - fine_tile_k * kernel_config.TILE_SIZE

                fine_p[
                    fine_tile_index,
                    fine_local_i,
                    fine_local_j,
                    fine_local_k,
                ] += error


def smooth(
    p: Any,
    b: Any,
    delta: float,
    iterations: int,
    tile_map: Any,
    active_tiles: Any,
    active_tile_count: int | None,
    field_shape: tuple[int, int, int],
) -> None:
    """
    Apply red-black Gauss-Seidel smoothing to one sparse multigrid level.

    Level 0 currently has no compact active-tile list and therefore uses the
    existing level-0 kernel. Coarse levels are dispatched only over their
    compact active-tile lists.
    """
    nx, ny, nz = field_shape

    is_level_0 = active_tiles is None

    if is_level_0:
        blocks = kernel_config.volume_blocks_per_grid(
            field_shape,
            kernel_config.THREADS_PER_BLOCK_3D,
        )

        for _ in range(iterations):
            rbgs_step_level_0[
                blocks,
                kernel_config.THREADS_PER_BLOCK_3D,
            ](
                p,
                b,
                delta,
                0,
                tile_map,
                nx,
                ny,
                nz,
            )

            rbgs_step_level_0[
                blocks,
                kernel_config.THREADS_PER_BLOCK_3D,
            ](
                p,
                b,
                delta,
                1,
                tile_map,
                nx,
                ny,
                nz,
            )

    else:
        if active_tile_count is None:
            raise ValueError("active_tile_count is required for sparse coarse levels")

        if active_tile_count == 0:
            return

        for _ in range(iterations):
            rbgs_step_sparse[
                active_tile_count,
                kernel_config.THREADS_PER_BLOCK_3D,
            ](
                p,
                b,
                delta,
                0,
                tile_map,
                active_tiles,
                nx,
                ny,
                nz,
            )

            rbgs_step_sparse[
                active_tile_count,
                kernel_config.THREADS_PER_BLOCK_3D,
            ](
                p,
                b,
                delta,
                1,
                tile_map,
                active_tiles,
                nx,
                ny,
                nz,
            )

    boundary_blocks = kernel_config.volume_blocks_per_grid(
        field_shape,
        kernel_config.THREADS_PER_BLOCK_3D,
    )

    BC.pressure_poisson_apply_neumann_bcs[
        boundary_blocks,
        kernel_config.THREADS_PER_BLOCK_3D,
    ](
        p,
        tile_map,
        nx,
        ny,
        nz,
    )


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
    tile_map: Any,
    multigrid_tile_maps: list[Any],
    multigrid_active_tiles: list[Any],
    multigrid_active_tile_counts: list[int],
    multigrid_level_shapes: list[tuple[int, int, int]],
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

    """
    if level == 0:
        p = p_level0
        b = b_level0
        delta = base_delta

        current_tile_map = tile_map
        current_active_tiles = None
        current_active_tile_count = None
        current_shape = (nx, ny, nz)

    else:
        current_level = level - 1

        p = p_levels[current_level]
        b = b_levels[current_level]
        delta = delta_levels[current_level]

        current_tile_map = multigrid_tile_maps[current_level]
        current_active_tiles = multigrid_active_tiles[current_level]
        current_active_tile_count = multigrid_active_tile_counts[current_level]
        current_shape = multigrid_level_shapes[current_level]

    smooth(
        p,
        b,
        delta,
        pre_smooth,
        tile_map=current_tile_map,
        active_tiles=current_active_tiles,
        active_tile_count=current_active_tile_count,
        field_shape=current_shape,
    )

    last_level = len(p_levels)

    if level == last_level:
        smooth(
            p,
            b,
            delta,
            coarse_smooth,
            tile_map=current_tile_map,
            active_tiles=current_active_tiles,
            active_tile_count=current_active_tile_count,
            field_shape=current_shape,
        )
        return

    coarse_level = level

    coarse_p = p_levels[coarse_level]
    coarse_b = b_levels[coarse_level]

    coarse_tile_map = multigrid_tile_maps[coarse_level]
    coarse_active_tiles = multigrid_active_tiles[coarse_level]
    coarse_active_tile_count = multigrid_active_tile_counts[coarse_level]
    coarse_shape = multigrid_level_shapes[coarse_level]

    if coarse_active_tile_count == 0:
        return

    coarse_p[:coarse_active_tile_count].copy_to_device(
        zero_levels[coarse_level][:coarse_active_tile_count]
    )

    coarse_b[:coarse_active_tile_count].copy_to_device(
        zero_levels[coarse_level][:coarse_active_tile_count]
    )

    restrict_residual_sparse[
        coarse_active_tile_count,
        kernel_config.THREADS_PER_BLOCK_3D,
    ](
        p,
        b,
        coarse_b,
        delta,
        current_tile_map,
        coarse_tile_map,
        coarse_active_tiles,
        *current_shape,
        *coarse_shape,
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
        tile_map,
        multigrid_tile_maps,
        multigrid_active_tiles,
        multigrid_active_tile_counts,
        multigrid_level_shapes,
    )

    prolongate_add_nearest_sparse[
        coarse_active_tile_count,
        kernel_config.THREADS_PER_BLOCK_3D,
    ](
        coarse_p,
        p,
        coarse_tile_map,
        current_tile_map,
        coarse_active_tiles,
        *coarse_shape,
        *current_shape,
    )

    smooth(
        p,
        b,
        delta,
        post_smooth,
        tile_map=current_tile_map,
        active_tiles=current_active_tiles,
        active_tile_count=current_active_tile_count,
        field_shape=current_shape,
    )
