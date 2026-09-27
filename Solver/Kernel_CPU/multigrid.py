from typing import Any

import numpy as np
from numba import njit, prange

import Solver.Kernel_CPU.kernel_config as kernel_config
import Solver.Kernel_CPU.sparse_managment as sparse_managment
import Solver.Kernel_CPU.Boundary_Conditions.domain_bc as BC

CPU_FIELD_DTYPE = kernel_config.CPU_FIELD_DTYPE


def build_coarse_tile_level(
    fine_tile_map: Any,
    coarse_tile_map: Any,
    coarse_active_tiles: Any,
    coarse_active_tile_count: Any,
) -> None:
    """Activate coarse tiles covering at least one active fine tile."""
    coarse_tile_map.fill(-1)
    fine_coords = np.argwhere(fine_tile_map != -1)
    coarse_active_tile_count[0] = 0
    if fine_coords.size == 0:
        return
    coarse_coords = np.unique(fine_coords // 2, axis=0)
    count = len(coarse_coords)
    if count > len(coarse_active_tiles):
        raise RuntimeError("CPU multigrid sparse tile capacity exceeded")
    coarse_active_tiles[:count] = coarse_coords
    for pool_index, (tile_i, tile_j, tile_k) in enumerate(coarse_coords):
        coarse_tile_map[tile_i, tile_j, tile_k] = pool_index
    coarse_active_tile_count[0] = count


def clear_tile_map(tile_map: Any) -> None:
    """Reset every entry of a tile map to inactive."""
    tile_map.fill(-1)


def build_coarse_tile_hierarchy(
    level_0_tile_map: Any,
    tile_maps: list[Any],
    active_tiles: list[Any],
    active_tile_counts: list[Any],
) -> list[int]:
    """Rebuild all coarse tile maps from the current level-0 tile map."""
    fine_tile_map = level_0_tile_map
    counts = []
    for coarse_tile_map, coarse_active_tiles, count_buffer in zip(
        tile_maps, active_tiles, active_tile_counts
    ):
        clear_tile_map(coarse_tile_map)
        build_coarse_tile_level(
            fine_tile_map, coarse_tile_map, coarse_active_tiles, count_buffer
        )
        counts.append(int(count_buffer[0]))
        fine_tile_map = coarse_tile_map
    return counts


def create_multigrid_levels(
    shape: tuple[int, int, int],
    delta: float,
    level_0_pool_capacity: int,
    min_size: int = 8,
) -> Any:
    p_levels = []
    b_levels = []
    delta_levels = []
    tile_maps = []
    active_tiles = []
    active_tile_counts = []
    level_shapes = []

    nx, ny, nz = ((value + 1) // 2 for value in shape)
    level = 1
    parent_capacity = int(level_0_pool_capacity)

    while nx >= min_size and ny >= min_size and nz >= min_size:
        level_shape = (nx, ny, nz)
        tile_shape = tuple(
            (value + kernel_config.TILE_SIZE - 1) // kernel_config.TILE_SIZE
            for value in level_shape
        )
        capacity = min(parent_capacity, int(np.prod(tile_shape)))
        pool_shape = (
            capacity,
            kernel_config.TILE_SIZE,
            kernel_config.TILE_SIZE,
            kernel_config.TILE_SIZE,
        )

        p_levels.append(np.empty(pool_shape, dtype=CPU_FIELD_DTYPE))
        b_levels.append(np.empty(pool_shape, dtype=CPU_FIELD_DTYPE))
        tile_maps.append(np.full(tile_shape, -1, dtype=np.int32))
        active_tiles.append(np.empty((capacity, 3), dtype=np.int32))
        active_tile_counts.append(np.zeros(1, dtype=np.int32))
        level_shapes.append(level_shape)
        delta_levels.append(delta * (2**level))

        parent_capacity = capacity
        nx, ny, nz = ((value + 1) // 2 for value in level_shape)
        level += 1

    return (
        p_levels,
        b_levels,
        delta_levels,
        tile_maps,
        active_tiles,
        active_tile_counts,
        level_shapes,
    )


@njit(cache=True, parallel=True)
def rbgs_step_sparse(
    p: Any,
    b: Any,
    delta: float,
    parity: int,
    tile_map: Any,
    active_tiles: Any,
    active_tile_count: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    tile_size = kernel_config.TILE_SIZE
    delta2 = delta * delta

    for active_index in prange(active_tile_count):
        tile_i, tile_j, tile_k = active_tiles[active_index]
        pool_index = tile_map[tile_i, tile_j, tile_k]
        if pool_index == -1:
            continue

        for local_i in range(tile_size):
            i = tile_i * tile_size + local_i
            for local_j in range(tile_size):
                j = tile_j * tile_size + local_j
                for local_k in range(tile_size):
                    k = tile_k * tile_size + local_k
                    if (
                        i < 1
                        or j < 1
                        or k < 1
                        or i >= nx - 1
                        or j >= ny - 1
                        or k >= nz - 1
                        or ((i + j + k) & 1) != parity
                    ):
                        continue
                    p[pool_index, local_i, local_j, local_k] = (
                        sparse_managment.get_pool_value(p, tile_map, i + 1, j, k, 0.0)
                        + sparse_managment.get_pool_value(p, tile_map, i - 1, j, k, 0.0)
                        + sparse_managment.get_pool_value(p, tile_map, i, j + 1, k, 0.0)
                        + sparse_managment.get_pool_value(p, tile_map, i, j - 1, k, 0.0)
                        + sparse_managment.get_pool_value(p, tile_map, i, j, k + 1, 0.0)
                        + sparse_managment.get_pool_value(p, tile_map, i, j, k - 1, 0.0)
                        - delta2 * b[pool_index, local_i, local_j, local_k]
                    ) / 6.0


@njit(cache=True, parallel=True)
def rbgs_step_level_0(
    p: Any,
    b: Any,
    delta: float,
    parity: int,
    tile_map: Any,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Perform one red or black sweep on the sparse finest grid."""
    tile_size = kernel_config.TILE_SIZE
    delta2 = delta * delta
    for flat in prange(nx * ny * nz):
        i = flat // (ny * nz)
        remainder = flat % (ny * nz)
        j = remainder // nz
        k = remainder % nz
        if (
            i < 1
            or j < 1
            or k < 1
            or i >= nx - 1
            or j >= ny - 1
            or k >= nz - 1
            or ((i + j + k) & 1) != parity
        ):
            continue
        tile_i, tile_j, tile_k = i // tile_size, j // tile_size, k // tile_size
        pool_index = tile_map[tile_i, tile_j, tile_k]
        if pool_index == -1:
            continue
        local_i, local_j, local_k = i % tile_size, j % tile_size, k % tile_size
        p[pool_index, local_i, local_j, local_k] = (
            sparse_managment.get_pool_value(p, tile_map, i + 1, j, k, 0.0)
            + sparse_managment.get_pool_value(p, tile_map, i - 1, j, k, 0.0)
            + sparse_managment.get_pool_value(p, tile_map, i, j + 1, k, 0.0)
            + sparse_managment.get_pool_value(p, tile_map, i, j - 1, k, 0.0)
            + sparse_managment.get_pool_value(p, tile_map, i, j, k + 1, 0.0)
            + sparse_managment.get_pool_value(p, tile_map, i, j, k - 1, 0.0)
            - delta2 * b[pool_index, local_i, local_j, local_k]
        ) / 6.0


@njit(inline="always", cache=True)
def residual_sparse(
    p: Any,
    b: Any,
    inv_delta2: float,
    tile_map: Any,
    i: int,
    j: int,
    k: int,
) -> Any:
    """Evaluate the residual at one cell of a sparse level."""
    tile_size = kernel_config.TILE_SIZE
    pool_index = tile_map[i // tile_size, j // tile_size, k // tile_size]
    if pool_index == -1:
        return 0.0, False
    laplace = (
        sparse_managment.get_pool_value(p, tile_map, i + 1, j, k, 0.0)
        + sparse_managment.get_pool_value(p, tile_map, i - 1, j, k, 0.0)
        + sparse_managment.get_pool_value(p, tile_map, i, j + 1, k, 0.0)
        + sparse_managment.get_pool_value(p, tile_map, i, j - 1, k, 0.0)
        + sparse_managment.get_pool_value(p, tile_map, i, j, k + 1, 0.0)
        + sparse_managment.get_pool_value(p, tile_map, i, j, k - 1, 0.0)
        - 6.0 * sparse_managment.get_pool_value(p, tile_map, i, j, k, 0.0)
    ) * inv_delta2
    rhs = sparse_managment.get_pool_value(b, tile_map, i, j, k, 0.0)
    return rhs - laplace, True


@njit(cache=True, parallel=True)
def restrict_residual_sparse(
    fine_p: Any,
    fine_b: Any,
    coarse_b: Any,
    fine_delta: float,
    fine_tile_map: Any,
    coarse_tile_map: Any,
    coarse_active_tiles: Any,
    coarse_active_tile_count: int,
    fine_nx: int,
    fine_ny: int,
    fine_nz: int,
    coarse_nx: int,
    coarse_ny: int,
    coarse_nz: int,
) -> None:
    tile_size = kernel_config.TILE_SIZE
    inv_delta2 = 1.0 / (fine_delta * fine_delta)

    for coarse_pool_index in prange(coarse_active_tile_count):
        tile_i, tile_j, tile_k = coarse_active_tiles[coarse_pool_index]
        for local_i in range(tile_size):
            I = tile_i * tile_size + local_i
            for local_j in range(tile_size):
                J = tile_j * tile_size + local_j
                for local_k in range(tile_size):
                    K = tile_k * tile_size + local_k
                    if I >= coarse_nx or J >= coarse_ny or K >= coarse_nz:
                        continue

                    residual_sum = 0.0
                    residual_count = 0
                    for di in range(2):
                        i = 2 * I + di
                        for dj in range(2):
                            j = 2 * J + dj
                            for dk in range(2):
                                k = 2 * K + dk
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
                                residual_count += 1
                    coarse_b[coarse_pool_index, local_i, local_j, local_k] = (
                        residual_sum / residual_count if residual_count else 0.0
                    )


@njit(cache=True, parallel=True)
def prolongate_add_nearest_sparse(
    coarse_e: Any,
    fine_p: Any,
    coarse_active_tiles: Any,
    coarse_active_tile_count: int,
    fine_tile_map: Any,
    coarse_nx: int,
    coarse_ny: int,
    coarse_nz: int,
    fine_nx: int,
    fine_ny: int,
    fine_nz: int,
) -> None:
    tile_size = kernel_config.TILE_SIZE
    for coarse_pool_index in prange(coarse_active_tile_count):
        tile_i, tile_j, tile_k = coarse_active_tiles[coarse_pool_index]
        for local_i in range(tile_size):
            I = tile_i * tile_size + local_i
            for local_j in range(tile_size):
                J = tile_j * tile_size + local_j
                for local_k in range(tile_size):
                    K = tile_k * tile_size + local_k
                    if I >= coarse_nx or J >= coarse_ny or K >= coarse_nz:
                        continue
                    error = (
                        0.25 * coarse_e[coarse_pool_index, local_i, local_j, local_k]
                    )
                    for di in range(2):
                        i = 2 * I + di
                        for dj in range(2):
                            j = 2 * J + dj
                            for dk in range(2):
                                k = 2 * K + dk
                                if i >= fine_nx or j >= fine_ny or k >= fine_nz:
                                    continue
                                fine_pool = fine_tile_map[
                                    i // tile_size, j // tile_size, k // tile_size
                                ]
                                if fine_pool != -1:
                                    fine_p[
                                        fine_pool,
                                        i % tile_size,
                                        j % tile_size,
                                        k % tile_size,
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
    nx, ny, nz = field_shape
    for _ in range(iterations):
        if active_tiles is None:
            rbgs_step_level_0(p, b, delta, 0, tile_map, nx, ny, nz)
            rbgs_step_level_0(p, b, delta, 1, tile_map, nx, ny, nz)
        else:
            rbgs_step_sparse(
                p,
                b,
                delta,
                0,
                tile_map,
                active_tiles,
                active_tile_count,
                nx,
                ny,
                nz,
            )
            rbgs_step_sparse(
                p,
                b,
                delta,
                1,
                tile_map,
                active_tiles,
                active_tile_count,
                nx,
                ny,
                nz,
            )
    BC.pressure_poisson_apply_neumann_bcs(p, tile_map, nx, ny, nz)


@njit(cache=True, parallel=True)
def clear_sparse_fields(p: Any, b: Any, active_tile_count: int) -> None:
    for pool_index in prange(active_tile_count):
        p[pool_index].fill(0.0)
        b[pool_index].fill(0.0)


def v_cycle(
    level: int,
    p_levels: list[Any],
    b_levels: list[Any],
    p_level0: Any,
    b_level0: Any,
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
    if level == 0:
        p, b, delta = p_level0, b_level0, base_delta
        current_map = tile_map
        current_tiles = None
        current_count = None
        current_shape = (nx, ny, nz)
    else:
        current = level - 1
        p, b, delta = p_levels[current], b_levels[current], delta_levels[current]
        current_map = multigrid_tile_maps[current]
        current_tiles = multigrid_active_tiles[current]
        current_count = multigrid_active_tile_counts[current]
        current_shape = multigrid_level_shapes[current]

    smooth(
        p,
        b,
        delta,
        pre_smooth,
        current_map,
        current_tiles,
        current_count,
        current_shape,
    )
    if level == len(p_levels):
        smooth(
            p,
            b,
            delta,
            coarse_smooth,
            current_map,
            current_tiles,
            current_count,
            current_shape,
        )
        return

    coarse = level
    coarse_count = multigrid_active_tile_counts[coarse]
    if coarse_count == 0:
        return
    coarse_p, coarse_b = p_levels[coarse], b_levels[coarse]
    coarse_map = multigrid_tile_maps[coarse]
    coarse_tiles = multigrid_active_tiles[coarse]
    coarse_shape = multigrid_level_shapes[coarse]
    clear_sparse_fields(coarse_p, coarse_b, coarse_count)
    restrict_residual_sparse(
        p,
        b,
        coarse_b,
        delta,
        current_map,
        coarse_map,
        coarse_tiles,
        coarse_count,
        *current_shape,
        *coarse_shape,
    )
    v_cycle(
        level + 1,
        p_levels,
        b_levels,
        p_level0,
        b_level0,
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
    prolongate_add_nearest_sparse(
        coarse_p,
        p,
        coarse_tiles,
        coarse_count,
        current_map,
        *coarse_shape,
        *current_shape,
    )
    smooth(
        p,
        b,
        delta,
        post_smooth,
        current_map,
        current_tiles,
        current_count,
        current_shape,
    )
