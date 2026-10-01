from typing import Any

import numpy as np
import pyopencl as cl

import Solver.Kernel.kernel_config as kernel_config
import Solver.Kernel.helper as helper
from Solver.Kernel.timing import record_kernel_event

FIELD_DTYPE = kernel_config.FIELD_DTYPE


def create_multigrid_levels(
    context: cl.Context,
    shape: tuple[int, int, int],
    delta: float,
    level_0_pool_capacity: int,
    min_size: int = 8,
) -> Any:
    """
    Allocate the sparse coarse levels used below the sparse simulation grid.

    Every level halves each dimension with upward rounding and doubles the
    physical cell spacing. Pressure, right-hand-side, tile maps, active-tile
    lists are allocated until any dimension would fall below ``min_size``.
    """
    p_levels = []
    b_levels = []
    delta_levels = []
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
            helper.device_array(
                context,
                pool_shape,
                dtype=FIELD_DTYPE,
            )
        )

        b_levels.append(
            helper.device_array(
                context,
                pool_shape,
                dtype=FIELD_DTYPE,
            )
        )

        tile_maps.append(
            helper.to_device(
                context,
                np.full(
                    tile_shape,
                    -1,
                    dtype=np.int32,
                ),
            )
        )

        active_tiles.append(
            helper.device_array(
                context,
                (pool_capacity, 3),
                dtype=np.int32,
            )
        )

        active_tile_counts.append(
            helper.to_device(context, np.zeros(1, dtype=np.int32))
        )

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
        tile_maps,
        active_tiles,
        active_tile_counts,
        level_shapes,
    )


def build_coarse_tile_hierarchy(
    queue: cl.CommandQueue,
    multigrid_kernels: dict[str, cl.Kernel],
    level_0_tile_map: Any,
    level_0_tile_shape: tuple[int, int, int],
    tile_maps: list[Any],
    level_shapes: list[tuple[int, int, int]],
    active_tiles: list[Any],
    active_tile_counts: list[Any],
) -> list[Any]:
    """
    Rebuild all coarse tile maps from the current level-0 tile map.

    Return the device-side active-tile count buffer for every coarse level.

    Counts deliberately remain on the device. Later kernels dispatch over the
    allocated tile capacity and reject workgroups beyond the device count,
    avoiding a blocking device-to-host readback for every multigrid level.
    """
    fine_tile_map = level_0_tile_map
    fine_tile_shape = level_0_tile_shape

    for level in range(len(tile_maps)):
        coarse_tile_map = tile_maps[level]
        coarse_active_tiles = active_tiles[level]
        coarse_active_tile_count = active_tile_counts[level]

        coarse_tile_shape = tuple(
            (size + kernel_config.TILE_SIZE - 1) // kernel_config.TILE_SIZE
            for size in level_shapes[level]
        )

        helper.fill_device(
            queue,
            coarse_active_tile_count,
            0,
            dtype=np.int32,
        )

        helper.fill_device(
            queue,
            coarse_tile_map,
            -1,
            dtype=np.int32,
        )

        coarse_active_tile_capacity = coarse_active_tiles.size // (
            3 * np.dtype(np.int32).itemsize
        )

        multigrid_kernels["build_coarse_tile_level"](
            queue,
            coarse_tile_shape,
            None,
            fine_tile_map,
            coarse_tile_map,
            coarse_active_tiles,
            coarse_active_tile_count,
            np.int32(coarse_active_tile_capacity),
            np.int32(fine_tile_shape[0]),
            np.int32(fine_tile_shape[1]),
            np.int32(fine_tile_shape[2]),
            np.int32(coarse_tile_shape[0]),
            np.int32(coarse_tile_shape[1]),
            np.int32(coarse_tile_shape[2]),
        )

        fine_tile_map = coarse_tile_map
        fine_tile_shape = coarse_tile_shape

    return active_tile_counts


def v_cycle(
    multigrid_kernels: dict[str, cl.Kernel],
    queue: cl.CommandQueue,
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
    multigrid_active_tile_counts: list[Any],
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
        multigrid_kernels,
        queue,
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
            multigrid_kernels,
            queue,
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

    current_tile_shape = tuple(
        (size + kernel_config.TILE_SIZE - 1) // kernel_config.TILE_SIZE
        for size in current_shape
    )

    coarse_tile_shape = tuple(
        (size + kernel_config.TILE_SIZE - 1) // kernel_config.TILE_SIZE
        for size in coarse_shape
    )

    local_work_size = kernel_config.THREADS_PER_BLOCK_3D

    coarse_active_tile_capacity = coarse_active_tiles.size // (
        3 * np.dtype(np.int32).itemsize
    )

    global_work_size = (
        coarse_active_tile_capacity * local_work_size[0],
        local_work_size[1],
        local_work_size[2],
    )

    multigrid_kernels["restrict_residual_sparse"](
        queue,
        global_work_size,
        local_work_size,
        p,
        b,
        coarse_p,
        coarse_b,
        np.float32(delta),
        current_tile_map,
        coarse_tile_map,
        coarse_active_tiles,
        coarse_active_tile_count,
        np.int32(current_shape[0]),
        np.int32(current_shape[1]),
        np.int32(current_shape[2]),
        np.int32(coarse_shape[0]),
        np.int32(coarse_shape[1]),
        np.int32(coarse_shape[2]),
        np.int32(current_tile_shape[1]),
        np.int32(current_tile_shape[2]),
        np.int32(coarse_tile_shape[1]),
        np.int32(coarse_tile_shape[2]),
    )

    v_cycle(
        multigrid_kernels,
        queue,
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

    multigrid_kernels["prolongate_add_nearest_sparse"](
        queue,
        global_work_size,
        local_work_size,
        coarse_p,
        p,
        coarse_tile_map,
        current_tile_map,
        coarse_active_tiles,
        coarse_active_tile_count,
        np.int32(coarse_shape[0]),
        np.int32(coarse_shape[1]),
        np.int32(coarse_shape[2]),
        np.int32(current_shape[0]),
        np.int32(current_shape[1]),
        np.int32(current_shape[2]),
        np.int32(coarse_tile_shape[1]),
        np.int32(coarse_tile_shape[2]),
        np.int32(current_tile_shape[1]),
        np.int32(current_tile_shape[2]),
    )

    smooth(
        multigrid_kernels,
        queue,
        p,
        b,
        delta,
        post_smooth,
        tile_map=current_tile_map,
        active_tiles=current_active_tiles,
        active_tile_count=current_active_tile_count,
        field_shape=current_shape,
    )


def smooth(
    multigrid_kernels: dict[str, cl.Kernel],
    queue: cl.CommandQueue,
    p: Any,
    b: Any,
    delta: float,
    iterations: int,
    tile_map: Any,
    active_tiles: Any,
    active_tile_count: Any,
    field_shape: tuple[int, int, int],
) -> None:
    """
    Apply red-black Gauss-Seidel smoothing to one sparse multigrid level.
    """
    nx, ny, nz = field_shape

    tile_shape = tuple(
        (size + kernel_config.TILE_SIZE - 1) // kernel_config.TILE_SIZE
        for size in field_shape
    )

    local_work_size = kernel_config.THREADS_PER_BLOCK_3D

    is_level_0 = active_tiles is None

    if is_level_0:
        global_work_size = (
            tile_shape[0] * local_work_size[0],
            tile_shape[1] * local_work_size[1],
            tile_shape[2] * local_work_size[2],
        )

        kernel_name = "rbgs_step_level_0"

        kernel_args = (
            p,
            b,
            np.float32(delta),
            tile_map,
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            np.int32(tile_shape[0]),
            np.int32(tile_shape[1]),
            np.int32(tile_shape[2]),
        )

    else:
        active_tile_capacity = active_tiles.size // (3 * np.dtype(np.int32).itemsize)

        global_work_size = (
            active_tile_capacity * local_work_size[0],
            local_work_size[1],
            local_work_size[2],
        )

        kernel_name = "rbgs_step_sparse"

        kernel_args = (
            p,
            b,
            np.float32(delta),
            tile_map,
            active_tiles,
            active_tile_count,
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            np.int32(tile_shape[1]),
            np.int32(tile_shape[2]),
        )

    program = multigrid_kernels[kernel_name].program

    red_kernel = cl.Kernel(program, kernel_name)
    black_kernel = cl.Kernel(program, kernel_name)

    red_kernel.set_args(
        *kernel_args[:3],
        np.int32(0),
        *kernel_args[3:],
    )

    black_kernel.set_args(
        *kernel_args[:3],
        np.int32(1),
        *kernel_args[3:],
    )

    for _ in range(iterations):
        red_event = cl.enqueue_nd_range_kernel(
            queue,
            red_kernel,
            global_work_size,
            local_work_size,
        )
        record_kernel_event(red_kernel, red_event)
        black_event = cl.enqueue_nd_range_kernel(
            queue,
            black_kernel,
            global_work_size,
            local_work_size,
        )
        record_kernel_event(black_kernel, black_event)
    # Neumann boundary conditions
    boundary_global_work_size = (
        tile_shape[0] * local_work_size[0],
        tile_shape[1] * local_work_size[1],
        tile_shape[2] * local_work_size[2],
    )

    timed_boundary_kernel = multigrid_kernels["pressure_poisson_apply_neumann_bcs"]
    boundary_kernel = getattr(timed_boundary_kernel, "kernel", timed_boundary_kernel)

    boundary_kernel.set_args(
        p,
        tile_map,
        np.int32(nx),
        np.int32(ny),
        np.int32(nz),
        np.int32(tile_shape[0]),
        np.int32(tile_shape[1]),
        np.int32(tile_shape[2]),
    )

    boundary_event = cl.enqueue_nd_range_kernel(
        queue,
        boundary_kernel,
        boundary_global_work_size,
        local_work_size,
    )
    record_kernel_event(boundary_kernel, boundary_event)
