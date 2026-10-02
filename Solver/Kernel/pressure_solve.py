import numpy as np
import pyopencl as cl

from typing import Any

import Solver.Kernel.kernel_config as kernel_config
import Solver.Kernel.multigrid as multigrid


def remove_rhs_mean(
    queue: cl.CommandQueue,
    pressure_solve_kernels: dict[str, cl.Kernel],
    b: Any,
    index_tile_map: Any,
    tile_shape: tuple[int, int, int],
    rhs_partial_sums: Any,
    rhs_partial_counts: Any,
    rhs_mean_buffer: Any,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    interior_cell_count = max(
        (nx - 2) * (ny - 2) * (nz - 2),
        1,
    )

    reduction_blocks = kernel_config.reduction_blocks_per_grid(interior_cell_count)

    reduction_threads = kernel_config.REDUCTION_THREADS_PER_BLOCK

    pressure_solve_kernels["rhs_sum_count_partial_kernel"](
        queue,
        (reduction_blocks * reduction_threads,),
        (reduction_threads,),
        b,
        index_tile_map,
        rhs_partial_sums,
        rhs_partial_counts,
        np.int32(nx),
        np.int32(ny),
        np.int32(nz),
        np.int32(tile_shape[1]),
        np.int32(tile_shape[2]),
    )

    pressure_solve_kernels["rhs_mean_kernel"](
        queue,
        (reduction_threads,),
        (reduction_threads,),
        rhs_partial_sums,
        rhs_partial_counts,
        np.int32(reduction_blocks),
        rhs_mean_buffer,
    )

    local_work_size = kernel_config.THREADS_PER_BLOCK_3D

    global_work_size = (
        tile_shape[0] * local_work_size[0],
        tile_shape[1] * local_work_size[1],
        tile_shape[2] * local_work_size[2],
    )

    pressure_solve_kernels["subtract_rhs_mean_kernel"](
        queue,
        global_work_size,
        local_work_size,
        b,
        rhs_mean_buffer,
        index_tile_map,
        np.int32(nx),
        np.int32(ny),
        np.int32(nz),
        np.int32(tile_shape[0]),
        np.int32(tile_shape[1]),
        np.int32(tile_shape[2]),
    )


def pressure_poisson_multigrid(
    pressure_solve_kernels: Any,
    multigrid_kernels: Any,
    queue: Any,
    global_work_size: Any,
    local_work_size: Any,
    u: Any,
    v: Any,
    w: Any,
    p: Any,
    T: Any,
    b: Any,
    dt: float,
    geometry_source_masks: Any,
    particle_source_masks: Any,
    source_noise_scales: Any,
    source_noise_amplitudes: Any,
    source_noise_seeds: Any,
    extra_pressure: Any,
    delta: float,
    rho: Any,
    expansion_rate: float,
    t_reference: float,
    index_tile_map: Any,
    tile_shape: tuple[int, int, int],
    u_initial: float,
    v_initial: float,
    w_initial: float,
    p_levels: Any,
    b_levels: Any,
    delta_levels: Any,
    multigrid_tile_maps: Any,
    multigrid_active_tiles: Any,
    multigrid_active_tile_counts: Any,
    multigrid_level_shapes: Any,
    num_vcycles: Any,
    rhs_partial_sums: Any,
    rhs_partial_counts: Any,
    rhs_mean_buffer: Any,
    nx: int,
    ny: int,
    nz: int,
) -> Any:
    r"""
    Solve the pressure Poisson equation with repeated multigrid V-cycles.

    The projection pressure satisfies

    .. math::

        \nabla^2 p = \frac{\rho}{\Delta t}\,\nabla\!\cdot\mathbf{u}.

    The right-hand side is assembled and made compatible with Neumann
    boundary conditions before the multigrid hierarchy reduces the residual.
    """
    pressure_solve_kernels["pressure_equation_right_side"](
        queue,
        global_work_size,
        local_work_size,
        u,
        v,
        w,
        b,
        np.float32(dt),
        np.float32(delta),
        np.float32(rho),
        index_tile_map,
        np.float32(u_initial),
        np.float32(v_initial),
        np.float32(w_initial),
        np.int32(nx),
        np.int32(ny),
        np.int32(nz),
        np.int32(tile_shape[0]),
        np.int32(tile_shape[1]),
        np.int32(tile_shape[2]),
    )

    pressure_solve_kernels["reset_inactive_pressure"](
        queue,
        global_work_size,
        local_work_size,
        p,
        index_tile_map,
        np.int32(nx),
        np.int32(ny),
        np.int32(nz),
        np.int32(tile_shape[0]),
        np.int32(tile_shape[1]),
        np.int32(tile_shape[2]),
    )

    for source_idx, source_mask in enumerate(geometry_source_masks):
        pressure_solve_kernels["add_artifical_divergence"](
            queue,
            global_work_size,
            local_work_size,
            T,
            source_mask,
            np.float32(extra_pressure[source_idx]),
            np.float32(source_noise_scales[source_idx]),
            np.float32(source_noise_amplitudes[source_idx]),
            np.float32(source_noise_seeds[source_idx]),
            np.float32(expansion_rate),
            np.float32(t_reference),
            b,
            index_tile_map,
            np.float32(rho),
            np.float32(delta),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            np.float32(dt),
            np.int32(tile_shape[0]),
            np.int32(tile_shape[1]),
            np.int32(tile_shape[2]),
        )

    for source_idx, source_mask in enumerate(particle_source_masks):
        pressure_solve_kernels["add_artifical_divergence"](
            queue,
            global_work_size,
            local_work_size,
            T,
            source_mask,
            np.float32(extra_pressure[source_idx]),
            np.float32(source_noise_scales[source_idx]),
            np.float32(source_noise_amplitudes[source_idx]),
            np.float32(source_noise_seeds[source_idx]),
            np.float32(expansion_rate),
            np.float32(t_reference),
            b,
            index_tile_map,
            np.float32(rho),
            np.float32(delta),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            np.float32(dt),
            np.int32(tile_shape[0]),
            np.int32(tile_shape[1]),
            np.int32(tile_shape[2]),
        )

    remove_rhs_mean(
        queue,
        pressure_solve_kernels,
        b,
        index_tile_map,
        tile_shape,
        rhs_partial_sums,
        rhs_partial_counts,
        rhs_mean_buffer,
        nx,
        ny,
        nz,
    )

    multigrid_active_tile_count_buffers = multigrid.build_coarse_tile_hierarchy(
        queue,
        multigrid_kernels,
        index_tile_map,
        tile_shape,
        multigrid_tile_maps,
        multigrid_level_shapes,
        multigrid_active_tiles,
        multigrid_active_tile_counts,
    )

    for _ in range(num_vcycles):
        multigrid.v_cycle(
            multigrid_kernels,
            queue,
            0,
            p_levels,
            b_levels,
            p,
            b,
            delta,
            delta_levels,
            pre_smooth=2,
            post_smooth=4,
            coarse_smooth=20,
            nx=nx,
            ny=ny,
            nz=nz,
            index_tile_map=index_tile_map,
            multigrid_tile_maps=multigrid_tile_maps,
            multigrid_active_tiles=multigrid_active_tiles,
            multigrid_active_tile_counts=multigrid_active_tile_count_buffers,
            multigrid_level_shapes=multigrid_level_shapes,
        )

    return p
