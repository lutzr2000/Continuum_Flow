import numpy as np
import pyopencl as cl

from typing import Any

import Solver.Kernel.kernel_config as kernel_config

FIELD_DTYPE = kernel_config.FIELD_DTYPE


def compute_new_timestep(
    queue: cl.CommandQueue,
    timestep_kernels: dict[str, cl.Kernel],
    u: Any,
    v: Any,
    w: Any,
    index_tile_map: Any,
    tile_shape: tuple[int, int, int],
    active_tile_count: int,
    maxima: Any,
    partial_maxima: Any,
    delta: float,
    cfl_max: float,
    max_dt: float | None = None,
) -> float:
    r"""
    Compute a stable timestep from the component-wise velocity maxima.

    The timestep satisfies the configured CFL limit independently along every
    coordinate axis.
    """
    eps = 1e-12

    if active_tile_count <= 0:
        return float(max_dt)

    total_tile_count = int(np.prod(tile_shape))

    threads = kernel_config.REDUCTION_THREADS_PER_BLOCK
    blocks = kernel_config.reduction_blocks_per_grid(total_tile_count)

    global_work_size = (blocks * threads,)

    local_work_size = (threads,)

    local_memory_size = threads * np.dtype(FIELD_DTYPE).itemsize

    timestep_kernels["velocity_maxima_partial"](
        queue,
        global_work_size,
        local_work_size,
        u,
        v,
        w,
        index_tile_map,
        partial_maxima,
        np.int32(total_tile_count),
        np.int32(tile_shape[1]),
        np.int32(tile_shape[2]),
        cl.LocalMemory(local_memory_size),
        cl.LocalMemory(local_memory_size),
        cl.LocalMemory(local_memory_size),
    )

    timestep_kernels["velocity_maxima_final"](
        queue,
        (threads,),
        local_work_size,
        partial_maxima,
        maxima,
        np.int32(blocks),
        cl.LocalMemory(local_memory_size),
        cl.LocalMemory(local_memory_size),
        cl.LocalMemory(local_memory_size),
    )

    maxima_host = np.empty(
        3,
        dtype=FIELD_DTYPE,
    )

    cl.enqueue_copy(
        queue,
        maxima_host,
        maxima,
    ).wait()

    abs_u_max, abs_v_max, abs_w_max = maxima_host

    cfl_delta = cfl_max * delta

    dt_conv = min(
        cfl_delta / max(float(abs_u_max), eps),
        cfl_delta / max(float(abs_v_max), eps),
        cfl_delta / max(float(abs_w_max), eps),
    )

    return min(
        dt_conv,
        float(max_dt),
    )
