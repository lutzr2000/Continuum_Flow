import pyopencl as cl
from typing import Any

import Solver.Kernel.kernel_config as kernel_config

FIELD_DTYPE = kernel_config.FIELD_DTYPE


def reset_pools(queue: cl.CommandQueue, dst_pools: Any, fill_pool: Any) -> None:
    """
    Reset scratch pools from a shared fill pool.
    """
    for dst_pool in dst_pools:
        cl.enqueue_copy(
            queue,
            dst_pool,
            fill_pool,
        )
