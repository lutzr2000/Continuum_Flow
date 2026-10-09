import numpy as np
import pyopencl as cl

TILE_SIZE: int = 4
SPARSE_TILE_GROWTH_PERCENT: float = 5.0
MAX_REDUCTION_BLOCKS: int = 1024
REDUCTION_THREADS_PER_BLOCK: int = 256
THREADS_PER_BLOCK_3D: tuple[int, int, int] = (4, 4, 4)
THREADS_PER_BLOCK_2D: tuple[int, int] = (4, 4)
FIELD_DTYPE = np.float32


def configure_for_device(device, backend: str) -> None:
    """Choose safe work sizes from the selected OpenCL device limits."""
    global MAX_REDUCTION_BLOCKS, REDUCTION_THREADS_PER_BLOCK
    global THREADS_PER_BLOCK_3D, THREADS_PER_BLOCK_2D

    max_group_size = int(device.max_work_group_size)
    max_items = tuple(int(value) for value in device.max_work_item_sizes)
    local_memory = int(device.local_mem_size)

    THREADS_PER_BLOCK_3D = (TILE_SIZE, TILE_SIZE, TILE_SIZE)
    THREADS_PER_BLOCK_2D = (TILE_SIZE, TILE_SIZE)

    backend = str(backend).upper()
    preferred_reduction_size = 64 if backend == "CPU" else 256

    # The velocity reduction uses three float arrays in local memory.
    local_memory_limit = max(1, local_memory // (3 * np.dtype(FIELD_DTYPE).itemsize))
    reduction_limit = min(
        preferred_reduction_size,
        max_group_size,
        max_items[0],
        local_memory_limit,
    )

    REDUCTION_THREADS_PER_BLOCK = 1 << (reduction_limit.bit_length() - 1)

    compute_units = max(1, int(device.max_compute_units))
    groups_per_unit = 4 if backend == "CPU" else 32
    MAX_REDUCTION_BLOCKS = max(1, min(1024, compute_units * groups_per_unit))


def constrain_to_compiled_kernels(kernels, device) -> None:
    """Apply the strictest work-group limit reported by compiled kernels."""
    global REDUCTION_THREADS_PER_BLOCK

    limits = [
        int(
            kernel.get_work_group_info(
                cl.kernel_work_group_info.WORK_GROUP_SIZE,
                device,
            )
        )
        for kernel in kernels
    ]
    if not limits:
        return

    compiled_limit = min(limits)

    limit = min(REDUCTION_THREADS_PER_BLOCK, compiled_limit)
    REDUCTION_THREADS_PER_BLOCK = 1 << (limit.bit_length() - 1)


def reduction_blocks_per_grid(total_count: int) -> int:
    blocks = (
        total_count + REDUCTION_THREADS_PER_BLOCK - 1
    ) // REDUCTION_THREADS_PER_BLOCK

    return max(
        1,
        min(
            blocks,
            MAX_REDUCTION_BLOCKS,
        ),
    )
