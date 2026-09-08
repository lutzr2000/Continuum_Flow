import numpy as np

THREADS_PER_BLOCK_3D: tuple[int, int, int] = (4, 4, 4)
THREADS_PER_BLOCK_2D: tuple[int, int] = (4, 4)

TILE_SIZE: int = 4
TILE_DILATE: int = 3  # Number of tiles

SPARSE_TILE_GROWTH_PERCENT: float = 5.0

REDUCTION_THREADS_PER_BLOCK: int = 256
MAX_REDUCTION_BLOCKS: int = 1024

GPU_FIELD_DTYPE = np.float32


def volume_blocks_per_grid(
    shape: tuple[int, int, int],
    threadsperblock: tuple[int, int, int] = THREADS_PER_BLOCK_3D,
) -> tuple[int, int, int]:
    """
    Compute the CUDA grid dimensions for a full 3D volume launch.

    The number of blocks along each axis is determined by dividing the
    corresponding volume dimension by the number of threads per block and
    rounding up to ensure that the complete volume is covered.

    Parameters
    ----------
    shape
        Number of elements along the x-, y-, and z-axis of the volume.
    threadsperblock
        Number of CUDA threads per block along each axis.

    Returns
    -------
    tuple[int, int, int]
        Number of CUDA blocks required along the x-, y-, and z-axis.
    """
    return tuple(
        (int(shape[axis]) + threadsperblock[axis] - 1) // threadsperblock[axis]
        for axis in range(3)
    )


def reduction_blocks_per_grid(
    total_size: int,
    threadsperblock: int = REDUCTION_THREADS_PER_BLOCK,
    max_blocks: int = MAX_REDUCTION_BLOCKS,
) -> int:
    """
    Compute the CUDA grid size for a 1D reduction launch.

    The required number of blocks is computed from the total number of
    elements and the number of threads per block. The result is capped at
    ``max_blocks`` to limit the number of reduction blocks.

    Parameters
    ----------
    total_size
        Total number of elements processed by the reduction.
    threadsperblock
        Number of CUDA threads per block.
    max_blocks
        Maximum number of CUDA blocks that may be launched.

    Returns
    -------
    int
        Number of CUDA blocks required for the reduction, capped at
        ``max_blocks``.
    """
    return min(
        max_blocks,
        (int(total_size) + threadsperblock - 1) // threadsperblock,
    )
