THREADS_PER_BLOCK_3D = (4, 4, 4)
THREADS_PER_BLOCK_2D = (4, 4)
TILE_SIZE = 4
TILE_DILATE = 2 # number of tiles
SPARSE_TILE_GROWTH_PERCENT = 5.0
REDUCTION_THREADS_PER_BLOCK = 256
MAX_REDUCTION_BLOCKS = 1024


def volume_blocks_per_grid(shape, threadsperblock=THREADS_PER_BLOCK_3D):
    """
    Return the 3D grid shape for one full-volume CUDA launch.
    """
    return tuple(
        (int(shape[axis]) + threadsperblock[axis] - 1) // threadsperblock[axis]
        for axis in range(3)
    )


def reduction_blocks_per_grid(
    total_size,
    threadsperblock=REDUCTION_THREADS_PER_BLOCK,
    max_blocks=MAX_REDUCTION_BLOCKS,
):
    """
    Return the 1D grid size for a reduction launch.
    """
    return min(max_blocks, (int(total_size) + threadsperblock - 1) // threadsperblock)
