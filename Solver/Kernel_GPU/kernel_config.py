THREADS_PER_BLOCK_3D = (8, 8, 8)
THREADS_PER_BLOCK_2D = (4, 4)
ACTIVE_TILE_SIZE = 4
ACTIVE_TILE_THREADS_PER_BLOCK = (ACTIVE_TILE_SIZE, ACTIVE_TILE_SIZE, ACTIVE_TILE_SIZE)
ACTIVE_TILE_MASK_THREADS_PER_BLOCK = (4, 4, 4)
ACTIVE_TILE_PADDING_CELLS = 4
REDUCTION_THREADS_PER_BLOCK = 256
MAX_REDUCTION_BLOCKS = 1024
MAX_VELOCITY_INCREMENT_FACTOR = 0.5


def volume_blocks_per_grid(shape, threadsperblock=THREADS_PER_BLOCK_3D):
    """
    Return the 3D grid shape for one full-volume CUDA launch.
    """
    return tuple(
        (int(shape[axis]) + threadsperblock[axis] - 1) // threadsperblock[axis]
        for axis in range(3)
    )


def active_tile_shape(shape, tile_size=ACTIVE_TILE_SIZE):
    """
    Return the number of active-mask tiles along each axis.
    """
    return tuple((int(shape[axis]) + tile_size - 1) // tile_size for axis in range(3))


def active_tile_padding_tiles(
    tile_size=ACTIVE_TILE_SIZE, padding_cells=ACTIVE_TILE_PADDING_CELLS
):
    """
    Return the tile-radius used to dilate the active scalar region.
    """
    return max(0, (int(padding_cells) + tile_size - 1) // tile_size)


def reduction_blocks_per_grid(
    total_size,
    threadsperblock=REDUCTION_THREADS_PER_BLOCK,
    max_blocks=MAX_REDUCTION_BLOCKS,
):
    """
    Return the 1D grid size for a reduction launch.
    """
    return min(max_blocks, (int(total_size) + threadsperblock - 1) // threadsperblock)
