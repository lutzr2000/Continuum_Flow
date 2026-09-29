import numpy as np

TILE_SIZE: int = 4
SPARSE_TILE_GROWTH_PERCENT: float = 5.0
MAX_REDUCTION_BLOCKS: int = 1024
REDUCTION_THREADS_PER_BLOCK: int = 256
THREADS_PER_BLOCK_3D: tuple[int, int, int] = (4, 4, 4)
FIELD_DTYPE = np.float32


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
