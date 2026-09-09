import numpy as np

THREADS_PER_BLOCK_3D: tuple[int, int, int] = (4, 4, 4)
THREADS_PER_BLOCK_2D: tuple[int, int] = (4, 4)

TILE_SIZE: int = 4
TILE_DILATE: int = 3  # Number of tiles

SPARSE_TILE_GROWTH_PERCENT: float = 5.0

REDUCTION_THREADS_PER_BLOCK: int = 256
MAX_REDUCTION_BLOCKS: int = 1024

GPU_FIELD_DTYPE = np.float32
