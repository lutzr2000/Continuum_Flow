from typing import Any

import math

import numpy as np
from numba import njit, prange

import Solver.Kernel_CPU.kernel_config as kernel_config

IDENTITY_4 = np.eye(4)
ZERO_4 = np.zeros((4, 4))


# ------------setup------------------
def update_source_tile_mask(
    source_tile_mask: Any,
    source_base_masks: Any,
    t: float,
    delta: float,
    origin: tuple[int, int, int],
) -> None:
    """
    Rebuild the coarse tile-activity mask from all transformed source bounds.
    """
    source_tile_mask[:] = False

    tile_size = kernel_config.TILE_SIZE

    for base_masks in source_base_masks:
        for entry in base_masks:
            mesh_object = entry["mesh_object"]
            voxels = entry["voxels"]

            matrix, _ = get_matrix_data(
                entry["matrix_times"],
                entry["matrix_matrices"],
                entry["matrix_rates"],
                t,
            )

            bounds_min = np.asarray(voxels["bounds_min"])
            bounds_max = np.asarray(voxels["bounds_max"])

            center = (bounds_min + bounds_max) * 0.5
            extent = (bounds_max - bounds_min) * 0.5

            linear = matrix[:3, :3]
            translation = matrix[:3, 3]

            world_center = linear @ center + translation
            world_extent = np.abs(linear) @ extent

            world_min = world_center - world_extent
            world_max = world_center + world_extent

            cell_min = np.floor((world_min - origin) / delta)
            cell_max = np.ceil((world_max - origin) / delta)

            tile_min = np.floor_divide(
                cell_min,
                tile_size,
            )

            tile_max = np.floor_divide(
                cell_max,
                tile_size,
            )

            tile_min = np.maximum(tile_min, 0)

            tile_max = np.minimum(
                tile_max,
                np.asarray(source_tile_mask.shape) - 1,
            )

            mark_source_tiles(
                source_tile_mask,
                int(tile_min[0]),
                int(tile_min[1]),
                int(tile_min[2]),
                int(tile_max[0]),
                int(tile_max[1]),
                int(tile_max[2]),
            )


@njit(cache=True, parallel=True)
def mark_source_tiles(
    source_tile_mask: Any,
    offset_i: Any,
    offset_j: Any,
    offset_k: Any,
    max_i: Any,
    max_j: Any,
    max_k: Any,
) -> None:
    """
    Mark one tile region as occupied by source geometry.
    """
    size_i = max_i - offset_i + 1
    size_j = max_j - offset_j + 1
    size_k = max_k - offset_k + 1

    total = size_i * size_j * size_k

    for idx in prange(total):
        i = idx // (size_j * size_k)
        remainder = idx % (size_j * size_k)
        j = remainder // size_k
        k = remainder % size_k

        i += offset_i
        j += offset_j
        k += offset_k

        if (
            i < source_tile_mask.shape[0]
            and j < source_tile_mask.shape[1]
            and k < source_tile_mask.shape[2]
        ):
            source_tile_mask[i, j, k] = True
