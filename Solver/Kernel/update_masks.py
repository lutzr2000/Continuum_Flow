from typing import Any
import numpy as np
import pyopencl as cl

import Solver.Kernel.kernel_config as kernel_config
import Solver.Kernel.helper as helper


def update_source_tile_mask(
    queue: cl.CommandQueue,
    sources_program: cl.Program,
    source_tile_mask: Any,
    source_base_masks: Any,
    tile_shape: tuple[int, int, int],
    t: float,
    delta: float,
    origin: tuple[int, int, int],
) -> None:
    """
    Rebuild the coarse tile-activity mask from all transformed source bounds.

    Each source mesh's animated world-space bounds are converted to clamped
    grid-tile bounds, then the enclosed tile range is marked on the GPU.
    """
    cl.enqueue_fill_buffer(
        queue,
        source_tile_mask,
        np.uint8(0),
        0,
        source_tile_mask.size,
    )

    tile_size = kernel_config.TILE_SIZE

    for base_masks in source_base_masks:
        for entry in base_masks:
            mesh_object = entry["mesh_object"]
            voxels = entry["voxels"]

            matrix, _ = helper.get_matrix_data(
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
                np.asarray(tile_shape) - 1,
            )

            if np.any(tile_min > tile_max):
                continue

            sources_program.mark_source_tiles(
                queue,
                (
                    int(tile_max[0] - tile_min[0] + 1),
                    int(tile_max[1] - tile_min[1] + 1),
                    int(tile_max[2] - tile_min[2] + 1),
                ),
                None,
                source_tile_mask,
                np.int32(tile_shape[0]),
                np.int32(tile_shape[1]),
                np.int32(tile_shape[2]),
                np.int32(tile_min[0]),
                np.int32(tile_min[1]),
                np.int32(tile_min[2]),
            )
