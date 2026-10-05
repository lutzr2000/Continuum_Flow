from pathlib import Path

import numpy as np

import continuum_vdb_writer


def _dense_scalar(
    sparse_values: np.ndarray,
    index_tile_map: np.ndarray,
    grid_shape: tuple[int, int, int],
) -> np.ndarray:
    tile_size = int(sparse_values.shape[1]) if sparse_values.shape[0] else 4
    padded_shape = tuple(
        int(tile_count) * tile_size for tile_count in index_tile_map.shape
    )
    dense = np.zeros(padded_shape, dtype=np.float32)

    active_tiles = index_tile_map >= 0
    if np.any(active_tiles):
        tile_view = dense.reshape(
            index_tile_map.shape[0],
            tile_size,
            index_tile_map.shape[1],
            tile_size,
            index_tile_map.shape[2],
            tile_size,
        ).transpose(0, 2, 4, 1, 3, 5)
        tile_view[active_tiles] = sparse_values[index_tile_map[active_tiles]]

    nx, ny, nz = grid_shape
    if dense.shape != grid_shape:
        dense = np.ascontiguousarray(dense[:nx, :ny, :nz])
    return dense


def _dense_vector(
    fields: dict[str, np.ndarray],
    index_tile_map: np.ndarray,
    grid_shape: tuple[int, int, int],
) -> np.ndarray:
    """Prepare future velocity output as contiguous ``(x, y, z, 3)`` data."""
    return np.stack(
        [
            _dense_scalar(fields[name], index_tile_map, grid_shape)
            for name in ("u", "v", "w")
        ],
        axis=-1,
    )


def write_snapshot(
    *,
    output_path: Path,
    frame: int,
    fields: dict[str, np.ndarray],
    index_tile_map: np.ndarray,
    grid_shape: tuple[int, int, int],
    voxel_size: float,
    origin: tuple[float, float, float],
    precision: str,
) -> None:
    """Write one snapshot. The current wheel path intentionally writes smoke only."""
    smoke = fields.get("smoke")
    if smoke is None:
        return

    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / f"frame_{frame:06d}.vdb"
    density = _dense_scalar(smoke, index_tile_map, grid_shape)
    continuum_vdb_writer.write_float_grid(
        filepath,
        "density",
        density,
        voxel_size,
        origin,
        "none",
        precision,
        0.0,
        0.0,
        "fog_volume",
    )
