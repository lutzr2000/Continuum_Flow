from pathlib import Path

import numpy as np

import continuum_vdb_writer


def write_snapshot(
    *,
    output_path: Path,
    frame: int,
    fields: dict[str, np.ndarray],
    index_tile_map: np.ndarray,
    grid_shape: tuple[int, int, int],
    voxel_size: float,
    origin: tuple[float, float, float],
    compression: str,
    precision: str,
    field_names: dict[str, str],
) -> None:
    filepath = output_path / f"frame_{frame:06d}.vdb"

    grid_names = {
        name: field_names[name]
        for name in ("smoke", "temperature", "pressure", "fuel", "flame")
    }

    scalar_fields = {
        grid_names[name]: values
        for name, values in fields.items()
        if name in grid_names
    }

    vector_fields = {}

    if all(name in fields for name in ("u", "v", "w")):
        vector_fields[field_names["velocity"]] = (
            fields["u"],
            fields["v"],
            fields["w"],
        )

    continuum_vdb_writer.write_sparse_grids(
        filepath,
        scalar_fields,
        vector_fields,
        index_tile_map,
        grid_shape,
        voxel_size,
        origin,
        compression,
        precision,
        0.0,
    )
