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
) -> None:
    if not fields:
        return

    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / f"frame_{frame:06d}.vdb"

    grid_names = {
        "smoke": "density",
        "temperature": "temperature",
        "pressure": "pressure",
        "flame": "flame",
    }

    scalar_fields = {
        grid_names[name]: values
        for name, values in fields.items()
        if name in grid_names
    }

    vector_fields = {}

    if all(name in fields for name in ("u", "v", "w")):
        vector_fields["velocity"] = (
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
