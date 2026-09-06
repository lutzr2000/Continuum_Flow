import json
import os
import sys
from multiprocessing import shared_memory

def _load_writer_config_from_argv(argv):
    if len(argv) < 2:
        return {}
    return json.loads(argv[1])


WRITER_CONFIG = _load_writer_config_from_argv(sys.argv)

for _dependency_path in reversed(WRITER_CONFIG.get("parent_sys_path") or ()):
    if _dependency_path and _dependency_path not in sys.path:
        sys.path.insert(0, _dependency_path)

import numpy as np
import numba
import openvdb


VDB_BRICK_TILES = 8


def get_writer_config():
    return WRITER_CONFIG


def open_scalar_array(grid_payload):
    _field_name, field_info = next(iter(grid_payload["fields"].items()))
    shape = tuple(field_info["shape"])
    shm = shared_memory.SharedMemory(name=field_info["shm_name"])
    arr = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
    return arr, shm, shape


def open_tile_map(payload):
    tile_map_info = payload["tile_map"]
    shape = tuple(tile_map_info["shape"])
    shm = shared_memory.SharedMemory(name=tile_map_info["shm_name"])
    arr = np.ndarray(shape, dtype=np.int32, buffer=shm.buf)
    return arr, shm


def open_active_tiles(payload):
    active_tiles_info = payload["active_tiles"]
    shape = tuple(active_tiles_info["shape"])
    shm = shared_memory.SharedMemory(name=active_tiles_info["shm_name"])
    arr = np.ndarray(shape, dtype=np.int32, buffer=shm.buf)
    return arr, shm, int(active_tiles_info["count"])


def prune_scalar_grid(grid):
    try:
        grid.prune()
    except TypeError:
        try:
            grid.prune()
        except Exception:
            pass
    except AttributeError:
        pass


@numba.njit(cache=True, nogil=True)
def fill_vdb_brick(
    brick_values,
    sparse_arr,
    tile_metadata,
    brick_origin_i,
    brick_origin_j,
    brick_origin_k,
    tile_size,
):
    """Scatter one metadata group directly into a reusable dense brick."""
    for tile_number in range(tile_metadata.shape[0]):
        tile_index = int(tile_metadata[tile_number, 0])
        local_i = int(tile_metadata[tile_number, 1]) - brick_origin_i
        local_j = int(tile_metadata[tile_number, 2]) - brick_origin_j
        local_k = int(tile_metadata[tile_number, 3]) - brick_origin_k

        for i in range(tile_size):
            for j in range(tile_size):
                for k in range(tile_size):
                    brick_values[
                        local_i + i,
                        local_j + j,
                        local_k + k,
                    ] = sparse_arr[tile_index, i, j, k]


def copy_sparse_tiles_into_grid(
    grid,
    sparse_arr,
    active_tiles,
    active_tile_count,
    tile_size,
):
    """Copy sparse tiles in bricks to minimize Python-to-OpenVDB calls."""
    if active_tile_count <= 0:
        return

    brick_cell_size = int(VDB_BRICK_TILES * tile_size)

    metadata = active_tiles[:active_tile_count]
    brick_coordinates = metadata[:, 1:4] // brick_cell_size

    order = np.lexsort(
        (
            brick_coordinates[:, 2],
            brick_coordinates[:, 1],
            brick_coordinates[:, 0],
        )
    )

    sorted_metadata = np.ascontiguousarray(metadata[order])
    sorted_coordinates = brick_coordinates[order]

    group_starts = np.concatenate(
        (
            np.asarray((0,), dtype=np.int64),
            np.flatnonzero(
                np.any(
                    np.diff(sorted_coordinates, axis=0),
                    axis=1,
                )
            )
            + 1,
            np.asarray((active_tile_count,), dtype=np.int64),
        )
    )

    brick_values = None

    for group_index in range(len(group_starts) - 1):
        start = int(group_starts[group_index])
        end = int(group_starts[group_index + 1])

        brick_tiles = sorted_metadata[start:end]

        # For a single tile there is no benefit in allocating/filling
        # an entire dense brick.
        if end - start == 1:
            tile_meta = brick_tiles[0]

            tile_values = sparse_arr[
                int(tile_meta[0]),
                :tile_size,
                :tile_size,
                :tile_size,
            ]

            grid.copyFromArray(
                tile_values,
                ijk=(
                    int(tile_meta[1]),
                    int(tile_meta[2]),
                    int(tile_meta[3]),
                ),
            )
            continue

        brick_key = sorted_coordinates[start]

        brick_origin = (
            int(brick_key[0]) * brick_cell_size,
            int(brick_key[1]) * brick_cell_size,
            int(brick_key[2]) * brick_cell_size,
        )

        if brick_values is None:
            brick_values = np.empty(
                (
                    brick_cell_size,
                    brick_cell_size,
                    brick_cell_size,
                ),
                dtype=sparse_arr.dtype,
            )

        brick_values.fill(0)

        fill_vdb_brick(
            brick_values,
            sparse_arr,
            brick_tiles,
            brick_origin[0],
            brick_origin[1],
            brick_origin[2],
            tile_size,
        )

        grid.copyFromArray(
            brick_values,
            ijk=brick_origin,
        )


def write_vdb(payload):
    output_vdb_path = payload["output_path"]

    config = get_writer_config()
    simulation = config.get("simulation") or {}

    if not isinstance(simulation, dict) or not simulation:
        raise ValueError(
            "Writer config must contain a non-empty 'simulation' object."
        )

    output_cfg = simulation.get("outputs", [{}])[0]
    precision = output_cfg.get("precision", "float32")

    delta = float(simulation.get("domain").get("resolution"))
    nx = int(simulation["domain"]["grid"]["nx"])
    ny = int(simulation["domain"]["grid"]["ny"])

    origin = (
        -0.5 * nx * delta,
        -0.5 * ny * delta,
        0.0,
    )

    transform = openvdb.createLinearTransform(
        voxelSize=delta
    )
    transform.postTranslate(origin)

    grids = []
    open_shared_memory = []

    try:
        grid_payloads = list(payload["grids"])

        tile_map = None
        active_tiles = None
        active_tile_count = 0

        if "tile_map" in payload:
            tile_map, tile_map_shm = open_tile_map(payload)
            open_shared_memory.append(tile_map_shm)

        if "active_tiles" in payload:
            (
                active_tiles,
                active_tiles_shm,
                active_tile_count,
            ) = open_active_tiles(payload)

            open_shared_memory.append(active_tiles_shm)

        for grid_payload in grid_payloads:
            grid_name = grid_payload["name"]

            arr, shm, _shape = open_scalar_array(grid_payload)
            open_shared_memory.append(shm)

            grid = openvdb.FloatGrid(background=0.0)
            grid.name = grid_name
            grid.transform = transform

            if hasattr(grid, "saveFloatAsHalf"):
                grid.saveFloatAsHalf = (
                    precision == "float16"
                )

            if grid_payload.get("layout") == "sparse_tiles":
                copy_sparse_tiles_into_grid(
                    grid,
                    arr,
                    active_tiles,
                    active_tile_count,
                    int(grid_payload["tile_size"]),
                )
            else:
                grid.copyFromArray(arr)

            prune_scalar_grid(grid)

            grids.append(grid)

        os.makedirs(
            os.path.dirname(output_vdb_path),
            exist_ok=True,
        )

        output_tmp_path = f"{output_vdb_path}.tmp"

        try:
            openvdb.write(
                output_tmp_path,
                grids=grids,
            )

            os.replace(
                output_tmp_path,
                output_vdb_path,
            )

        except Exception:
            try:
                if os.path.exists(output_tmp_path):
                    os.remove(output_tmp_path)
            except OSError:
                pass

            raise

    finally:
        for shm in open_shared_memory:
            shm.close()


def main():
    """
    Run a persistent JSON-lines VDB writer process.
    """
    for line in sys.stdin:
        line = line.strip()

        if not line:
            continue

        if line == "__QUIT__":
            break

        try:
            write_vdb(json.loads(line))

            sys.stdout.write(
                '{"status": "ok"}\n'
            )

        except Exception as exc:
            sys.stdout.write(
                json.dumps(
                    {
                        "status": "error",
                        "message": str(exc),
                    }
                )
                + "\n"
            )

        sys.stdout.flush()


if __name__ == "__main__":
    main()