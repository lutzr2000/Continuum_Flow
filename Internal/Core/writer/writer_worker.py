import json
import os
import sys
from multiprocessing import shared_memory

import numpy as np
import openvdb


WRITER_CONFIG = {}


def _load_writer_config_from_argv(argv):
    if len(argv) < 2:
        return {}
    return json.loads(argv[1])


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


def copy_sparse_tiles_into_grid(grid, sparse_arr, tile_map, dense_shape, tile_size):
    nx, ny, nz = (int(dense_shape[0]), int(dense_shape[1]), int(dense_shape[2]))
    tiles_x, tiles_y, tiles_z = tile_map.shape

    for tile_i in range(tiles_x):
        cell_i_start = tile_i * tile_size
        if cell_i_start >= nx:
            break

        cell_i_end = min(cell_i_start + tile_size, nx)

        for tile_j in range(tiles_y):
            cell_j_start = tile_j * tile_size
            if cell_j_start >= ny:
                break

            cell_j_end = min(cell_j_start + tile_size, ny)

            for tile_k in range(tiles_z):
                tile_index = int(tile_map[tile_i, tile_j, tile_k])
                if tile_index < 0:
                    continue

                cell_k_start = tile_k * tile_size
                if cell_k_start >= nz:
                    break

                cell_k_end = min(cell_k_start + tile_size, nz)

                tile_values = sparse_arr[
                    tile_index,
                    : cell_i_end - cell_i_start,
                    : cell_j_end - cell_j_start,
                    : cell_k_end - cell_k_start,
                ]

                grid.copyFromArray(
                    np.ascontiguousarray(tile_values),
                    ijk=(cell_i_start, cell_j_start, cell_k_start),
                )


def write_vdb(payload):
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

    origin = (-0.5 * nx * delta, -0.5 * ny * delta, 0.0)
    output_vdb_path = payload["output_path"]

    grids = []
    open_shared_memory = []

    transform = openvdb.createLinearTransform(voxelSize=delta)
    transform.postTranslate(origin)

    try:
        grid_payloads = list(payload["grids"])
        tile_map = None

        if "tile_map" in payload:
            tile_map, tile_map_shm = open_tile_map(payload)
            open_shared_memory.append(tile_map_shm)

        for grid_payload in grid_payloads:
            grid_name = grid_payload["name"]

            # Always write each field as its own scalar grid, including u/v/w.
            arr, shm, _shape = open_scalar_array(grid_payload)
            open_shared_memory.append(shm)

            grid = openvdb.FloatGrid(background=0.0)
            grid.name = grid_name
            grid.transform = transform

            if hasattr(grid, "saveFloatAsHalf"):
                grid.saveFloatAsHalf = (precision == "float16")

            if grid_payload.get("layout") == "sparse_tiles":
                copy_sparse_tiles_into_grid(
                    grid,
                    arr,
                    tile_map,
                    tuple(grid_payload["dense_shape"]),
                    int(grid_payload["tile_size"]),
                )
            else:
                grid.copyFromArray(arr)
            prune_scalar_grid(grid)

            grids.append(grid)

        os.makedirs(os.path.dirname(output_vdb_path), exist_ok=True)
        output_tmp_path = f"{output_vdb_path}.tmp"
        try:
            openvdb.write(output_tmp_path, grids=grids)
            os.replace(output_tmp_path, output_vdb_path)
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
            sys.stdout.write('{"status": "ok"}\n')
        except Exception as exc:
            sys.stdout.write(
                json.dumps({"status": "error", "message": str(exc)}) + "\n"
            )
        sys.stdout.flush()


if __name__ == "__main__":
    WRITER_CONFIG = _load_writer_config_from_argv(sys.argv)
    main()
