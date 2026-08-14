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


def copy_sparse_tiles_into_grid(grid, sparse_arr, active_tiles, active_tile_count):
    for tile_meta in active_tiles[:active_tile_count]:
        tile_index = int(tile_meta[0])
        cell_i_start = int(tile_meta[1])
        cell_j_start = int(tile_meta[2])
        cell_k_start = int(tile_meta[3])
        size_i = int(tile_meta[4])
        size_j = int(tile_meta[5])
        size_k = int(tile_meta[6])

        tile_values = sparse_arr[
            tile_index,
            :size_i,
            :size_j,
            :size_k,
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
        active_tiles = None
        active_tile_count = 0

        if "tile_map" in payload:
            tile_map, tile_map_shm = open_tile_map(payload)
            open_shared_memory.append(tile_map_shm)
        if "active_tiles" in payload:
            active_tiles, active_tiles_shm, active_tile_count = open_active_tiles(
                payload
            )
            open_shared_memory.append(active_tiles_shm)

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
                    active_tiles,
                    active_tile_count,
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
