import json
import os
import sys

import numpy as np
import openvdb

from multiprocessing import shared_memory


def open_array(info):
    shm = shared_memory.SharedMemory(name=info["shm_name"])

    array = np.ndarray(
        tuple(info["shape"]),
        dtype=np.dtype(info["dtype"]),
        buffer=shm.buf,
    )

    return array, shm


def write_vdb(payload):
    tile_size = int(payload["tile_size"])
    delta = float(payload["delta"])
    nx = int(payload["nx"])
    ny = int(payload["ny"])

    transform = openvdb.createLinearTransform(voxelSize=delta)
    transform.postTranslate(
        (
            -0.5 * nx * delta,
            -0.5 * ny * delta,
            0.0,
        )
    )

    tile_map, tile_map_shm = open_array(payload["tile_map"])

    active_positions = np.argwhere(tile_map >= 0)
    active_slots = tile_map[
        active_positions[:, 0],
        active_positions[:, 1],
        active_positions[:, 2],
    ].copy()

    del tile_map
    tile_map_shm.close()

    grids = []

    for name, info in payload["fields"].items():
        array, shm = open_array(info)

        grid = openvdb.FloatGrid(background=0.0)
        grid.name = name
        grid.transform = transform

        for position, slot in zip(active_positions, active_slots):
            i, j, k = position

            grid.copyFromArray(
                array[int(slot)],
                ijk=(
                    int(i) * tile_size,
                    int(j) * tile_size,
                    int(k) * tile_size,
                ),
            )

        grid.prune()
        grids.append(grid)

        del array
        shm.close()

    output_path = payload["output_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    temp_path = output_path + ".tmp"

    try:
        openvdb.write(temp_path, grids=grids)
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main():
    for line in sys.stdin:
        line = line.strip()

        if line == "__QUIT__":
            break

        try:
            write_vdb(json.loads(line))
            response = {"status": "ok"}

        except Exception as exc:
            response = {
                "status": "error",
                "message": str(exc),
            }

        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
