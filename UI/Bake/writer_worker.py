import json
import os
import sys
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import openvdb


WRITER_CONFIG = {}

def _load_writer_config_from_argv(argv):
    if len(argv) < 2:
        return {}

    config_path = Path(argv[1]).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Writer config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def get_writer_config():
    return WRITER_CONFIG


def write_vdb(payload):
    config = get_writer_config()
    simulations = config.get("simulations", {})
    simulation = simulations[0]
    sparse_threshold = float(simulation.get("outputs", [{}])[0].get("sparse_threshold", 0.0))

    delta = simulation.get("domain").get("resolution")
    nx = simulation["domain"]["grid"]["nx"]
    ny = simulation["domain"]["grid"]["ny"]

    origin = (-0.5 * nx * delta, -0.5 * ny * delta, 0.0)

    output_vdb_path = payload["output_path"]

    grids = []
    open_shared_memory = []

    try:
        for grid_payload in payload["grids"]:
            grid_name = grid_payload["name"]

            field_name, field_info = next(iter(grid_payload["fields"].items()))

            shape = tuple(field_info["shape"])
            shm_name = field_info["shm_name"]

            precision = simulation.get("outputs", [{}])[0].get("precision", "float32")
            dtype = np.float16 if precision == "float16" else np.float32

            shm = shared_memory.SharedMemory(name=shm_name)
            open_shared_memory.append(shm)

            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

            grid = openvdb.FloatGrid(background=0.0)
            grid.name = grid_name

            transform = openvdb.createLinearTransform(voxelSize=float(delta))
            transform.postTranslate(origin)
            grid.transform = transform

            accessor = grid.getAccessor()

            active = np.abs(arr) > sparse_threshold
            xs, ys, zs = np.nonzero(active)

            for x, y, z in zip(xs, ys, zs):
                accessor.setValueOn(
                    (int(x), int(y), int(z)),
                    float(arr[x, y, z])
                )

            try:
                grid.prune()
            except AttributeError:
                grid.pruneGrid()

            grids.append(grid)

        os.makedirs(os.path.dirname(output_vdb_path), exist_ok=True)
        openvdb.write(output_vdb_path, grids=grids)

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
