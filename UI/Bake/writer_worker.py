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

    output_cfg = simulation.get("outputs", [{}])[0]
    sparse_threshold = float(output_cfg.get("sparse_threshold", 0.0))
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
        for grid_payload in payload["grids"]:
            grid_name = grid_payload["name"]
            field_name, field_info = next(iter(grid_payload["fields"].items()))

            shape = tuple(field_info["shape"])
            shm_name = field_info["shm_name"]

            dtype = np.float16 if precision == "float16" else np.float32

            shm = shared_memory.SharedMemory(name=shm_name)
            open_shared_memory.append(shm)

            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

            # copyFromArray arbeitet je nach Binding meist mit float32 am besten
            if arr.dtype != np.float32:
                arr_for_vdb = np.asarray(arr, dtype=np.float32)
            else:
                arr_for_vdb = arr

            grid = openvdb.FloatGrid(background=0.0)
            grid.name = grid_name
            grid.transform = transform

            grid.copyFromArray(arr_for_vdb)

            try:
                if sparse_threshold > 0.0:
                    grid.prune(sparse_threshold)
                else:
                    grid.prune()
            except TypeError:
                grid.prune()
            except AttributeError:
                pass

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
