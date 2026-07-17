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


def write_vdb(payload):
    config = get_writer_config()
    simulations = config.get("simulations") or []
    if not isinstance(simulations, list) or not simulations:
        raise ValueError(
            "Writer config must contain a non-empty 'simulations' list."
        )
    simulation = simulations[0]

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
