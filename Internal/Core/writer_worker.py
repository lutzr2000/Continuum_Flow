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
        grid_by_name = {
            grid_payload["name"]: grid_payload
            for grid_payload in grid_payloads
        }
        used = set()

        for grid_payload in grid_payloads:
            grid_name = grid_payload["name"]

            if grid_name in used:
                continue

            # Combine u/v/w into one vector velocity grid.
            if grid_name == "u" and "v" in grid_by_name and "w" in grid_by_name:
                u_arr, u_shm, _shape = open_scalar_array(grid_by_name["u"])
                v_arr, v_shm, _ = open_scalar_array(grid_by_name["v"])
                w_arr, w_shm, _ = open_scalar_array(grid_by_name["w"])

                open_shared_memory.extend([u_shm, v_shm, w_shm])

                vec_arr = np.stack((u_arr, v_arr, w_arr), axis=-1)

                grid_class = (
                    getattr(openvdb, "Vec3SGrid", None)
                    or getattr(openvdb, "VectorGrid", None)
                    or getattr(openvdb, "Vec3fGrid", None)
                )
                if grid_class is None:
                    raise AttributeError(
                        "No supported OpenVDB vector grid class available"
                    )

                grid = grid_class()
                grid.name = "velocity"
                grid.transform = transform

                if hasattr(grid, "saveFloatAsHalf"):
                    grid.saveFloatAsHalf = (precision == "float16")

                grid.copyFromArray(vec_arr)

                # Do not prune Vec3 grids here. This binding expects Vec3s tolerance.
                grids.append(grid)
                used.update(("u", "v", "w"))
                continue

            # Scalar grid path.
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
            used.add(grid_name)

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
