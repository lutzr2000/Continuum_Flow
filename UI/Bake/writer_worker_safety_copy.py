import json
import os
import sys
from multiprocessing import shared_memory
from pathlib import Path
from time import perf_counter

import numpy as np
import openvdb


WRITER_CONFIG = {}
WRITER_TIMING_STATS = {}
WRITER_TIMING_FRAME_COUNT = 0
WRITER_TIMING_OUTPUT_DIR = None


def _record_timing(name, elapsed):
    WRITER_TIMING_STATS[name] = WRITER_TIMING_STATS.get(name, 0.0) + float(elapsed)


def _set_timing_output_dir_from_payload(payload):
    global WRITER_TIMING_OUTPUT_DIR

    output_path = payload.get("output_path")
    if not output_path:
        return

    output_dir = Path(output_path).resolve().parent
    if WRITER_TIMING_OUTPUT_DIR is None:
        WRITER_TIMING_OUTPUT_DIR = output_dir


def _write_timing_summary():
    if WRITER_TIMING_OUTPUT_DIR is None or not WRITER_TIMING_STATS:
        return

    summary_path = WRITER_TIMING_OUTPUT_DIR / f"writer_timing_summary_{os.getpid()}.txt"
    total_runtime = WRITER_TIMING_STATS.get("write_vdb.total", 0.0)

    lines = [
        "Writer timing summary:",
        f"  pid: {os.getpid()}",
        f"  frames: {WRITER_TIMING_FRAME_COUNT}",
    ]

    for name, elapsed in sorted(
        WRITER_TIMING_STATS.items(), key=lambda item: item[1], reverse=True
    ):
        share = (elapsed / total_runtime * 100.0) if total_runtime > 0.0 else 0.0
        lines.append(f"  {name}: {elapsed:.3f} s ({share:.1f}%)")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    global WRITER_TIMING_FRAME_COUNT

    _set_timing_output_dir_from_payload(payload)

    total_start = perf_counter()
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

    section_start = perf_counter()
    transform = openvdb.createLinearTransform(voxelSize=delta)
    transform.postTranslate(origin)
    _record_timing("write_vdb.prepare_transform", perf_counter() - section_start)

    try:
        for grid_payload in payload["grids"]:
            grid_name = grid_payload["name"]
            _field_name, field_info = next(iter(grid_payload["fields"].items()))

            shape = tuple(field_info["shape"])
            shm_name = field_info["shm_name"]

            dtype = np.float16 if precision == "float16" else np.float32

            section_start = perf_counter()
            shm = shared_memory.SharedMemory(name=shm_name)
            open_shared_memory.append(shm)
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            _record_timing("write_vdb.open_shared_array", perf_counter() - section_start)

            section_start = perf_counter()
            # copyFromArray arbeitet je nach Binding meist mit float32 am besten
            if arr.dtype != np.float32:
                arr_for_vdb = np.asarray(arr, dtype=np.float32)
            else:
                arr_for_vdb = arr
            _record_timing("write_vdb.prepare_array", perf_counter() - section_start)

            section_start = perf_counter()
            grid = openvdb.FloatGrid(background=0.0)
            grid.name = grid_name
            grid.transform = transform
            _record_timing("write_vdb.prepare_grid", perf_counter() - section_start)

            section_start = perf_counter()
            grid.copyFromArray(arr_for_vdb)
            _record_timing("write_vdb.copy_from_array", perf_counter() - section_start)

            section_start = perf_counter()
            try:
                if sparse_threshold > 0.0:
                    grid.prune(sparse_threshold)
                else:
                    grid.prune()
            except TypeError:
                grid.prune()
            except AttributeError:
                pass
            _record_timing("write_vdb.prune_grid", perf_counter() - section_start)

            grids.append(grid)

        section_start = perf_counter()
        os.makedirs(os.path.dirname(output_vdb_path), exist_ok=True)
        _record_timing("write_vdb.ensure_output_dir", perf_counter() - section_start)

        section_start = perf_counter()
        openvdb.write(output_vdb_path, grids=grids)
        _record_timing("write_vdb.openvdb_write", perf_counter() - section_start)

        WRITER_TIMING_FRAME_COUNT += 1

    finally:
        section_start = perf_counter()
        for shm in open_shared_memory:
            shm.close()
        _record_timing("write_vdb.cleanup", perf_counter() - section_start)
        _record_timing("write_vdb.total", perf_counter() - total_start)


def main():
    """
    Run a persistent JSON-lines VDB writer process.
    """
    try:
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
    finally:
        _write_timing_summary()


if __name__ == "__main__":
    WRITER_CONFIG = _load_writer_config_from_argv(sys.argv)
    main()
