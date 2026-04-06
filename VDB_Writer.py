import os
import sys
import json
from multiprocessing import shared_memory

import numpy as np
import openvdb


def write_vdb(payload):
    """
    creates one vdb file from shared-memory field data.

    Args:
        payload (dict): metadata for the target file and shared-memory fields
    Returns:
        None
    """
    output_vdb_path = payload['output_path']
    grids = []
    open_shared_memory = []
    for variable_name, field_info in payload['fields'].items():
        shm = shared_memory.SharedMemory(name=field_info['shm_name'])
        open_shared_memory.append(shm)
        array = np.ndarray(
            tuple(field_info['shape']),
            dtype=np.dtype(field_info['dtype']),
            buffer=shm.buf,
        )
        grid = openvdb.FloatGrid()
        grid.name = variable_name
        grid.copyFromArray(np.asarray(array, dtype=np.float32))
        grids.append(grid)

    output_dir = os.path.dirname(output_vdb_path)
    openvdb.write(output_vdb_path, grids=grids)
    for shm in open_shared_memory:
        shm.close()


def main():
    """
    keeps a persistent writer process alive and handles incoming write jobs.

    Args:
        None
    Returns:
        None
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == '__QUIT__':
            break

        try:
            payload = json.loads(line)
            write_vdb(payload)
            sys.stdout.write('{"status": "ok"}\n')
        except Exception as exc:
            sys.stdout.write(json.dumps({'status': 'error', 'message': str(exc)}) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
