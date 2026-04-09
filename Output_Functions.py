import json
import os
import queue
import subprocess
import threading
from multiprocessing import shared_memory

import numpy as np

FIELD_NAMES = ('u', 'v', 'w', 'p', 'T', 'smoke', 'fuel', 'flame')


def create_output_field_map(u, v, w, p, T, smoke, fuel, flame):
    """
    collects all simulation fields in one dictionary for simpler output handling.

    Args:
        u (3d-array): x-velocity field
        v (3d-array): y-velocity field
        w (3d-array): z-velocity field
        p (3d-array): pressure field
        T (3d-array): temperature field
        smoke (3d-array): smoke field
        fuel (3d-array): fuel field
        flame (3d-array): flame field
    Returns:
        dict: field names mapped to numpy arrays
    """
    return dict(zip(FIELD_NAMES, (u, v, w, p, T, smoke, fuel, flame)))


def setup_output(outpath, output_variables, template_fields, queue_size, blender_python_exe, vdb_writer_script, delta):
    """
    prepares the shared-memory buffers and starts the persistent writer thread.

    Args:
        outpath (str): output directory for vdb files
        output_variables (list[str]): names of the fields that should be written
        template_fields (dict): field names mapped to template arrays
        queue_size (int): number of reusable buffer sets
        blender_python_exe (str): path to Blender's Python executable
        vdb_writer_script (str): path to the VDB writer script
        delta (float): grid spacing
    Returns:
        tuple: write queue, buffer queue, writer thread and list of shared memory blocks
    """
    os.makedirs(outpath, exist_ok=True)

    write_queue = queue.Queue(maxsize=queue_size)
    buffer_pool = queue.Queue(maxsize=queue_size)
    shared_memory_blocks = []

    for _ in range(queue_size):
        fields = {}
        for variable_name in output_variables:
            template_array = template_fields[variable_name]
            shm = shared_memory.SharedMemory(create=True, size=template_array.nbytes)
            shared_memory_blocks.append(shm)
            fields[variable_name] = {
                'array': np.ndarray(template_array.shape, dtype=template_array.dtype, buffer=shm.buf),
                'shape': template_array.shape,
                'dtype': str(template_array.dtype),
                'shm_name': shm.name,
            }
        buffer_pool.put(fields)

    writer_thread = threading.Thread(
        target=writer_thread_func,
        args=(write_queue, buffer_pool, outpath, delta, output_variables, blender_python_exe, vdb_writer_script),
        daemon=True,
    )
    writer_thread.start()

    return write_queue, buffer_pool, writer_thread, shared_memory_blocks


def enqueue_output(write_queue, buffer_pool, output_variables, source_fields, output_index, time_value):
    """
    copies one output frame into shared memory and queues it for writing.

    Args:
        write_queue (queue.Queue): queue with pending output jobs
        buffer_pool (queue.Queue): queue with reusable shared-memory buffers
        output_variables (list[str]): names of the fields that should be written
        source_fields (dict): field names mapped to current simulation arrays
        output_index (int): output frame index
        time_value (float): physical simulation time
    Returns:
        None
    """
    fields = buffer_pool.get()
    for variable_name in output_variables:
        np.copyto(fields[variable_name]['array'], source_fields[variable_name])
    write_queue.put((int(output_index), float(time_value), fields))


def shutdown_output(write_queue, writer_thread, shared_memory_blocks):
    """
    waits for all output work to finish and releases the shared-memory buffers.

    Args:
        write_queue (queue.Queue): queue with pending output jobs
        writer_thread (threading.Thread): background writer thread
        shared_memory_blocks (list): shared memory objects used for output buffering
    Returns:
        None
    """
    write_queue.join()
    write_queue.put(None)
    write_queue.join()
    writer_thread.join()
    cleanup_output_buffers(shared_memory_blocks)


def create_writer_payload(fields, output_variables, output_path, time_value, delta):
    """
    builds the metadata package that is sent to the persistent VDB writer.

    Args:
        fields (dict): output buffer with shared-memory numpy views
        output_variables (list[str]): names of the fields that should be written
        output_path (str): full path of the target vdb file
        time_value (float): physical simulation time
        delta (float): grid spacing
    Returns:
        dict: serializable writer payload
    """
    return {
        'output_path': output_path,
        'time': float(time_value),
        'delta': float(delta),
        'fields': {
            variable_name: {
                'shape': fields[variable_name]['shape'],
                'dtype': fields[variable_name]['dtype'],
                'shm_name': fields[variable_name]['shm_name'],
            }
            for variable_name in output_variables
        },
    }


def writer_thread_func(write_queue, buffer_pool, outpath, delta, output_variables, blender_python_exe, vdb_writer_script):
    """
    runs a persistent Blender Python process and forwards queued output jobs to it.

    Args:
        write_queue (queue.Queue): queue with pending output jobs
        buffer_pool (queue.Queue): queue with reusable shared-memory buffers
        outpath (str): output directory for vdb files
        delta (float): grid spacing
        output_variables (list[str]): names of the fields that should be written
        blender_python_exe (str): path to Blender's Python executable
        vdb_writer_script (str): path to the VDB writer script
    Returns:
        None
    """
    writer_process = None
    try:
        writer_process = subprocess.Popen(
            [blender_python_exe, vdb_writer_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        while True:
            item = write_queue.get()
            if item is None:
                if writer_process.stdin is not None:
                    writer_process.stdin.write('__QUIT__\n')
                    writer_process.stdin.flush()
                    writer_process.stdin.close()
                write_queue.task_done()
                break

            output_idx, time_value, fields = item
            vdb_output_path = os.path.join(outpath, f'frame_{output_idx:06d}.vdb')
            writer_payload = create_writer_payload(fields, output_variables, vdb_output_path, time_value, delta)

            try:
                writer_process.stdin.write(json.dumps(writer_payload) + '\n')
                writer_process.stdin.flush()
                response_line = writer_process.stdout.readline()
                if not response_line:
                    raise RuntimeError(writer_process.stderr.read().strip())

                response = json.loads(response_line)
                if response.get('status') != 'ok':
                    raise RuntimeError(response.get('message', 'unknown VDB writer error'))
            except Exception as exc:
                print(f'VDB write failed for output {output_idx}: {exc}')

            buffer_pool.put(fields)
            write_queue.task_done()
    finally:
        if writer_process is not None:
            writer_process.wait()


def cleanup_output_buffers(shared_memory_blocks):
    """
    closes and releases all shared-memory output buffers.

    Args:
        shared_memory_blocks (list): shared memory objects used for output buffering
    Returns:
        None
    """
    for shm in shared_memory_blocks:
        shm.close()
        shm.unlink()
