import json
import os
import queue
import socket
import threading
from multiprocessing import shared_memory

import numpy as np


def setup_output(
    outpath,
    frame_start,
    buffered_variables,
    template_fields,
    queue_size,
    forwarder_count,
    delta,
    host_writer_endpoint,
    storage_dtype=None,
):
    """
    prepares shared-memory buffers and starts host-writer forwarder threads.

    Args:
        outpath (str): output directory for vdb files
        frame_start (int): Blender frame number used for the first written VDB
        buffered_variables (list[str]): names of the fields that should be copied into shared memory
        template_fields (dict): field names mapped to CUDA device arrays
        queue_size (int): number of reusable buffer sets
        forwarder_count (int): number of host-writer forwarder threads
        delta (float): grid spacing
        host_writer_endpoint (dict): host/port for Blender's in-process writer
        storage_dtype (dtype-like): target dtype used for VDB file storage
    Returns:
        tuple: write queue, buffer queue, forwarder threads and list of shared-memory blocks
    """
    if not host_writer_endpoint:
        raise RuntimeError('No host VDB writer endpoint was configured.')

    os.makedirs(outpath, exist_ok=True)

    write_queue = queue.Queue(maxsize=queue_size)
    buffer_pool = queue.Queue(maxsize=queue_size)
    shared_memory_blocks = []
    storage_dtype = np.dtype(storage_dtype or np.float16)

    for _ in range(queue_size):
        fields = {}
        for variable_name in buffered_variables:
            template_array = template_fields[variable_name]
            shape = tuple(template_array.shape)
            source_dtype = np.dtype(template_array.dtype)
            nbytes = int(np.prod(shape)) * source_dtype.itemsize
            shm = shared_memory.SharedMemory(create=True, size=nbytes)
            shared_memory_blocks.append(shm)
            fields[variable_name] = {
                'array': np.ndarray(shape, dtype=source_dtype, buffer=shm.buf),
                'shape': shape,
                'dtype': str(source_dtype),
                'storage_dtype': str(storage_dtype),
                'shm_name': shm.name,
            }
        buffer_pool.put(fields)

    writer_threads = []
    for _ in range(forwarder_count):
        writer_thread = threading.Thread(
            target=writer_thread_func,
            args=(
                write_queue,
                buffer_pool,
                outpath,
                frame_start,
                delta,
                buffered_variables,
                host_writer_endpoint,
            ),
            daemon=True,
        )
        writer_thread.start()
        writer_threads.append(writer_thread)

    return write_queue, buffer_pool, writer_threads, shared_memory_blocks


def enqueue_device_output(
    write_queue,
    buffer_pool,
    buffered_variables,
    source_device_fields,
    output_index,
    time_value,
    output_field_config,
):
    """
    copies one output frame directly from CUDA device arrays into shared memory.

    Args:
        write_queue (queue.Queue): queue with pending output jobs
        buffer_pool (queue.Queue): queue with reusable shared-memory buffers
        buffered_variables (list[str]): names of the fields that should be copied into shared memory
        source_device_fields (dict): field names mapped to CUDA device arrays
        output_index (int): output frame index
        time_value (float): physical simulation time
    Returns:
        None
    """
    fields = buffer_pool.get()
    for variable_name in buffered_variables:
        field_info = fields[variable_name]
        source_device_fields[variable_name].copy_to_host(field_info['array'])
    write_queue.put((int(output_index), float(time_value), fields, output_field_config))


def shutdown_output(write_queue, writer_threads, shared_memory_blocks):
    """
    waits for all forwarded output work to finish and releases shared-memory buffers.

    Args:
        write_queue (queue.Queue): queue with pending output jobs
        writer_threads (list[threading.Thread]): background host-writer forwarder threads
        shared_memory_blocks (list): shared memory objects used for output buffering
    Returns:
        None
    """
    write_queue.join()
    for _ in writer_threads:
        write_queue.put(None)
    write_queue.join()
    for writer_thread in writer_threads:
        writer_thread.join()
    cleanup_output_buffers(shared_memory_blocks)


def _domain_origin_from_shape(shape, delta):
    """Return the world-space origin used by the BlenderCFD domain preview."""
    nx, ny, _nz = shape
    return (
        -0.5 * float(nx) * float(delta),
        -0.5 * float(ny) * float(delta),
        0.0,
    )


def _build_grid_payload(fields, output_field_config):
    """Describe the VDB grids that should be assembled from buffered fields."""
    grid_entries = []

    velocity_cfg = output_field_config.get('u', {})
    if velocity_cfg.get('export') and all(component in fields for component in ('u', 'v', 'w')):
        grid_entries.append(
            {
                'name': 'velocity',
                'field_names': ['u', 'v', 'w'],
                'grid_type': 'vector',
                'shape': fields['u']['shape'],
                'dtype': fields['u']['dtype'],
                'storage_dtype': fields['u']['storage_dtype'],
                'sparse': bool(velocity_cfg.get('sparse', False)),
            }
        )

    scalar_grid_names = {
        'p': 'pressure',
        'T': 'temperature',
        'smoke': 'density',
        'fuel': 'fuel',
        'flame': 'flame',
    }
    for field_name, grid_name in scalar_grid_names.items():
        field_cfg = output_field_config.get(field_name, {})
        if not field_cfg.get('export') or field_name not in fields:
            continue
        grid_entries.append(
            {
                'name': grid_name,
                'field_names': [field_name],
                'grid_type': 'scalar',
                'shape': fields[field_name]['shape'],
                'dtype': fields[field_name]['dtype'],
                'storage_dtype': fields[field_name]['storage_dtype'],
                'sparse': bool(field_cfg.get('sparse', False)),
            }
        )

    return grid_entries


def create_writer_payload(fields, output_field_config, output_path, time_value, delta):
    """
    builds the metadata package that is sent to Blender's host VDB writer.

    Args:
        fields (dict): output buffer with shared-memory numpy views
        output_field_config (dict): concrete export settings for the buffered fields
        output_path (str): full path of the target vdb file
        time_value (float): physical simulation time
        delta (float): grid spacing
    Returns:
        dict: serializable writer payload
    """
    grid_entries = _build_grid_payload(fields, output_field_config)
    first_field_shape = grid_entries[0]['shape'] if grid_entries else (0, 0, 0)

    payload = {
        'output_path': output_path,
        'time': float(time_value),
        'delta': float(delta),
        'transform': {
            'voxel_size': float(delta),
            'origin': _domain_origin_from_shape(first_field_shape, delta),
        },
        'grids': [
            {
                'name': grid_entry['name'],
                'grid_type': grid_entry['grid_type'],
                'shape': grid_entry['shape'],
                'dtype': grid_entry['dtype'],
                'storage_dtype': grid_entry['storage_dtype'],
                'sparse': grid_entry['sparse'],
                'fields': {
                    field_name: {
                        'shape': fields[field_name]['shape'],
                        'dtype': fields[field_name]['dtype'],
                        'storage_dtype': fields[field_name]['storage_dtype'],
                        'shm_name': fields[field_name]['shm_name'],
                    }
                    for field_name in grid_entry['field_names']
                },
            }
            for grid_entry in grid_entries
        ],
    }

    if any(grid_entry.get('sparse', False) for grid_entry in payload['grids']):
        payload['sparse_mask_fields'] = {
            variable_name: {
                'shape': fields[variable_name]['shape'],
                'dtype': fields[variable_name]['dtype'],
                'storage_dtype': fields[variable_name]['storage_dtype'],
                'shm_name': fields[variable_name]['shm_name'],
            }
            for variable_name in ('smoke', 'flame')
            if variable_name in fields
        }

    return payload


def _send_payload_to_host_writer(writer_file, writer_payload):
    """Send one VDB write job to Blender's host writer and wait for the ACK."""
    writer_file.write((json.dumps(writer_payload) + '\n').encode('utf-8'))
    writer_file.flush()
    response_line = writer_file.readline()
    if not response_line:
        raise RuntimeError('host VDB writer closed the connection')

    response = json.loads(response_line.decode('utf-8'))
    if response.get('status') != 'ok':
        raise RuntimeError(response.get('message', 'unknown host VDB writer error'))


def writer_thread_func(
    write_queue,
    buffer_pool,
    outpath,
    frame_start,
    delta,
    buffered_variables,
    host_writer_endpoint,
):
    """
    connects to Blender's host writer and forwards queued output jobs to it.

    Args:
        write_queue (queue.Queue): queue with pending output jobs
        buffer_pool (queue.Queue): queue with reusable shared-memory buffers
        outpath (str): output directory for vdb files
        frame_start (int): Blender frame number used for the first written VDB
        delta (float): grid spacing
        buffered_variables (list[str]): names of the fields that should be copied into shared memory
        host_writer_endpoint (dict): host/port for Blender's in-process writer
    Returns:
        None
    """
    writer_socket = None
    writer_file = None
    try:
        writer_socket = socket.create_connection(
            (host_writer_endpoint['host'], int(host_writer_endpoint['port']))
        )
        writer_file = writer_socket.makefile('rwb')

        while True:
            item = write_queue.get()
            if item is None:
                writer_file.write(b'__QUIT__\n')
                writer_file.flush()
                writer_file.close()
                write_queue.task_done()
                break

            output_idx, time_value, fields, output_field_config = item
            frame_idx = int(frame_start) + int(output_idx)
            vdb_output_path = os.path.join(outpath, f'frame_{frame_idx:06d}.vdb')
            writer_payload = create_writer_payload(fields, output_field_config, vdb_output_path, time_value, delta)

            try:
                _send_payload_to_host_writer(writer_file, writer_payload)
            except Exception as exc:
                print(f'VDB write failed for frame {frame_idx}: {exc}')

            buffer_pool.put(fields)
            write_queue.task_done()
    finally:
        if writer_file is not None and not writer_file.closed:
            writer_file.close()
        if writer_socket is not None:
            writer_socket.close()


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
