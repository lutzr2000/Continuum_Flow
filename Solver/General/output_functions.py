import json
import os
import queue
import socket
import threading
from multiprocessing import shared_memory

import numpy as np

try:
    from numba import cuda
except ImportError:
    cuda = None

import Solver.Kernel_GPU.kernel_config as kernel_config

if cuda is not None:

    @cuda.jit(cache=True)
    def _cast_field_to_float16_kernel(source_field, output_field):
        """
        Cast one CUDA field into a float16 staging buffer.
        """
        i, j, k = cuda.grid(3)
        nx, ny, nz = source_field.shape

        if i >= nx or j >= ny or k >= nz:
            return

        output_field[i, j, k] = np.float16(source_field[i, j, k])

else:
    _cast_field_to_float16_kernel = None


def _is_cuda_array(field):
    """
    Return whether the passed field looks like a CUDA device array.
    """
    return cuda is not None and hasattr(field, "copy_to_host")


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
    """
    if not host_writer_endpoint:
        raise RuntimeError("No host VDB writer endpoint was configured.")

    os.makedirs(outpath, exist_ok=True)

    write_queue = queue.Queue(maxsize=queue_size)
    buffer_pool = queue.Queue(maxsize=queue_size)
    shared_memory_blocks = []
    storage_dtype = np.dtype(storage_dtype or np.float16)
    use_float16_staging = cuda is not None and storage_dtype == np.dtype(np.float16)
    device_staging_fields = {}
    field_layouts = {}

    for variable_name in buffered_variables:
        template_array = template_fields[variable_name]
        shape = tuple(template_array.shape)
        source_dtype = np.dtype(template_array.dtype)
        buffer_dtype = source_dtype

        if (
            use_float16_staging
            and _is_cuda_array(template_array)
            and source_dtype != storage_dtype
        ):
            device_staging_fields[variable_name] = cuda.device_array(
                shape, dtype=storage_dtype
            )
            buffer_dtype = storage_dtype

        field_layouts[variable_name] = {
            "shape": shape,
            "dtype": str(buffer_dtype),
            "storage_dtype": str(storage_dtype),
            "nbytes": int(np.prod(shape)) * buffer_dtype.itemsize,
        }

    for _ in range(queue_size):
        fields = {}
        for variable_name in buffered_variables:
            layout = field_layouts[variable_name]
            shape = layout["shape"]
            buffer_dtype = np.dtype(layout["dtype"])
            shm = shared_memory.SharedMemory(create=True, size=layout["nbytes"])
            shared_memory_blocks.append(shm)
            fields[variable_name] = {
                "array": np.ndarray(shape, dtype=buffer_dtype, buffer=shm.buf),
                "shape": shape,
                "dtype": layout["dtype"],
                "storage_dtype": layout["storage_dtype"],
                "shm_name": shm.name,
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
                host_writer_endpoint,
            ),
            daemon=True,
        )
        writer_thread.start()
        writer_threads.append(writer_thread)

    return (
        write_queue,
        buffer_pool,
        writer_threads,
        shared_memory_blocks,
        (device_staging_fields or None),
    )


def enqueue_device_output(
    write_queue,
    buffer_pool,
    buffered_variables,
    source_device_fields,
    device_staging_fields,
    output_index,
    time_value,
    output_field_config,
    sparse_threshold,
):
    """
    copies one output frame directly from CUDA device arrays into shared memory.
    """
    fields = buffer_pool.get()

    staged_variables = []
    if device_staging_fields:
        threadsperblock_3d = kernel_config.THREADS_PER_BLOCK_3D
        for variable_name in buffered_variables:
            staging_buffer = device_staging_fields.get(variable_name)
            if staging_buffer is None:
                continue
            source_field = source_device_fields[variable_name]
            blockspergrid_3d = kernel_config.volume_blocks_per_grid(
                source_field.shape, threadsperblock_3d
            )
            _cast_field_to_float16_kernel[blockspergrid_3d, threadsperblock_3d](
                source_field, staging_buffer
            )
            staged_variables.append(variable_name)

        if staged_variables:
            cuda.synchronize()

    for variable_name in buffered_variables:
        field_info = fields[variable_name]
        source_field = source_device_fields[variable_name]
        if device_staging_fields and variable_name in device_staging_fields:
            source_field = device_staging_fields[variable_name]
        source_field.copy_to_host(field_info["array"])

    write_queue.put(
        (
            int(output_index),
            float(time_value),
            fields,
            output_field_config,
            float(sparse_threshold),
        )
    )


def enqueue_host_output(
    write_queue,
    buffer_pool,
    buffered_variables,
    source_fields,
    output_index,
    time_value,
    output_field_config,
    sparse_threshold,
):
    """
    Copy one host-resident output frame into shared memory and enqueue it.
    """
    fields = buffer_pool.get()
    for variable_name in buffered_variables:
        np.copyto(fields[variable_name]["array"], source_fields[variable_name])
    write_queue.put(
        (
            int(output_index),
            float(time_value),
            fields,
            output_field_config,
            float(sparse_threshold),
        )
    )


def shutdown_output(write_queue, writer_threads, shared_memory_blocks):
    """
    waits for all forwarded output work to finish and releases shared-memory buffers.
    """
    write_queue.join()
    for _ in writer_threads:
        write_queue.put(None)
    write_queue.join()
    for writer_thread in writer_threads:
        writer_thread.join()
    cleanup_output_buffers(shared_memory_blocks)


def _domain_origin_from_shape(shape, delta):
    """
    Return the world-space origin used by the Continuum Flow domain preview.
    """
    nx, ny, _nz = shape
    return (
        -0.5 * float(nx) * float(delta),
        -0.5 * float(ny) * float(delta),
        0.0,
    )


def _build_grid_payload(fields, output_field_config):
    """
    Describe the VDB grids that should be assembled from buffered fields.
    """
    grid_entries = []

    velocity_cfg = output_field_config.get("u", {})
    if velocity_cfg.get("export") and all(
        component in fields for component in ("u", "v", "w")
    ):
        grid_entries.append(
            {
                "name": "velocity",
                "field_names": ["u", "v", "w"],
                "grid_type": "vector",
                "shape": fields["u"]["shape"],
                "dtype": fields["u"]["dtype"],
                "storage_dtype": fields["u"]["storage_dtype"],
                "sparse": bool(velocity_cfg.get("sparse", False)),
            }
        )

    scalar_grid_names = {
        "p": "pressure",
        "T": "temperature",
        "smoke": "density",
        "fuel": "fuel",
        "flame": "flame",
    }
    for field_name, grid_name in scalar_grid_names.items():
        field_cfg = output_field_config.get(field_name, {})
        if not field_cfg.get("export") or field_name not in fields:
            continue
        grid_entries.append(
            {
                "name": grid_name,
                "field_names": [field_name],
                "grid_type": "scalar",
                "shape": fields[field_name]["shape"],
                "dtype": fields[field_name]["dtype"],
                "storage_dtype": fields[field_name]["storage_dtype"],
                "sparse": bool(field_cfg.get("sparse", False)),
            }
        )

    return grid_entries


def create_writer_payload(
    fields, output_field_config, output_path, time_value, delta, sparse_threshold
):
    """
    builds the metadata package that is sent to Blender's host VDB writer.
    """
    grid_entries = _build_grid_payload(fields, output_field_config)
    first_field_shape = grid_entries[0]["shape"] if grid_entries else (0, 0, 0)

    payload = {
        "output_path": output_path,
        "time": float(time_value),
        "delta": float(delta),
        "transform": {
            "voxel_size": float(delta),
            "origin": _domain_origin_from_shape(first_field_shape, delta),
        },
        "sparse_threshold": float(sparse_threshold),
        "grids": [
            {
                "name": grid_entry["name"],
                "grid_type": grid_entry["grid_type"],
                "shape": grid_entry["shape"],
                "dtype": grid_entry["dtype"],
                "storage_dtype": grid_entry["storage_dtype"],
                "sparse": grid_entry["sparse"],
                "fields": {
                    field_name: {
                        "shape": fields[field_name]["shape"],
                        "dtype": fields[field_name]["dtype"],
                        "storage_dtype": fields[field_name]["storage_dtype"],
                        "shm_name": fields[field_name]["shm_name"],
                    }
                    for field_name in grid_entry["field_names"]
                },
            }
            for grid_entry in grid_entries
        ],
    }

    if any(grid_entry.get("sparse", False) for grid_entry in payload["grids"]):
        payload["sparse_mask_fields"] = {
            variable_name: {
                "shape": fields[variable_name]["shape"],
                "dtype": fields[variable_name]["dtype"],
                "storage_dtype": fields[variable_name]["storage_dtype"],
                "shm_name": fields[variable_name]["shm_name"],
            }
            for variable_name in ("smoke", "flame")
            if variable_name in fields
        }

    return payload


def _send_payload_to_host_writer(writer_file, writer_payload):
    """
    Send one VDB write job to Blender's host writer and wait for the ACK.
    """
    writer_file.write((json.dumps(writer_payload) + "\n").encode("utf-8"))
    writer_file.flush()
    response_line = writer_file.readline()
    if not response_line:
        raise RuntimeError("host VDB writer closed the connection")

    response = json.loads(response_line.decode("utf-8"))
    if response.get("status") != "ok":
        raise RuntimeError(response.get("message", "unknown host VDB writer error"))


def writer_thread_func(
    write_queue,
    buffer_pool,
    outpath,
    frame_start,
    delta,
    host_writer_endpoint,
):
    """
    connects to Blender's host writer and forwards queued output jobs to it.
    """
    writer_socket = None
    writer_file = None
    try:
        while True:
            item = write_queue.get()
            frame_idx = None
            try:
                if item is None:
                    if writer_file is not None and not writer_file.closed:
                        writer_file.write(b"__QUIT__\n")
                        writer_file.flush()
                        writer_file.close()
                        writer_file = None
                    break

                (
                    output_idx,
                    time_value,
                    fields,
                    output_field_config,
                    sparse_threshold,
                ) = item
                frame_idx = int(frame_start) + int(output_idx)
                vdb_output_path = os.path.join(outpath, f"frame_{frame_idx:06d}.vdb")
                writer_payload = create_writer_payload(
                    fields,
                    output_field_config,
                    vdb_output_path,
                    time_value,
                    delta,
                    sparse_threshold,
                )

                if writer_file is None or writer_file.closed:
                    writer_socket = socket.create_connection(
                        (
                            host_writer_endpoint["host"],
                            int(host_writer_endpoint["port"]),
                        )
                    )
                    writer_file = writer_socket.makefile("rwb")

                _send_payload_to_host_writer(writer_file, writer_payload)
            except Exception as exc:
                if frame_idx is None:
                    print(f"VDB writer thread connection shutdown failed: {exc}")
                else:
                    print(f"VDB write failed for frame {frame_idx}: {exc}")
                if writer_file is not None and not writer_file.closed:
                    writer_file.close()
                writer_file = None
                if writer_socket is not None:
                    writer_socket.close()
                writer_socket = None
            finally:
                if item is not None:
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
    """
    for shm in shared_memory_blocks:
        shm.close()
        shm.unlink()
