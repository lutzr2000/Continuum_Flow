import json
import os
import queue
import socket
import threading
from multiprocessing import shared_memory
from time import perf_counter
import numpy as np
from numba import cuda

import Solver.Kernel_GPU.kernel_config as kernel_config


@cuda.jit(cache=True)
def cast_float16(source_field, output_field):
    """
    Cast one CUDA field into a float16 staging buffer.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = source_field.shape

    if i >= nx or j >= ny or k >= nz:
        return

    output_field[i, j, k] = np.float16(source_field[i, j, k])


def setup_output(
    simulations,
    outpath,
    shape
):
    """
    Prepares one shared-memory buffer and starts one host-writer thread.
    """
    os.makedirs(outpath, exist_ok=True)

    shared_memory_blocks = []
    field_to_output = {}
    field_layouts = {}
    ouput_list = simulations["outputs"][0]["variables"]

    for variable_name in ouput_list:

        if simulations.get("outputs")[0].get("precision") == "float16":
            buffer_dtype = np.float16
        else:
            buffer_dtype = np.float32

        field_to_output[variable_name] = cuda.device_array(
            shape,
            dtype=buffer_dtype,
        )

        field_layouts[variable_name] = {
            "shape": shape,
            "dtype": str(buffer_dtype),
            "storage_dtype": str(simulations.get("outputs")[0].get("precision")),
            "nbytes": int(np.prod(shape)) * np.dtype(buffer_dtype).itemsize,
        }

    fields = {}

    for variable_name in ouput_list:
        layout = field_layouts[variable_name]
        shape = layout["shape"]
        buffer_dtype = np.dtype(layout["dtype"])

        shm = shared_memory.SharedMemory(
            create=True,
            size=layout["nbytes"],
        )
        shared_memory_blocks.append(shm)

        fields[variable_name] = {
            "array": np.ndarray(shape, dtype=buffer_dtype, buffer=shm.buf),
            "shape": shape,
            "dtype": layout["dtype"],
            "storage_dtype": layout["storage_dtype"],
            "shm_name": shm.name,
        }

    return shared_memory_blocks, fields, field_to_output


def enqueue_device_output(
    simulations,
    fields,
    sim_fields,
    field_to_output,
    output_index,
    t,
    delta,
    origin
):
    """
    Copies one output frame from CUDA device arrays into shared memory
    and sends it directly to the host VDB writer.
    """
    buffered_variables = simulations["outputs"][0]["variables"]
    host_writer_endpoint = simulations.get("host_vdb_writer")
    sparse_threshold = simulations.get("settings").get("adaptive_domain_threshold")
    frame_start = simulations.get("settings").get("start_frame")
    outpath = simulations.get("outputs").get("output_path")
    ouput_list = simulations["outputs"][0]["variables"]

    for variable_name in buffered_variables:
        source_field = sim_fields[variable_name]

        if simulations["outputs"][0].get("precision", "float32") == "float16":
            staging_buffer = field_to_output[variable_name]
            blockspergrid_3d = kernel_config.volume_blocks_per_grid(
                source_field.shape,
                kernel_config.THREADS_PER_BLOCK_3D,
            )
            cast_float16[
                blockspergrid_3d,
                kernel_config.THREADS_PER_BLOCK_3D,
            ](source_field, staging_buffer)
            source_field = staging_buffer

        source_field.copy_to_host(fields[variable_name]["array"])

    frame_idx = int(frame_start) + int(output_index)
    vdb_output_path = os.path.join(outpath, f"frame_{frame_idx:06d}.vdb")

    writer_payload = create_writer_payload(
        fields,
        ouput_list,
        vdb_output_path,
        t,
        delta,
        origin,
        sparse_threshold,
    )

    with socket.create_connection(
        (
            host_writer_endpoint["host"],
            int(host_writer_endpoint["port"]),
        )
    ) as writer_socket:
        writer_file = writer_socket.makefile("rwb")
        _send_payload_to_host_writer(writer_file, writer_payload)
        writer_file.close()


def writer_thread_func(
    write_queue,
    buffer_pool,
    outpath,
    frame_start,
    delta,
    origin,
    host_writer_endpoint,
):
    """
    connects to Blender's host writer and forwards queued output jobs to it.
    """
    while True:
        item = write_queue.get()

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
            origin,
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

        buffer_pool.put(fields)
        write_queue.task_done()
            
        writer_file.close()
        writer_socket.close()



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
    fields, output_field_config, output_path, time_value, delta, origin, sparse_threshold
):
    """
    builds the metadata package that is sent to Blender's host VDB writer.
    """
    grid_entries = _build_grid_payload(fields, output_field_config)

    payload = {
        "output_path": output_path,
        "time": float(time_value),
        "delta": float(delta),
        "transform": {
            "voxel_size": float(delta),
            "origin": origin,
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


def shutdown_output(shared_memory_blocks):
    """
    closes and releases all shared-memory output buffers.
    """
    for shm in shared_memory_blocks:
        shm.close()
        shm.unlink()