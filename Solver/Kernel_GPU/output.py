import json
import os
import queue
import socket
import threading
from multiprocessing import shared_memory
import numpy as np
from numba import cuda

import Solver.Kernel_GPU.kernel_config as kernel_config


def _enabled_output_field_names(output_fields):
    """
    Return only output field names that are explicitly enabled in the config.
    """
    enabled_fields = []
    for field_name, field_cfg in (output_fields or {}).items():
        if bool((field_cfg or {}).get("enabled", False)):
            enabled_fields.append(field_name)

    if "velocity" in enabled_fields:
        enabled_fields.remove("velocity")
        enabled_fields.extend(["u", "v", "w"])
    return enabled_fields


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
    output_fields  = simulations["outputs"][0]["fields"]
    output_list = _enabled_output_field_names(output_fields)

    for variable_name in output_list:

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
            "dtype": np.dtype(buffer_dtype).name,
            "nbytes": int(np.prod(shape)) * np.dtype(buffer_dtype).itemsize,
        }

    fields = {}

    for variable_name in output_list:
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
            "shm_name": shm.name,
        }

    mask_dtype = np.bool_
    mask_nbytes = int(np.prod(shape)) * np.dtype(mask_dtype).itemsize

    mask_shm = shared_memory.SharedMemory(
        create=True,
        size=mask_nbytes,
    )
    shared_memory_blocks.append(mask_shm)

    fields["output_mask"] = {
        "array": np.ndarray(shape, dtype=mask_dtype, buffer=mask_shm.buf),
        "shape": tuple(shape),
        "dtype": np.dtype(mask_dtype).name,
        "shm_name": mask_shm.name,
    }

    writer_socket = socket.create_connection(
        (
            simulations["outputs"][0]["host_vdb_writer"]["host"],
            int(simulations["outputs"][0]["host_vdb_writer"]["port"]),
        )
    )
    writer_file = writer_socket.makefile("rwb")

    return shared_memory_blocks, fields, field_to_output, writer_socket, writer_file


def enqueue_device_output(
    simulations,
    fields,
    sim_fields,
    field_to_output,
    output_index,
    t,
    writer_file,
    output_mask
):
    """
    Copies one output frame from CUDA device arrays into shared memory
    and sends it directly to the host VDB writer.
    """
    output_cfg = ((simulations.get("outputs") or [None])[0]) or {}
    frame_start = simulations.get("settings").get("start_frame")
    outpath = output_cfg.get("output_path")
    output_fields  = output_cfg["fields"]
    output_list = _enabled_output_field_names(output_fields)

    for variable_name in output_list:
        source_field = sim_fields[variable_name]

        if output_cfg.get("precision", "float32") == "float16":
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

    output_mask.copy_to_host(fields["output_mask"]["array"])
    frame_idx = int(frame_start) + int(output_index)
    output_path = os.path.join(outpath, f"frame_{frame_idx:06d}.vdb")

    writer_payload = create_writer_payload(
        fields,
        output_list,
        output_path,
        t,
    )
    _send_payload_to_host_writer(writer_file, writer_payload)


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


def _vdb_grid_name(field_name):
    """
    Map internal solver field names to their exported VDB grid names.
    """
    return "density" if field_name == "smoke" else field_name

def create_writer_payload(
    fields,
    output_list,
    output_path,
    time_value,
):
    payload = {
        "output_path": output_path,
        "time": float(time_value),
        "active_mask": {
            "shape": fields["output_mask"]["shape"],
            "dtype": fields["output_mask"]["dtype"],
            "shm_name": fields["output_mask"]["shm_name"],
        },
        "grids": [],
    }

    for field_name in output_list:

        payload["grids"].append({
            "name": _vdb_grid_name(field_name),
            "shape": fields[field_name]["shape"],
            "fields": {
                field_name: {
                    "shape": fields[field_name]["shape"],
                    "dtype": str(fields[field_name]["array"].dtype),
                    "shm_name": fields[field_name]["shm_name"],
                }
            },
        })

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


def shutdown_output(shared_memory_blocks, writer_socket, writer_file):
    writer_file.close()
    writer_socket.close()

    for shm in shared_memory_blocks:
        shm.close()
        shm.unlink()