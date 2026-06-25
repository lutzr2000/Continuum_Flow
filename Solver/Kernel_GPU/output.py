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


def setup_output(
    simulations,
    outpath,
    shape
):
    os.makedirs(outpath, exist_ok=True)

    shared_memory_blocks = []
    fields = {}

    output_fields = simulations["outputs"][0]["fields"]
    output_list = _enabled_output_field_names(output_fields)

    buffer_dtype = np.float32
    nbytes = int(np.prod(shape)) * np.dtype(buffer_dtype).itemsize

    for variable_name in output_list:
        shm = shared_memory.SharedMemory(
            create=True,
            size=nbytes,
        )
        shared_memory_blocks.append(shm)

        fields[variable_name] = {
            "array": np.ndarray(shape, dtype=buffer_dtype, buffer=shm.buf),
            "shape": tuple(shape),
            "shm_name": shm.name,
        }

    writer_socket = socket.create_connection(
        (
            simulations["outputs"][0]["host_vdb_writer"]["host"],
            int(simulations["outputs"][0]["host_vdb_writer"]["port"]),
        )
    )
    writer_file = writer_socket.makefile("rwb")

    writer_busy = False

    return shared_memory_blocks, fields, writer_socket, writer_file, writer_busy


def enqueue_device_output(
    simulations,
    fields,
    sim_fields,
    output_index,
    t,
    writer_file,
    writer_busy,
):
    if writer_busy:
        _wait_for_writer_ack(writer_file)
        writer_busy = False

    output_cfg = ((simulations.get("outputs") or [None])[0]) or {}
    frame_start = simulations.get("settings").get("start_frame")
    outpath = output_cfg.get("output_path")
    output_fields = output_cfg["fields"]
    output_list = _enabled_output_field_names(output_fields)

    for variable_name in output_list:
        source_field = sim_fields[variable_name]
        source_field.copy_to_host(fields[variable_name]["array"])

    frame_idx = int(frame_start) + int(output_index)
    output_path = os.path.join(outpath, f"frame_{frame_idx:06d}.vdb")

    writer_payload = create_writer_payload(
        fields,
        output_list,
        output_path,
        t,
    )
    _send_payload_without_wait(writer_file, writer_payload)
    return True


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
        "grids": [],
    }

    for field_name in output_list:
        payload["grids"].append({
            "name": _vdb_grid_name(field_name),
            "shape": fields[field_name]["shape"],
            "fields": {
                field_name: {
                    "shape": fields[field_name]["shape"],
                    "shm_name": fields[field_name]["shm_name"],
                }   
            },
        })

    return payload


def _send_payload_without_wait(writer_file, writer_payload):
    writer_file.write((json.dumps(writer_payload) + "\n").encode("utf-8"))
    writer_file.flush()


def _wait_for_writer_ack(writer_file):
    response_line = writer_file.readline()
    if not response_line:
        raise RuntimeError("host VDB writer closed the connection")

    response = json.loads(response_line.decode("utf-8"))
    if response.get("status") != "ok":
        raise RuntimeError(response.get("message", "unknown host VDB writer error"))


def shutdown_output(shared_memory_blocks, writer_socket, writer_file, writer_busy):
    if writer_busy:
        _wait_for_writer_ack(writer_file)

    writer_file.close()
    writer_socket.close()

    for shm in shared_memory_blocks:
        shm.close()
        shm.unlink()