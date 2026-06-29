import json
import os
import socket
from multiprocessing import shared_memory
import numpy as np


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


def setup_output(simulations, outpath, shape):
    os.makedirs(outpath, exist_ok=True)

    output_cfg = simulations["outputs"][0]
    output_fields = output_cfg["fields"]
    output_list = _enabled_output_field_names(output_fields)

    writer_count = int(
        output_cfg.get("host_vdb_writer", {}).get(
            "process_count",
            ((output_cfg.get("performance") or {}).get("writer_processes", 1)),
        )
    )

    shared_memory_blocks = []
    writer_slots = []

    buffer_dtype = np.float32
    nbytes = int(np.prod(shape)) * np.dtype(buffer_dtype).itemsize

    for _slot_index in range(writer_count):
        fields = {}

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
                output_cfg["host_vdb_writer"]["host"],
                int(output_cfg["host_vdb_writer"]["port"]),
            )
        )
        writer_file = writer_socket.makefile("rwb")

        writer_slots.append({
            "fields": fields,
            "socket": writer_socket,
            "file": writer_file,
            "busy": False,
        })

    return shared_memory_blocks, writer_slots


def _get_writer_slot(writer_slots, output_index):
    slot_count = len(writer_slots)

    start = int(output_index) % slot_count

    for offset in range(slot_count):
        slot = writer_slots[(start + offset) % slot_count]
        if not slot["busy"]:
            return slot

    slot = writer_slots[start]
    _wait_for_writer_ack(slot["file"])
    slot["busy"] = False
    return slot


def enqueue_device_output(
    simulations,
    writer_slots,
    sim_fields,
    output_index,
    t,
):
    output_cfg = ((simulations.get("outputs") or [None])[0]) or {}
    frame_start = simulations.get("settings").get("start_frame")
    outpath = output_cfg.get("output_path")
    output_fields = output_cfg["fields"]
    output_list = _enabled_output_field_names(output_fields)

    slot = _get_writer_slot(writer_slots, output_index)
    fields = slot["fields"]

    for variable_name in output_list:
        source_field = sim_fields[variable_name]
        target_array = fields[variable_name]["array"]
        if hasattr(source_field, "copy_to_host"):
            source_field.copy_to_host(target_array)
        else:
            np.copyto(target_array, source_field)

    frame_idx = int(frame_start) + int(output_index)
    output_path = os.path.join(outpath, f"frame_{frame_idx:06d}.vdb")

    writer_payload = create_writer_payload(
        fields,
        output_list,
        output_path,
        t,
    )

    _send_payload_without_wait(slot["file"], writer_payload)
    slot["busy"] = True


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
        raise RuntimeError(f"Host VDB writer error: {response!r}")


def shutdown_output(shared_memory_blocks, writer_slots):
    for slot in writer_slots:
        if slot["busy"]:
            _wait_for_writer_ack(slot["file"])
            slot["busy"] = False

    for slot in writer_slots:
        slot["file"].close()
        slot["socket"].close()

    for shm in shared_memory_blocks:
        shm.close()
        shm.unlink()
