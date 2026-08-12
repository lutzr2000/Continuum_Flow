import json
import os
import socket
from multiprocessing import shared_memory
import numpy as np


def _reconstruct_sparse_field_to_dense_host(field_info, output_array):
    tile_map_device = field_info["tile_map"]
    sparse_field_device = field_info["data"]
    tile_size = int(field_info["tile_size"])

    output_array.fill(0.0)

    tile_map_host = tile_map_device.copy_to_host()
    sparse_field_host = sparse_field_device.copy_to_host()

    nx, ny, nz = output_array.shape
    tiles_x, tiles_y, tiles_z = tile_map_host.shape

    for tile_i in range(tiles_x):
        cell_i_start = tile_i * tile_size
        if cell_i_start >= nx:
            break

        cell_i_end = min(cell_i_start + tile_size, nx)

        for tile_j in range(tiles_y):
            cell_j_start = tile_j * tile_size
            if cell_j_start >= ny:
                break

            cell_j_end = min(cell_j_start + tile_size, ny)

            for tile_k in range(tiles_z):
                tile_index = int(tile_map_host[tile_i, tile_j, tile_k])
                if tile_index == -1:
                    continue

                cell_k_start = tile_k * tile_size
                if cell_k_start >= nz:
                    break

                cell_k_end = min(cell_k_start + tile_size, nz)

                output_array[
                    cell_i_start:cell_i_end,
                    cell_j_start:cell_j_end,
                    cell_k_start:cell_k_end,
                ] = sparse_field_host[
                    tile_index,
                    : cell_i_end - cell_i_start,
                    : cell_j_end - cell_j_start,
                    : cell_k_end - cell_k_start,
                ]


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

    # Round-robin bevorzugen
    start = int(output_index) % slot_count

    for offset in range(slot_count):
        slot = writer_slots[(start + offset) % slot_count]
        if not slot["busy"]:
            return slot

    # Alle busy: auf den Round-robin-Slot warten
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
        if isinstance(source_field, dict):
            _reconstruct_sparse_field_to_dense_host(
                source_field,
                fields[variable_name]["array"],
            )
        else:
            source_field.copy_to_host(fields[variable_name]["array"])

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
