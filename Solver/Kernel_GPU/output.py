import json
import os
import select
import socket
from multiprocessing import shared_memory

import numpy as np

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


def _tile_shape_for_dense_shape(shape):
    tile_size = int(kernel_config.TILE_SIZE)
    return tuple((int(axis_size) + tile_size - 1) // tile_size for axis_size in shape)


def _sparse_pool_shape_for_dense_shape(shape):
    tile_shape = _tile_shape_for_dense_shape(shape)
    tile_size = int(kernel_config.TILE_SIZE)
    return (
        int(np.prod(tile_shape)),
        tile_size,
        tile_size,
        tile_size,
    )


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

    tile_shape = _tile_shape_for_dense_shape(shape)
    sparse_pool_shape = _sparse_pool_shape_for_dense_shape(shape)

    tile_map_dtype = np.int32
    tile_map_nbytes = int(np.prod(tile_shape)) * np.dtype(tile_map_dtype).itemsize

    active_tile_meta_shape = (int(np.prod(tile_shape)), 7)
    active_tile_meta_dtype = np.int32
    active_tile_meta_nbytes = (
        int(np.prod(active_tile_meta_shape))
        * np.dtype(active_tile_meta_dtype).itemsize
    )

    sparse_pool_dtype = np.float32
    sparse_pool_nbytes = (
        int(np.prod(sparse_pool_shape)) * np.dtype(sparse_pool_dtype).itemsize
    )

    for _slot_index in range(writer_count):
        fields = {}

        tile_map_shm = shared_memory.SharedMemory(
            create=True,
            size=tile_map_nbytes,
        )
        shared_memory_blocks.append(tile_map_shm)
        tile_map_array = np.ndarray(
            tile_shape,
            dtype=tile_map_dtype,
            buffer=tile_map_shm.buf,
        )
        tile_map_array.fill(-1)

        active_tile_meta_shm = shared_memory.SharedMemory(
            create=True,
            size=active_tile_meta_nbytes,
        )
        shared_memory_blocks.append(active_tile_meta_shm)
        active_tile_meta_array = np.ndarray(
            active_tile_meta_shape,
            dtype=active_tile_meta_dtype,
            buffer=active_tile_meta_shm.buf,
        )
        active_tile_meta_array.fill(0)

        for variable_name in output_list:
            shm = shared_memory.SharedMemory(
                create=True,
                size=sparse_pool_nbytes,
            )
            shared_memory_blocks.append(shm)

            fields[variable_name] = {
                "array": np.ndarray(
                    sparse_pool_shape,
                    dtype=sparse_pool_dtype,
                    buffer=shm.buf,
                ),
                "dense_shape": tuple(shape),
                "pool_shape": sparse_pool_shape,
                "shm_name": shm.name,
            }

        writer_socket = socket.create_connection(
            (
                output_cfg["host_vdb_writer"]["host"],
                int(output_cfg["host_vdb_writer"]["port"]),
            )
        )
        writer_file = writer_socket.makefile("rwb")

        writer_slots.append(
            {
                "fields": fields,
                "tile_map": {
                    "array": tile_map_array,
                    "shape": tile_shape,
                    "shm_name": tile_map_shm.name,
                },
                "active_tiles": {
                    "array": active_tile_meta_array,
                    "shape": active_tile_meta_shape,
                    "shm_name": active_tile_meta_shm.name,
                },
                "socket": writer_socket,
                "file": writer_file,
                "busy": False,
            }
        )

    return shared_memory_blocks, writer_slots


def _get_writer_slot(writer_slots, output_index):
    _drain_ready_writer_acks(writer_slots)

    slot_count = len(writer_slots)

    start = int(output_index) % slot_count

    for offset in range(slot_count):
        slot = writer_slots[(start + offset) % slot_count]
        if not slot["busy"]:
            return slot

    print(
        f"All VDB writer slots busy. This slows down the simulation. Using more writers might improve performance"
    )
    slot = writer_slots[start]
    _wait_for_writer_ack(slot["file"])
    slot["busy"] = False
    return slot


def _drain_ready_writer_acks(writer_slots):
    busy_slots = [slot for slot in writer_slots if slot["busy"]]
    if not busy_slots:
        return

    ready_sockets, _, _ = select.select(
        [slot["socket"] for slot in busy_slots],
        [],
        [],
        0.0,
    )
    if not ready_sockets:
        return

    ready_socket_ids = {id(sock) for sock in ready_sockets}
    for slot in busy_slots:
        if id(slot["socket"]) not in ready_socket_ids:
            continue

        _wait_for_writer_ack(slot["file"])
        slot["busy"] = False


def _copy_sparse_field_to_shared_memory(field_data, shared_array, used_tile_count):
    if used_tile_count <= 0:
        return

    field_data[:used_tile_count].copy_to_host(shared_array[:used_tile_count])


def _build_active_tile_metadata(
    tile_map_host, active_tile_meta_array, dense_shape, tile_size
):
    active_coords = np.argwhere(tile_map_host >= 0)
    active_tile_count = int(active_coords.shape[0])
    if active_tile_count <= 0:
        return 0, 0

    dense_shape_array = np.asarray(dense_shape, dtype=np.int32)
    starts = active_coords.astype(np.int32, copy=False) * tile_size
    ends = np.minimum(starts + tile_size, dense_shape_array)
    sizes = ends - starts
    tile_indices = tile_map_host[
        active_coords[:, 0],
        active_coords[:, 1],
        active_coords[:, 2],
    ].astype(np.int32, copy=False)

    active_tile_meta_array[:active_tile_count, 0] = tile_indices
    active_tile_meta_array[:active_tile_count, 1:4] = starts
    active_tile_meta_array[:active_tile_count, 4:7] = sizes

    used_tile_count = int(tile_indices.max()) + 1
    return active_tile_count, used_tile_count


def enqueue_device_output(
    simulations,
    writer_slots,
    sim_fields,
    tile_map,
    tile_size,
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
    tile_map_slot = slot["tile_map"]
    active_tiles_slot = slot["active_tiles"]

    tile_map.copy_to_host(tile_map_slot["array"])
    tile_map_host = tile_map_slot["array"]

    active_tile_count, used_tile_count = _build_active_tile_metadata(
        tile_map_host,
        active_tiles_slot["array"],
        next(iter(fields.values()))["dense_shape"],
        int(tile_size),
    )

    for variable_name in output_list:
        _copy_sparse_field_to_shared_memory(
            sim_fields[variable_name],
            fields[variable_name]["array"],
            used_tile_count,
        )

    frame_idx = int(frame_start) + int(output_index)
    output_path = os.path.join(outpath, f"frame_{frame_idx:06d}.vdb")

    writer_payload = create_writer_payload(
        fields,
        tile_map_slot,
        active_tiles_slot,
        output_list,
        output_path,
        t,
        int(tile_size),
        active_tile_count,
        used_tile_count,
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
    tile_map,
    active_tiles,
    output_list,
    output_path,
    time_value,
    tile_size,
    active_tile_count,
    used_tile_count,
):
    payload = {
        "output_path": output_path,
        "time": float(time_value),
        "tile_map": {
            "shape": tile_map["shape"],
            "shm_name": tile_map["shm_name"],
        },
        "active_tiles": {
            "shape": active_tiles["shape"],
            "shm_name": active_tiles["shm_name"],
            "count": int(active_tile_count),
        },
        "grids": [],
    }

    for field_name in output_list:
        payload["grids"].append(
            {
                "name": _vdb_grid_name(field_name),
                "layout": "sparse_tiles",
                "dense_shape": fields[field_name]["dense_shape"],
                "tile_size": int(tile_size),
                "used_tile_count": int(used_tile_count),
                "fields": {
                    field_name: {
                        "shape": fields[field_name]["pool_shape"],
                        "shm_name": fields[field_name]["shm_name"],
                    }
                },
            }
        )

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
