from typing import Any

import json
import os
import select
import socket
import numba
from multiprocessing import shared_memory

import numpy as np

import Solver.Kernel_GPU.kernel_config as kernel_config


# ------------setup------------------
def setup_output(simulations: dict[str, Any], shape: tuple[int, int, int], tile_shape: tuple[int, int, int]) -> Any:
    """
    Setup output.
    """
    output_cfg = simulations["outputs"][0]
    output_fields = output_cfg["fields"]
    output_list = get_enabled_output_names(output_fields)

    writer_count = int(
        output_cfg.get("host_vdb_writer", {}).get(
            "process_count",
            ((output_cfg.get("performance") or {}).get("writer_processes", 1)),
        )
    )
    max_writer_count = int(
        output_cfg.get("host_vdb_writer", {}).get("max_process_count", writer_count)
    )

    shared_memory_blocks = []

    writer_context = {
        "output_cfg": output_cfg,
        "output_list": output_list,
        "shape": tuple(shape),
        "tile_shape": tuple(tile_shape),
        "shared_memory_blocks": shared_memory_blocks,
    }

    writer_state = {
        "slots": [],
        "max_count": max(writer_count, max_writer_count),
        "context": writer_context,
    }

    for _ in range(writer_count):
        grow_writer_slots(writer_state)

    return shared_memory_blocks, writer_state


def grow_writer_slots(writer_state: Any, prewarm: bool=False) -> Any:
    """
    Grow writer slots.
    """
    writer_slots = writer_state["slots"]

    if len(writer_slots) >= writer_state["max_count"]:
        return None

    slot = create_writer_slot(
        writer_state["context"],
        prewarm=prewarm,
    )
    writer_slots.append(slot)
    return slot


def create_writer_slot(writer_context: Any, prewarm: bool=False) -> Any:
    """
    Create writer slot.
    """
    output_cfg = writer_context["output_cfg"]
    output_list = writer_context["output_list"]
    shape = writer_context["shape"]

    tile_shape = writer_context["tile_shape"]
    shared_memory_blocks = writer_context["shared_memory_blocks"]

    active_tile_count_max = int(np.prod(tile_shape))
    tile_map_nbytes = int(active_tile_count_max * np.dtype(np.int32).itemsize)
    active_tile_meta_shape = (active_tile_count_max, 4)
    active_tile_meta_nbytes = int(
        np.prod(active_tile_meta_shape) * np.dtype(np.int32).itemsize
    )

    sparse_pool_shape = (
        int(np.prod(tile_shape)),
        kernel_config.TILE_SIZE,
        kernel_config.TILE_SIZE,
        kernel_config.TILE_SIZE,
    )

    sparse_pool_nbytes = int(
        int(np.prod(sparse_pool_shape))
        * np.dtype(kernel_config.GPU_FIELD_DTYPE).itemsize
    )

    fields = {}

    tile_map_shm = shared_memory.SharedMemory(
        create=True,
        size=tile_map_nbytes,
    )
    shared_memory_blocks.append(tile_map_shm)
    tile_map_array = np.ndarray(
        tile_shape,
        dtype=np.int32,
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
        dtype=np.int32,
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
                dtype=kernel_config.GPU_FIELD_DTYPE,
                buffer=shm.buf,
            ),
            "dense_shape": shape,
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
    if prewarm:
        writer_file.write(b"__WARMUP__\n")
        writer_file.flush()
        writer_file.readline()

    return {
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


def get_enabled_output_names(output_fields: dict[str, Any]) -> Any:
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


# ------------enqueue------------------
def enqueue_device_output(
    simulations: dict[str, Any],
    writer_state: Any,
    sim_fields: Any,
    tile_map: Any,
    tile_size: Any,
    active_tile_count: int,
    used_tile_count: int,
    output_index: int,
    t: float,
) -> None:
    """
    Enqueue device output.
    """
    output_cfg = ((simulations.get("outputs") or [None])[0]) or {}
    frame_start = simulations.get("settings").get("start_frame")
    outpath = output_cfg.get("output_path")
    output_fields = output_cfg["fields"]
    output_list = get_enabled_output_names(output_fields)

    slot = get_writer_slot(writer_state, output_index)
    fields = slot["fields"]
    tile_map_slot = slot["tile_map"]
    active_tiles_slot = slot["active_tiles"]

    tile_map.copy_to_host(tile_map_slot["array"])
    tile_map_host = tile_map_slot["array"]
    max_slot_index = int(tile_map_host.max())
    used_tile_count = 0 if max_slot_index < 0 else max_slot_index + 1

    write_metadata(
        tile_map_host,
        active_tiles_slot["array"],
        int(tile_size),
    )

    if used_tile_count > 0:
        for variable_name in output_list:
            sim_fields[variable_name][:used_tile_count].copy_to_host(
                fields[variable_name]["array"][:used_tile_count]
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

    slot["file"].write((json.dumps(writer_payload) + "\n").encode("utf-8"))
    slot["file"].flush()
    slot["busy"] = True


def get_writer_slot(writer_state: Any, output_index: int) -> Any:
    """
    Get writer slot.
    """
    writer_slots = writer_state["slots"]

    busy_slots = [slot for slot in writer_slots if slot["busy"]]

    if busy_slots:
        ready_sockets, _, _ = select.select(
            [slot["socket"] for slot in busy_slots],
            [],
            [],
            0.0,
        )

        if ready_sockets:
            ready_socket_ids = {id(sock) for sock in ready_sockets}

            for slot in busy_slots:
                if id(slot["socket"]) in ready_socket_ids:
                    slot["file"].readline()
                    slot["busy"] = False

    free_slot_count = sum(not slot["busy"] for slot in writer_slots)

    if free_slot_count <= 1:
        grow_writer_slots(writer_state, prewarm=True)

    slot_count = len(writer_slots)
    start = output_index % slot_count

    for offset in range(slot_count):
        slot = writer_slots[(start + offset) % slot_count]

        if not slot["busy"]:
            return slot

    new_slot = grow_writer_slots(writer_state, prewarm=True)

    if new_slot is not None:
        return new_slot

    slot = writer_slots[start]
    slot["file"].readline()
    slot["busy"] = False

    return slot


@numba.njit(cache=True, nogil=True)
def write_metadata(
    tile_map_host: Any,
    active_tile_meta_array: Any,
    tile_size: Any,
) -> None:
    """
    Write metadata.
    """
    dim_x = tile_map_host.shape[0]
    dim_y = tile_map_host.shape[1]
    dim_z = tile_map_host.shape[2]

    active_idx = 0

    for x in range(dim_x):
        start_x = x * tile_size

        for y in range(dim_y):
            start_y = y * tile_size

            for z in range(dim_z):
                tile_idx = tile_map_host[x, y, z]

                if tile_idx >= 0:
                    active_tile_meta_array[active_idx, 0] = tile_idx
                    active_tile_meta_array[active_idx, 1] = start_x
                    active_tile_meta_array[active_idx, 2] = start_y
                    active_tile_meta_array[active_idx, 3] = z * tile_size

                    active_idx += 1


def create_writer_payload(
    fields: Any,
    tile_map: Any,
    active_tiles: Any,
    output_list: Any,
    output_path: Any,
    time_value: float,
    tile_size: Any,
    active_tile_count: int,
    used_tile_count: int,
) -> Any:
    """
    Create writer payload.
    """
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
                "name": "density" if field_name == "smoke" else field_name,
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


def shutdown_output(shared_memory_blocks: Any, writer_state: Any) -> None:
    """
    Shutdown output.
    """
    writer_slots = writer_state["slots"]

    for slot in writer_slots:
        if slot["busy"]:
            slot["file"].readline()
            slot["busy"] = False

    for slot in writer_slots:
        slot["file"].close()
        slot["socket"].close()

    for shm in shared_memory_blocks:
        shm.close()
        shm.unlink()
