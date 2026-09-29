import pyopencl as cl
import socket
import select
import numpy as np
import json
import os

from typing import Any
from multiprocessing import shared_memory

import Solver.Kernel.kernel_config as kernel_config
import Solver.Kernel.helper as helper


# ------------setup------------------
def setup_output(
    context: cl.Context,
    simulation: Any,
    shape: tuple[int, int, int],
    tile_shape: tuple[int, int, int],
    sparse_tile_capacity: int,
) -> Any:
    """
    Initialize asynchronous frame-output state and its first writer slots.

    Enabled fields, grid dimensions, writer limits, and owned shared-memory
    blocks are collected into the context returned to the simulation loop.
    """
    output_cfg = simulation["outputs"][0]
    output_fields = output_cfg["fields"]
    output_list = get_enabled_output_names(output_fields)

    writer_count = int(output_cfg["host_vdb_writer"]["process_count"])

    shared_memory_blocks = []

    writer_context = {
        "context": context,
        "output_cfg": output_cfg,
        "output_list": output_list,
        "shape": tuple(shape),
        "tile_shape": tuple(tile_shape),
        "initial_tile_capacity": max(
            1,
            int(sparse_tile_capacity),
        ),
        "shared_memory_blocks": shared_memory_blocks,
    }

    writer_state = {
        "slots": [],
        "context": writer_context,
    }

    for _ in range(writer_count):
        writer_state["slots"].append(create_writer_slot(writer_context))

    return shared_memory_blocks, writer_state


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


def create_writer_slot(writer_context: Any) -> Any:
    """
    Create shared-memory buffers and a host-writer connection for one slot.

    The slot owns a tile map, active-tile metadata, one sparse pool per enabled
    field, and a socket stream used to submit work and receive completion
    acknowledgements.
    """
    context = writer_context["context"]
    output_cfg = writer_context["output_cfg"]
    output_list = writer_context["output_list"]
    shape = writer_context["shape"]

    tile_shape = writer_context["tile_shape"]
    initial_tile_capacity = writer_context["initial_tile_capacity"]
    shared_memory_blocks = writer_context["shared_memory_blocks"]

    active_tile_count_max = int(np.prod(tile_shape))

    tile_map_nbytes = int(active_tile_count_max * np.dtype(np.int32).itemsize)

    active_tile_meta_shape = (
        active_tile_count_max,
        4,
    )

    active_tile_meta_nbytes = int(
        np.prod(active_tile_meta_shape) * np.dtype(np.int32).itemsize
    )

    sparse_pool_shape = (
        initial_tile_capacity,
        kernel_config.TILE_SIZE,
        kernel_config.TILE_SIZE,
        kernel_config.TILE_SIZE,
    )

    sparse_pool_nbytes = int(
        np.prod(sparse_pool_shape) * np.dtype(kernel_config.FIELD_DTYPE).itemsize
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

    active_tile_meta_device = helper.device_array(
        context,
        active_tile_meta_shape,
        dtype=np.int32,
    )

    active_tile_count_device = helper.device_array(
        context,
        1,
        dtype=np.int32,
    )

    for variable_name in output_list:
        shm = shared_memory.SharedMemory(
            create=True,
            size=sparse_pool_nbytes,
        )

        shared_memory_blocks.append(shm)

        fields[variable_name] = {
            "array": np.ndarray(
                sparse_pool_shape,
                dtype=kernel_config.FIELD_DTYPE,
                buffer=shm.buf,
            ),
            "dense_shape": shape,
            "pool_shape": sparse_pool_shape,
            "shm_name": shm.name,
            "shm": shm,
        }

    writer_socket = socket.create_connection(
        (
            output_cfg["host_vdb_writer"]["host"],
            int(output_cfg["host_vdb_writer"]["port"]),
        )
    )

    writer_file = writer_socket.makefile("rwb")

    return {
        "fields": fields,
        "tile_map": {
            "array": tile_map_array,
            "shape": tile_shape,
            "shm_name": tile_map_shm.name,
        },
        "active_tiles": {
            "array": active_tile_meta_array,
            "device_array": active_tile_meta_device,
            "device_count": active_tile_count_device,
            "shape": active_tile_meta_shape,
            "shm_name": active_tile_meta_shm.name,
        },
        "socket": writer_socket,
        "file": writer_file,
        "busy": False,
    }


# ------------enqueue------------------
def enqueue_device_output(
    queue: cl.CommandQueue,
    output_kernels: dict[str, cl.Kernel],
    simulations: dict[str, Any],
    writer_state: Any,
    sim_fields: Any,
    tile_map: Any,
    tile_shape: tuple[int, int, int],
    tile_size: Any,
    active_tile_count: int,
    used_tile_count: int,
    output_index: int,
    t: float,
) -> None:
    """
    Copy one sparse GPU frame into shared memory and enqueue it for VDB output.

    A free writer slot is acquired, tile metadata and enabled field pools are
    transferred to its shared buffers, and a JSON payload referencing those
    buffers is sent to the host writer without waiting for file completion.
    """
    output_cfg = ((simulations.get("outputs") or [None])[0]) or {}
    frame_start = simulations.get("settings").get("start_frame")
    outpath = output_cfg.get("output_path")
    output_fields = output_cfg["fields"]
    output_list = get_enabled_output_names(output_fields)

    slot = get_writer_slot(writer_state, output_index)
    tile_map_slot = slot["tile_map"]
    active_tiles_slot = slot["active_tiles"]

    cl.enqueue_copy(
        queue,
        tile_map_slot["array"],
        tile_map,
    ).wait()

    tile_map_host = tile_map_slot["array"]
    max_slot_index = int(tile_map_host.max())
    used_tile_count = 0 if max_slot_index < 0 else max_slot_index + 1

    available_tile_count = int(
        next(iter(sim_fields.values())).size
        // (
            kernel_config.TILE_SIZE
            * kernel_config.TILE_SIZE
            * kernel_config.TILE_SIZE
            * np.dtype(kernel_config.FIELD_DTYPE).itemsize
        )
    )

    ensure_writer_slot_capacity(
        slot,
        writer_state["context"],
        available_tile_count,
    )

    fields = slot["fields"]

    helper.fill_device(
        queue,
        active_tiles_slot["device_count"],
        0,
        dtype=np.int32,
    )

    output_kernels["write_metadata"](
        queue,
        tile_shape,
        None,
        tile_map,
        active_tiles_slot["device_array"],
        active_tiles_slot["device_count"],
        np.int32(tile_size),
        np.int32(tile_shape[0]),
        np.int32(tile_shape[1]),
        np.int32(tile_shape[2]),
    )

    if active_tile_count > 0:
        cl.enqueue_copy(
            queue,
            active_tiles_slot["array"][:active_tile_count],
            active_tiles_slot["device_array"],
        ).wait()

    if used_tile_count > 0:
        for variable_name in output_list:
            cl.enqueue_copy(
                queue,
                fields[variable_name]["array"][:used_tile_count],
                sim_fields[variable_name],
            ).wait()

    frame_idx = int(frame_start) + int(output_index)
    output_path = os.path.join(
        outpath,
        f"frame_{frame_idx:06d}.vdb",
    )

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


def ensure_writer_slot_capacity(
    slot: Any,
    writer_context: Any,
    required_tile_count: int,
) -> None:
    """Grow an idle writer slot's field buffers to match the GPU sparse pool."""
    fields = slot["fields"]
    current_capacity = (
        min(field["pool_shape"][0] for field in fields.values()) if fields else 0
    )
    if required_tile_count <= current_capacity:
        return

    shared_memory_blocks = writer_context["shared_memory_blocks"]
    tile_size = kernel_config.TILE_SIZE
    pool_shape = (required_tile_count, tile_size, tile_size, tile_size)
    pool_nbytes = int(
        np.prod(pool_shape) * np.dtype(kernel_config.FIELD_DTYPE).itemsize
    )

    for field_name in writer_context["output_list"]:
        old_field = fields[field_name]
        if old_field["pool_shape"][0] >= required_tile_count:
            continue

        shm = shared_memory.SharedMemory(create=True, size=pool_nbytes)
        shared_memory_blocks.append(shm)
        replacement_field = {
            "array": np.ndarray(
                pool_shape,
                dtype=kernel_config.GPU_FIELD_DTYPE,
                buffer=shm.buf,
            ),
            "dense_shape": old_field["dense_shape"],
            "pool_shape": pool_shape,
            "shm_name": shm.name,
            "shm": shm,
        }
        old_shm = old_field["shm"]
        fields[field_name] = replacement_field
        del old_field["array"]
        old_shm.close()
        old_shm.unlink()
        shared_memory_blocks.remove(old_shm)


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
    Build the JSON-serializable description consumed by the host VDB writer.

    Large arrays remain in shared memory; the payload contains their names,
    shapes, sparse layout metadata, output path, and simulation time.
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


def get_writer_slot(writer_state: Any, output_index: int) -> Any:
    """
    Acquire a free output slot, waiting only when all fixed slots are busy.

    Completed sockets are polled first; the selected busy slot is awaited
    before reuse when no slot is immediately available.
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

    slot_count = len(writer_slots)
    start = output_index % slot_count

    for offset in range(slot_count):
        slot = writer_slots[(start + offset) % slot_count]

        if not slot["busy"]:
            return slot

    slot = writer_slots[start]
    slot["file"].readline()
    slot["busy"] = False

    return slot


# ------------shutdown------------------
def shutdown_output(shared_memory_blocks: Any, writer_state: Any) -> None:
    """
    Finish pending writes and release all writer and shared-memory resources.

    Busy slots are acknowledged before sockets are closed. Every shared-memory
    block owned by the output subsystem is then closed and unlinked.
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
