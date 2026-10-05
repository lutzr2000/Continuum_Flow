from typing import Any

import numpy as np
import pyopencl as cl

import Solver.Kernel.kernel_config as kernel_config


FIELDS = (
    ("velocity", "u", kernel_config.FIELD_DTYPE),
    ("velocity", "v", kernel_config.FIELD_DTYPE),
    ("velocity", "w", kernel_config.FIELD_DTYPE),
    ("pressure", "pressure", kernel_config.FIELD_DTYPE),
    ("temperature", "temperature", kernel_config.FIELD_DTYPE),
    ("smoke", "smoke", kernel_config.FIELD_DTYPE),
    ("fuel", "fuel", np.uint8),
    ("flame", "flame", kernel_config.FIELD_DTYPE),
)


def pinned_array(
    queue: cl.CommandQueue,
    shape: tuple[int, ...],
    dtype: Any,
) -> tuple[np.ndarray, cl.Buffer | None]:
    dtype = np.dtype(dtype)
    byte_count = int(np.prod(shape)) * dtype.itemsize

    if byte_count == 0:
        return np.empty(shape, dtype=dtype), None

    host_buffer = cl.Buffer(
        queue.context,
        cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR,
        size=byte_count,
    )
    host_array, _ = cl.enqueue_map_buffer(
        queue,
        host_buffer,
        cl.map_flags.READ | cl.map_flags.WRITE,
        0,
        shape,
        dtype,
        is_blocking=True,
    )

    return host_array, host_buffer


def output_to_memory(
    queue: cl.CommandQueue,
    host_queue: cl.CommandQueue,
    output_config: dict[str, Any],
    index_tile_map: cl.Buffer,
    tile_shape: tuple[int, int, int],
    used_pool_slots: int,
    *,
    u: cl.Buffer,
    v: cl.Buffer,
    w: cl.Buffer,
    pressure: cl.Buffer,
    temperature: cl.Buffer,
    smoke: cl.Buffer,
    fuel: cl.Buffer,
    flame: cl.Buffer,
) -> dict[str, Any]:
    """Copy the configured sparse field pools from the device to CPU arrays."""
    configured_fields = output_config.get("fields") or {}

    pool_shape = (
        max(0, int(used_pool_slots)),
        kernel_config.TILE_SIZE,
        kernel_config.TILE_SIZE,
        kernel_config.TILE_SIZE,
    )

    device_fields = {
        "u": u,
        "v": v,
        "w": w,
        "pressure": pressure,
        "temperature": temperature,
        "smoke": smoke,
        "fuel": fuel,
        "flame": flame,
    }

    arrays: dict[str, np.ndarray] = {}
    events: list[cl.Event] = []
    host_buffers: list[cl.Buffer] = []

    tile_map, tile_map_buffer = pinned_array(host_queue, tile_shape, np.int32)
    if tile_map_buffer is not None:
        host_buffers.append(tile_map_buffer)
    events.append(cl.enqueue_copy(queue, tile_map, index_tile_map, is_blocking=False))

    for config_name, array_name, dtype in FIELDS:
        if not (configured_fields.get(config_name) or {}).get("enabled", False):
            continue

        host_array, host_buffer = pinned_array(host_queue, pool_shape, dtype)
        arrays[array_name] = host_array
        if host_buffer is not None:
            host_buffers.append(host_buffer)

        if host_array.size:
            events.append(
                cl.enqueue_copy(
                    queue,
                    host_array,
                    device_fields[array_name],
                    is_blocking=False,
                )
            )

    cl.wait_for_events(events)

    copied_bytes = tile_map.nbytes + sum(array.nbytes for array in arrays.values())

    return {
        "fields": arrays,
        "index_tile_map": tile_map,
        "pool_shape": pool_shape,
        "copied_bytes": copied_bytes,
        "_host_buffers": host_buffers,
    }
