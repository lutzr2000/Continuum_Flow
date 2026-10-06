from __future__ import annotations

from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any

import numpy as np
import pyopencl as cl

from Solver.Kernel import writer
import Solver.Kernel.kernel_config as kernel_config

WRITER_COUNT = 2
PINNED_MEMORY_SLOTS = WRITER_COUNT + 1


def pinned_array(
    queue: cl.CommandQueue,
    shape: tuple[int, ...],
    dtype: Any,
) -> tuple[np.ndarray, cl.Buffer]:
    dtype = np.dtype(dtype)
    byte_count = max(1, int(np.prod(shape)) * dtype.itemsize)
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


def ensure_capacity(
    slot: dict[str, Any],
    host_queue: cl.CommandQueue,
    tile_shape: tuple[int, int, int],
    required_pool_slots: int,
    field_specs: tuple[tuple[str, str, Any], ...],
) -> None:
    required_pool_slots = max(1, int(required_pool_slots))

    if slot["tile_map"] is None:
        slot["tile_map"], tile_buffer = pinned_array(
            host_queue,
            tile_shape,
            np.int32,
        )
        slot["host_buffers"].append(tile_buffer)

    if required_pool_slots <= slot["pool_capacity"]:
        return

    slot["fields"].clear()
    slot["host_buffers"][1:] = []

    pool_shape = (
        required_pool_slots,
        kernel_config.TILE_SIZE,
        kernel_config.TILE_SIZE,
        kernel_config.TILE_SIZE,
    )

    for _, field_name, dtype in field_specs:
        host_array, host_buffer = pinned_array(
            host_queue,
            pool_shape,
            dtype,
        )
        slot["fields"][field_name] = host_array
        slot["host_buffers"].append(host_buffer)

    slot["pool_capacity"] = required_pool_slots


def _writer_loop(state: dict[str, Any]) -> None:
    while True:
        slot = state["write_queue"].get()

        try:
            if slot is None:
                return

            writer.write_snapshot(
                output_path=state["output_path"],
                frame=slot["frame"],
                fields={
                    name: values[: slot["used_pool_slots"]]
                    for name, values in slot["fields"].items()
                },
                index_tile_map=slot["tile_map"],
                grid_shape=state["grid_shape"],
                voxel_size=state["voxel_size"],
                origin=state["origin"],
                compression=state["compression"],
                precision=state["precision"],
                field_names=state["field_names"],
            )
        except BaseException as exc:
            with state["error_lock"]:
                state["errors"].append(exc)
        finally:
            if slot is not None:
                state["free_slots"].put(slot)
            state["write_queue"].task_done()


def create_output(
    context: cl.Context,
    output_config: dict[str, Any],
    grid_shape: tuple[int, int, int],
    tile_shape: tuple[int, int, int],
    voxel_size: float,
    origin: tuple[float, float, float],
) -> dict[str, Any]:
    configured_fields = output_config.get("fields") or {}

    field_specs = tuple(
        spec
        for spec in (
            ("velocity", "u", kernel_config.FIELD_DTYPE),
            ("velocity", "v", kernel_config.FIELD_DTYPE),
            ("velocity", "w", kernel_config.FIELD_DTYPE),
            ("pressure", "pressure", kernel_config.FIELD_DTYPE),
            ("temperature", "temperature", kernel_config.FIELD_DTYPE),
            ("smoke", "smoke", kernel_config.FIELD_DTYPE),
            ("flame", "flame", kernel_config.FIELD_DTYPE),
        )
        if bool((configured_fields.get(spec[0]) or {}).get("enabled", False))
    )

    enabled = bool(field_specs)

    state = {
        "enabled": enabled,
        "field_specs": field_specs,
        "grid_shape": tuple(int(value) for value in grid_shape),
        "tile_shape": tuple(int(value) for value in tile_shape),
        "voxel_size": float(voxel_size),
        "origin": tuple(float(value) for value in origin),
        "output_path": Path(output_config.get("output_path") or "."),
        "compression": str(output_config.get("compression", "none")),
        "precision": str(output_config.get("precision", "float16")),
        "field_names": {
            name: str((configured_fields.get(name) or {}).get("name") or default)
            for name, default in (
                ("velocity", "velocity"),
                ("pressure", "pressure"),
                ("temperature", "temperature"),
                ("smoke", "density"),
                ("fuel", "fuel"),
                ("flame", "flame"),
            )
        },
        "host_queue": cl.CommandQueue(context) if enabled else None,
        "free_slots": Queue(PINNED_MEMORY_SLOTS),
        "write_queue": Queue(PINNED_MEMORY_SLOTS),
        "threads": [],
        "errors": [],
        "error_lock": Lock(),
        "closed": False,
    }

    if not enabled:
        return state

    for _ in range(PINNED_MEMORY_SLOTS):
        state["free_slots"].put(
            {
                "tile_map": None,
                "fields": {},
                "host_buffers": [],
                "pool_capacity": 0,
                "frame": 0,
                "used_pool_slots": 0,
            }
        )

    for index in range(WRITER_COUNT):
        thread = Thread(
            target=_writer_loop,
            args=(state,),
            name=f"ContinuumVDBWriter-{index + 1}",
            daemon=True,
        )
        thread.start()
        state["threads"].append(thread)

    return state


def submit_output(
    state: dict[str, Any],
    *,
    queue: cl.CommandQueue,
    frame: int,
    used_pool_slots: int,
    index_tile_map: cl.Buffer,
    u: cl.Buffer,
    v: cl.Buffer,
    w: cl.Buffer,
    pressure: cl.Buffer,
    temperature: cl.Buffer,
    smoke: cl.Buffer,
    fuel: cl.Buffer,
    flame: cl.Buffer,
) -> None:
    if not state["enabled"]:
        return

    slot = state["free_slots"].get()

    ensure_capacity(
        slot,
        state["host_queue"],
        state["tile_shape"],
        used_pool_slots,
        state["field_specs"],
    )

    slot["frame"] = int(frame)
    slot["used_pool_slots"] = max(0, int(used_pool_slots))

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

    events = [
        cl.enqueue_copy(
            queue,
            slot["tile_map"],
            index_tile_map,
            is_blocking=False,
        )
    ]

    if slot["used_pool_slots"]:
        for _config_name, field_name, _dtype in state["field_specs"]:
            events.append(
                cl.enqueue_copy(
                    queue,
                    slot["fields"][field_name][: slot["used_pool_slots"]],
                    device_fields[field_name],
                    is_blocking=False,
                )
            )

    cl.wait_for_events(events)
    state["write_queue"].put(slot)


def close_output(state: dict[str, Any]) -> None:
    if state["closed"] or not state["enabled"]:
        return

    state["closed"] = True
    state["write_queue"].join()

    for _ in state["threads"]:
        state["write_queue"].put(None)

    for thread in state["threads"]:
        thread.join()
