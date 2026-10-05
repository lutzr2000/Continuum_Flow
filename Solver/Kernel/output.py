from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any

import numpy as np
import pyopencl as cl

from Solver.Kernel import writer
import Solver.Kernel.kernel_config as kernel_config


# Change this value to benchmark a different number of concurrent writers.
WRITER_COUNT = 2
SNAPSHOT_SLOT_COUNT = WRITER_COUNT + 1

# All solver fields use the same slot/queue path. Until the wheel can write
# several grids into one file, only smoke is enabled for capture below.
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
CURRENT_WHEEL_FIELDS = frozenset({"smoke"})


def _pinned_array(
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


@dataclass
class SnapshotSlot:
    tile_map: np.ndarray | None = None
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    host_buffers: list[cl.Buffer] = field(default_factory=list)
    pool_capacity: int = 0
    frame: int = 0
    used_pool_slots: int = 0

    def ensure_capacity(
        self,
        host_queue: cl.CommandQueue,
        tile_shape: tuple[int, int, int],
        required_pool_slots: int,
        field_specs: tuple[tuple[str, str, Any], ...],
    ) -> None:
        required_pool_slots = max(1, int(required_pool_slots))
        if self.tile_map is None:
            self.tile_map, tile_buffer = _pinned_array(host_queue, tile_shape, np.int32)
            self.host_buffers.append(tile_buffer)

        if required_pool_slots <= self.pool_capacity:
            return

        self.fields.clear()
        self.host_buffers[1:] = []
        pool_shape = (
            required_pool_slots,
            kernel_config.TILE_SIZE,
            kernel_config.TILE_SIZE,
            kernel_config.TILE_SIZE,
        )
        for _config_name, field_name, dtype in field_specs:
            host_array, host_buffer = _pinned_array(host_queue, pool_shape, dtype)
            self.fields[field_name] = host_array
            self.host_buffers.append(host_buffer)
        self.pool_capacity = required_pool_slots


class OutputManager:
    def __init__(
        self,
        context: cl.Context,
        output_config: dict[str, Any],
        grid_shape: tuple[int, int, int],
        tile_shape: tuple[int, int, int],
        voxel_size: float,
        origin: tuple[float, float, float],
    ) -> None:
        configured_fields = output_config.get("fields") or {}
        self.field_specs = tuple(
            spec
            for spec in FIELDS
            if spec[0] in CURRENT_WHEEL_FIELDS
            and bool((configured_fields.get(spec[0]) or {}).get("enabled", False))
        )
        self.enabled = bool(self.field_specs)
        self.grid_shape = tuple(int(value) for value in grid_shape)
        self.tile_shape = tuple(int(value) for value in tile_shape)
        self.voxel_size = float(voxel_size)
        self.origin = tuple(float(value) for value in origin)
        self.output_path = Path(output_config.get("output_path") or ".")
        self.precision = str(output_config.get("precision", "float16"))
        self.host_queue = cl.CommandQueue(context) if self.enabled else None
        self.free_slots: Queue[SnapshotSlot] = Queue(SNAPSHOT_SLOT_COUNT)
        self.write_queue: Queue[SnapshotSlot | None] = Queue(SNAPSHOT_SLOT_COUNT)
        self.threads: list[Thread] = []
        self.errors: list[BaseException] = []
        self.error_lock = Lock()
        self.closed = False

        if not self.enabled:
            return

        for _ in range(SNAPSHOT_SLOT_COUNT):
            self.free_slots.put(SnapshotSlot())
        for index in range(WRITER_COUNT):
            thread = Thread(
                target=self._writer_loop,
                name=f"ContinuumVDBWriter-{index + 1}",
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)

    def _raise_writer_error(self) -> None:
        with self.error_lock:
            if self.errors:
                raise RuntimeError("VDB writer failed") from self.errors[0]

    def _writer_loop(self) -> None:
        while True:
            slot = self.write_queue.get()
            try:
                if slot is None:
                    return
                writer.write_snapshot(
                    output_path=self.output_path,
                    frame=slot.frame,
                    fields={
                        name: values[: slot.used_pool_slots]
                        for name, values in slot.fields.items()
                    },
                    index_tile_map=slot.tile_map,
                    grid_shape=self.grid_shape,
                    voxel_size=self.voxel_size,
                    origin=self.origin,
                    precision=self.precision,
                )
            except BaseException as exc:
                with self.error_lock:
                    self.errors.append(exc)
            finally:
                if slot is not None:
                    self.free_slots.put(slot)
                self.write_queue.task_done()

    def submit(
        self,
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
        if not self.enabled:
            return
        self._raise_writer_error()

        slot = self.free_slots.get()
        slot.ensure_capacity(
            self.host_queue,
            self.tile_shape,
            used_pool_slots,
            self.field_specs,
        )
        slot.frame = int(frame)
        slot.used_pool_slots = max(0, int(used_pool_slots))

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
                slot.tile_map,
                index_tile_map,
                is_blocking=False,
            )
        ]
        if slot.used_pool_slots:
            for _config_name, field_name, _dtype in self.field_specs:
                events.append(
                    cl.enqueue_copy(
                        queue,
                        slot.fields[field_name][: slot.used_pool_slots],
                        device_fields[field_name],
                        is_blocking=False,
                    )
                )
        cl.wait_for_events(events)
        self.write_queue.put(slot)

    def close(self) -> None:
        if self.closed or not self.enabled:
            return
        self.closed = True
        self.write_queue.join()
        for _ in self.threads:
            self.write_queue.put(None)
        for thread in self.threads:
            thread.join()
        self._raise_writer_error()
