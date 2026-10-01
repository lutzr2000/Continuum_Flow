"""Thread-safe ownership and transfer for the live OpenCL preview snapshot."""

from __future__ import annotations

import math
import threading
from pathlib import Path

import numpy as np
import pyopencl as cl


class PreviewMetadata:
    def __init__(
        self,
        frame_index: int,
        active_tile_count: int,
        tile_shape: tuple[int, int, int],
        atlas_tile_shape: tuple[int, int, int],
        tile_size: int,
    ) -> None:
        self.frame_index = frame_index
        self.active_tile_count = active_tile_count
        self.tile_shape = tile_shape
        self.atlas_tile_shape = atlas_tile_shape
        self.tile_size = tile_size


class PreviewTransfer:
    """An asynchronous OpenCL-to-host transfer owned by the renderer."""

    def __init__(
        self,
        metadata: PreviewMetadata,
        tile_lookup: np.ndarray,
        fields: np.ndarray,
        event: cl.Event,
    ) -> None:
        self.metadata = metadata
        self.tile_lookup = tile_lookup
        self.fields = fields
        self.event = event

    @property
    def complete(self) -> bool:
        return (
            self.event.command_execution_status == cl.command_execution_status.COMPLETE
        )


class PreviewExchange:
    """Own one sparse GPU snapshot shared by the solver and renderer.

    The OpenCL-to-host methods form the only backend-specific transfer boundary.
    A future zero-copy implementation can replace ``begin_transfer`` without
    changing publication, ownership, or renderer frame selection.
    """

    FREE = "free"
    PUBLISHING = "publishing"
    READY = "ready"
    ACQUIRED = "acquired"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._enabled = False
        self._state = self.FREE
        self._context: cl.Context | None = None
        self._transfer_queue: cl.CommandQueue | None = None
        self._kernel: cl.Kernel | None = None
        self._tile_lookup: cl.Buffer | None = None
        self._fields: cl.Buffer | None = None
        self._counter: cl.Buffer | None = None
        self._capacity = -1
        self._metadata: PreviewMetadata | None = None
        self._publish_event: cl.Event | None = None

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)
            if not self._enabled and self._state == self.READY:
                self._state = self.FREE
                self._metadata = None
                self._publish_event = None

    def _refresh_publication_locked(self) -> None:
        """Promote a completed publication without blocking either thread."""
        if self._state != self.PUBLISHING or self._publish_event is None:
            return

        status = self._publish_event.command_execution_status

        if status == cl.command_execution_status.COMPLETE:
            if self._enabled:
                self._state = self.READY
            else:
                self._state = self.FREE
                self._metadata = None
                self._publish_event = None

    def close(self) -> None:
        with self._lock:
            self._enabled = False
            self._state = self.FREE
            self._context = None
            self._transfer_queue = None
            self._kernel = None
            self._tile_lookup = None
            self._fields = None
            self._counter = None
            self._metadata = None
            self._publish_event = None
            self._capacity = -1

    @property
    def allocated_bytes(self) -> int:
        """Return the bytes owned by the dedicated OpenCL snapshot."""
        with self._lock:
            return sum(
                buffer.size
                for buffer in (
                    self._tile_lookup,
                    self._fields,
                    self._counter,
                )
                if buffer is not None
            )

    @staticmethod
    def _atlas_shape(active_count: int) -> tuple[int, int, int]:
        if active_count <= 0:
            return (1, 1, 1)

        side = max(1, math.ceil(active_count ** (1.0 / 3.0)))
        depth = math.ceil(active_count / (side * side))

        return (side, side, depth)

    def _ensure_opencl(
        self,
        context: cl.Context,
        active_count: int,
        tile_size: int,
    ) -> None:
        if self._context is None or int(self._context.int_ptr) != int(context.int_ptr):
            source_path = (
                Path(__file__).resolve().parents[1] / "Kernel" / "OpenCL" / "preview.cl"
            )

            self._context = context
            self._transfer_queue = cl.CommandQueue(context)

            program = cl.Program(
                context,
                source_path.read_text(encoding="utf-8"),
            ).build()

            self._kernel = cl.Kernel(program, "pack_preview")
            self._capacity = -1

        if self._capacity == active_count:
            return

        atlas_shape = self._atlas_shape(active_count)
        atlas_cell_count = math.prod(atlas_shape) * tile_size**3

        self._fields = cl.Buffer(
            context,
            cl.mem_flags.READ_WRITE,
            max(1, atlas_cell_count * 2 * 4),
        )

        self._counter = cl.Buffer(
            context,
            cl.mem_flags.READ_WRITE,
            4,
        )

        self._capacity = active_count

    def try_publish(
        self,
        *,
        queue: cl.CommandQueue,
        frame_index: int,
        tile_map: cl.Buffer,
        smoke: cl.Buffer,
        flame: cl.Buffer,
        active_tile_count: int,
        tile_shape: tuple[int, int, int],
        tile_size: int,
    ) -> bool:
        """Queue a snapshot without ever waiting for renderer ownership."""

        with self._lock:
            self._refresh_publication_locked()

            if not self._enabled or self._state != self.FREE:
                return False

            self._state = self.PUBLISHING

        try:
            self._ensure_opencl(
                queue.context,
                active_tile_count,
                tile_size,
            )

            atlas_shape = self._atlas_shape(active_tile_count)
            tile_count = math.prod(tile_shape)
            lookup_size = max(
                1,
                tile_count * np.dtype(np.int32).itemsize,
            )

            if self._tile_lookup is None or self._tile_lookup.size != lookup_size:
                self._tile_lookup = cl.Buffer(
                    queue.context,
                    cl.mem_flags.READ_WRITE,
                    lookup_size,
                )

            metadata = PreviewMetadata(
                frame_index=int(frame_index),
                active_tile_count=int(active_tile_count),
                tile_shape=tuple(int(value) for value in tile_shape),
                atlas_tile_shape=atlas_shape,
                tile_size=int(tile_size),
            )

            cl.enqueue_fill_buffer(
                queue,
                self._counter,
                np.asarray(0, dtype=np.int32),
                0,
                4,
            )

            self._kernel.set_args(
                tile_map,
                smoke,
                flame,
                self._tile_lookup,
                self._fields,
                self._counter,
                np.int32(active_tile_count),
                np.int32(tile_shape[0]),
                np.int32(tile_shape[1]),
                np.int32(tile_shape[2]),
                np.int32(atlas_shape[0]),
                np.int32(atlas_shape[1]),
                np.int32(tile_size),
            )

            event = cl.enqueue_nd_range_kernel(
                queue,
                self._kernel,
                tile_shape,
                None,
            )

            with self._lock:
                self._metadata = metadata
                self._publish_event = event

            event.set_callback(
                cl.command_execution_status.COMPLETE,
                self._publication_complete,
            )

            return True

        except Exception:
            with self._lock:
                self._state = self.FREE
                self._metadata = None
                self._publish_event = None

            raise

    def _publication_complete(self, status: int) -> None:
        with self._lock:
            if self._state != self.PUBLISHING:
                return

            if status == cl.command_execution_status.COMPLETE and self._enabled:
                self._state = self.READY
            else:
                self._state = self.FREE
                self._metadata = None
                self._publish_event = None

    def begin_transfer(self) -> PreviewTransfer | None:
        """Acquire the newest snapshot and begin the isolated staging readback."""

        with self._lock:
            self._refresh_publication_locked()

            if not self._enabled or self._state != self.READY:
                return None

            self._state = self.ACQUIRED

            metadata = self._metadata
            publish_event = self._publish_event
            transfer_queue = self._transfer_queue
            tile_lookup_buffer = self._tile_lookup
            fields_buffer = self._fields

        tile_lookup = np.empty(
            (
                metadata.tile_shape[2],
                metadata.tile_shape[1],
                metadata.tile_shape[0],
            ),
            dtype=np.float32,
        )

        atlas_cells = math.prod(metadata.atlas_tile_shape) * metadata.tile_size**3

        fields = np.empty(
            atlas_cells * 2,
            dtype=np.float32,
        )

        wait_for = [publish_event] if publish_event is not None else None

        lookup_event = cl.enqueue_copy(
            transfer_queue,
            tile_lookup,
            tile_lookup_buffer,
            is_blocking=False,
            wait_for=wait_for,
        )

        fields_event = cl.enqueue_copy(
            transfer_queue,
            fields,
            fields_buffer,
            is_blocking=False,
            wait_for=wait_for,
        )

        event = cl.enqueue_marker(
            transfer_queue,
            wait_for=[
                lookup_event,
                fields_event,
            ],
        )

        return PreviewTransfer(
            metadata,
            tile_lookup,
            fields,
            event,
        )

    def release(self) -> None:
        with self._lock:
            if self._state == self.ACQUIRED:
                self._state = self.FREE
                self._metadata = None
                self._publish_event = None

    def release_when_complete(
        self,
        transfer: PreviewTransfer,
    ) -> None:
        """Release an abandoned acquisition after its readback has completed."""

        transfer.event.set_callback(
            cl.command_execution_status.COMPLETE,
            lambda _status: self.release(),
        )
