"""Solver-owned live-preview publication and OpenCL readback."""

import math
import threading

import numpy as np
import pyopencl as cl

lock = threading.Lock()
enabled = False
snapshot_queue = None
tile_lookup_buffer = None
fields_buffer = None
counter_buffer = None
capacity = -1
publish_event = None
transfer_event = None
pending_metadata = None
pending_snapshot = None
render_settings = None


def configure(config):
    """Configure preview publication directly from the exported solver config."""
    global enabled, render_settings

    simulation = config.get("simulation") or {}
    settings = simulation.get("preview") or {}
    domain = simulation.get("domain") or {}
    grid = domain.get("grid") or {}
    meta = config.get("meta") or {}

    with lock:
        enabled = bool(settings.get("enabled", False))
        render_settings = {
            "grid_shape": (
                int(grid.get("nx", 0)),
                int(grid.get("ny", 0)),
                int(grid.get("nz", 0)),
            ),
            "resolution": float(domain.get("resolution", 1.0)),
            "simulation_reference": (
                str(meta.get("node_tree_name", "")),
                str(simulation.get("node_name", "")),
            ),
        }


def is_enabled():
    with lock:
        return enabled


def atlas_shape(active_count):
    if active_count <= 0:
        return (1, 1, 1)

    side = max(1, math.ceil(active_count ** (1.0 / 3.0)))
    return (side, side, math.ceil(active_count / (side * side)))


def ensure_capacity(context, active_count, tile_size):
    global fields_buffer, counter_buffer, capacity

    if capacity == active_count:
        return

    atlas_cell_count = math.prod(atlas_shape(active_count)) * tile_size**3
    fields_buffer = cl.Buffer(
        context, cl.mem_flags.READ_WRITE, max(1, atlas_cell_count * 2 * 4)
    )
    counter_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, 4)
    capacity = active_count


def try_publish(
    *,
    context,
    queue,
    transfer_queue,
    kernel,
    index_tile_map,
    smoke,
    flame,
    active_tile_count,
    tile_shape,
    tile_size,
):
    """Pack and stage one snapshot without blocking the solver."""
    global snapshot_queue, tile_lookup_buffer, publish_event, pending_metadata

    with lock:
        if not enabled or publish_event is not None or transfer_event is not None:
            return False

    snapshot_queue = transfer_queue
    ensure_capacity(context, active_tile_count, tile_size)
    current_atlas_shape = atlas_shape(active_tile_count)
    lookup_size = max(1, math.prod(tile_shape) * np.dtype(np.float32).itemsize)

    if tile_lookup_buffer is None or tile_lookup_buffer.size != lookup_size:
        tile_lookup_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, lookup_size)

    cl.enqueue_fill_buffer(queue, counter_buffer, np.asarray(0, dtype=np.int32), 0, 4)
    kernel.set_args(
        index_tile_map,
        smoke,
        flame,
        tile_lookup_buffer,
        fields_buffer,
        counter_buffer,
        np.int32(active_tile_count),
        np.int32(tile_shape[0]),
        np.int32(tile_shape[1]),
        np.int32(tile_shape[2]),
        np.int32(current_atlas_shape[0]),
        np.int32(current_atlas_shape[1]),
        np.int32(tile_size),
    )
    publish_event = cl.enqueue_nd_range_kernel(queue, kernel, tile_shape, None)

    with lock:
        pending_metadata = {
            **render_settings,
            "tile_shape": tuple(tile_shape),
            "atlas_tile_shape": current_atlas_shape,
            "tile_size": int(tile_size),
        }
    return True


def take_snapshot():
    """Advance readback and return a complete CPU snapshot when available."""
    global publish_event, transfer_event, pending_metadata, pending_snapshot

    with lock:
        if transfer_event is not None:
            if (
                transfer_event.command_execution_status
                != cl.command_execution_status.COMPLETE
            ):
                return None

            snapshot = pending_snapshot
            transfer_event = None
            pending_snapshot = None
            return snapshot

        if publish_event is None:
            return None
        if (
            publish_event.command_execution_status
            != cl.command_execution_status.COMPLETE
        ):
            return None

        metadata = pending_metadata
        tile_shape = metadata["tile_shape"]
        atlas_shape = metadata["atlas_tile_shape"]
        tile_size = metadata["tile_size"]
        tile_lookup = np.empty(
            (tile_shape[2], tile_shape[1], tile_shape[0]), dtype=np.float32
        )
        fields = np.empty(math.prod(atlas_shape) * tile_size**3 * 2, dtype=np.float32)
        lookup_event = cl.enqueue_copy(
            snapshot_queue,
            tile_lookup,
            tile_lookup_buffer,
            is_blocking=False,
            wait_for=[publish_event],
        )
        fields_event = cl.enqueue_copy(
            snapshot_queue,
            fields,
            fields_buffer,
            is_blocking=False,
            wait_for=[publish_event],
        )
        transfer_event = cl.enqueue_marker(
            snapshot_queue, wait_for=[lookup_event, fields_event]
        )
        pending_snapshot = {
            **metadata,
            "tile_lookup": tile_lookup,
            "fields": fields,
        }
        publish_event = None
        pending_metadata = None
        return None


def allocated_bytes():
    with lock:
        return sum(
            buffer.size
            for buffer in (tile_lookup_buffer, fields_buffer, counter_buffer)
            if buffer is not None
        )


def close():
    """Disable publication and release all solver-owned preview resources."""
    global enabled, snapshot_queue
    global tile_lookup_buffer, fields_buffer, counter_buffer
    global capacity, publish_event, transfer_event
    global pending_metadata, pending_snapshot, render_settings

    with lock:
        enabled = False
        snapshot_queue = None
        tile_lookup_buffer = fields_buffer = counter_buffer = None
        capacity = -1
        publish_event = transfer_event = None
        pending_metadata = pending_snapshot = render_settings = None
