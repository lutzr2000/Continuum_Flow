import numpy as np
import pyopencl as cl

from pathlib import Path
from typing import Any

import Solver.Kernel.helper as helper
import Solver.Kernel.kernel_config as kernel_config

FIELD_DTYPE = kernel_config.FIELD_DTYPE


def load_particle_sources(
    context: cl.Context,
    sources: list[dict],
    bake_path: str,
    reference_matrix_data: tuple[Any, Any, Any] | None = None,
) -> list[list[dict]]:
    """
    Load the NPZ particle timelines connected to each source.
    """
    particle_sources = []

    for source in sources:
        source_entries = []

        for particle_input in source.get("particle_system_inputs") or []:
            relative_path = particle_input.get("particle_file")
            file_path = Path(bake_path) / relative_path

            with np.load(file_path, allow_pickle=False) as archive:
                times = np.asarray(
                    archive["times"],
                    dtype=FIELD_DTYPE,
                )

                offsets = np.asarray(
                    archive["offsets"],
                    dtype=np.uint32,
                )

                positions = np.asarray(
                    archive["positions"],
                    dtype=FIELD_DTYPE,
                ).reshape(-1, 3)

                sizes = np.asarray(
                    archive["sizes"],
                    dtype=FIELD_DTYPE,
                )

                velocities = np.asarray(
                    archive["velocities"],
                    dtype=FIELD_DTYPE,
                ).reshape(-1, 3)

            if reference_matrix_data is not None:
                transform_particle_frames_to_reference(
                    times,
                    offsets,
                    positions,
                    velocities,
                    reference_matrix_data,
                )

            frame_counts = np.diff(offsets).astype(
                np.int64,
                copy=False,
            )

            max_frame_count = int(frame_counts.max()) if frame_counts.size else 0

            buffer_shape = (
                max_frame_count,
                3,
            )

            source_entries.append(
                {
                    "times": times,
                    "offsets": offsets,
                    "positions": positions,
                    "sizes": sizes,
                    "velocities": velocities,
                    "current_positions_device": helper.device_array(
                        context,
                        buffer_shape,
                        dtype=FIELD_DTYPE,
                    ),
                    "next_positions_device": helper.device_array(
                        context,
                        buffer_shape,
                        dtype=FIELD_DTYPE,
                    ),
                    "current_velocities_device": helper.device_array(
                        context,
                        buffer_shape,
                        dtype=FIELD_DTYPE,
                    ),
                    "next_velocities_device": helper.device_array(
                        context,
                        buffer_shape,
                        dtype=FIELD_DTYPE,
                    ),
                    "previous_sample_positions_device": helper.device_array(
                        context,
                        buffer_shape,
                        dtype=FIELD_DTYPE,
                    ),
                    "current_sample_positions_device": helper.device_array(
                        context,
                        buffer_shape,
                        dtype=FIELD_DTYPE,
                    ),
                    "sample_time": None,
                    "sample_count": 0,
                    "loaded_frame_index": -1,
                    "radius": np.float32(
                        particle_input.get(
                            "radius",
                            0.0,
                        )
                    ),
                    "velocity_transfer": np.float32(
                        particle_input.get(
                            "velocity_transfer",
                            1.0,
                        )
                    ),
                }
            )

        particle_sources.append(source_entries)

    return particle_sources


def transform_particle_frames_to_reference(
    times: np.ndarray,
    offsets: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    reference_matrix_data: tuple[Any, Any, Any],
) -> None:
    """
    Convert exported world-space particle samples into reference space.
    """
    reference_times, reference_matrices, reference_rates = reference_matrix_data

    for frame_index, time_value in enumerate(times):
        start = int(offsets[frame_index])
        end = int(offsets[frame_index + 1])

        if start >= end:
            continue

        reference_matrix, _ = helper.get_matrix_data(
            reference_times,
            reference_matrices,
            reference_rates,
            float(time_value),
        )

        inverse = np.linalg.inv(reference_matrix).astype(FIELD_DTYPE)

        positions[start:end] = positions[start:end] @ inverse[:3, :3].T + inverse[:3, 3]

        velocities[start:end] = velocities[start:end] @ inverse[:3, :3].T


def update_source_tile_mask(
    queue: cl.CommandQueue,
    particles_kernels: dict[str, cl.Kernel],
    source_tile_mask: Any,
    particle_sources: list[list[dict]],
    tile_shape: tuple[int, int, int],
    time_value: float,
    delta: float,
    origin: tuple[float, float, float],
) -> None:
    """
    Mark every tile touched by a particle sphere.
    """
    threads = 128

    for source_entries in particle_sources:
        for entry in source_entries:
            previous_positions, previous_count, current_positions, count = (
                particle_motion_samples(
                    queue,
                    particles_kernels,
                    entry,
                    time_value,
                )
            )

            if count <= 0:
                continue

            particles_kernels["mark_particle_tiles"](
                queue,
                (count * threads,),
                (threads,),
                source_tile_mask,
                previous_positions,
                current_positions,
                np.int32(count),
                np.int32(previous_count),
                entry["radius"],
                np.float32(delta),
                np.float32(origin[0]),
                np.float32(origin[1]),
                np.float32(origin[2]),
                np.int32(tile_shape[0]),
                np.int32(tile_shape[1]),
                np.int32(tile_shape[2]),
            )


def particle_motion_samples(
    queue: cl.CommandQueue,
    particles_kernels: dict[str, cl.Kernel],
    entry: dict,
    time_value: float,
) -> tuple[Any, int, Any, int]:
    """
    Return particle positions at the previous and current solver times.
    """
    count, next_count, alpha = particle_frame(
        queue,
        entry,
        time_value,
    )

    if entry["sample_time"] != time_value:
        previous_positions = entry["current_sample_positions_device"]

        current_positions = entry["previous_sample_positions_device"]

        previous_count = entry["sample_count"]

        if count:
            particles_kernels["sample_interpolated_vectors"](
                queue,
                (count,),
                None,
                current_positions,
                entry["current_positions_device"],
                entry["next_positions_device"],
                np.int32(count),
                np.int32(next_count),
                np.float32(alpha),
            )

        # At the first solver sample there is no path to sweep yet.
        if entry["sample_time"] is None:
            previous_positions, current_positions = (
                current_positions,
                previous_positions,
            )

            if count:
                particles_kernels["sample_interpolated_vectors"](
                    queue,
                    (count,),
                    None,
                    current_positions,
                    entry["current_positions_device"],
                    entry["next_positions_device"],
                    np.int32(count),
                    np.int32(next_count),
                    np.float32(alpha),
                )

            previous_count = count

        entry["previous_sample_positions_device"] = previous_positions

        entry["current_sample_positions_device"] = current_positions

        entry["sample_count"] = count
        entry["previous_sample_count"] = previous_count
        entry["sample_time"] = time_value

    return (
        entry["previous_sample_positions_device"],
        entry.get(
            "previous_sample_count",
            entry["sample_count"],
        ),
        entry["current_sample_positions_device"],
        entry["sample_count"],
    )


def particle_frame(
    queue: cl.CommandQueue,
    entry: dict,
    time_value: float,
) -> tuple[int, int, float]:
    """
    Upload the current and next frames when the active frame interval changes.
    This avoids holding all particle data in memory for the whole simulation.
    """
    times = entry["times"]

    frame_index = int(
        np.searchsorted(
            times,
            time_value,
            side="right",
        )
        - 1
    )

    frame_index = min(
        max(frame_index, 0),
        times.size - 1,
    )

    offsets = entry["offsets"]

    current_start = int(offsets[frame_index])

    current_end = int(offsets[frame_index + 1])

    current_count = current_end - current_start

    next_frame_index = min(
        frame_index + 1,
        times.size - 1,
    )

    next_start = int(offsets[next_frame_index])

    next_end = int(offsets[next_frame_index + 1])

    next_count = next_end - next_start

    if entry["loaded_frame_index"] != frame_index:
        if current_count:
            cl.enqueue_copy(
                queue,
                entry["current_positions_device"],
                entry["positions"][current_start:current_end],
            )

            cl.enqueue_copy(
                queue,
                entry["current_velocities_device"],
                entry["velocities"][current_start:current_end],
            )

        if next_count:
            cl.enqueue_copy(
                queue,
                entry["next_positions_device"],
                entry["positions"][next_start:next_end],
            )

            cl.enqueue_copy(
                queue,
                entry["next_velocities_device"],
                entry["velocities"][next_start:next_end],
            )

        entry["loaded_frame_index"] = frame_index

    alpha = 0.0

    if frame_index + 1 < times.size:
        duration = float(times[frame_index + 1] - times[frame_index])

        if duration > 0.0:
            alpha = min(
                max(
                    (float(time_value) - float(times[frame_index])) / duration,
                    0.0,
                ),
                1.0,
            )

    return (
        current_count,
        next_count,
        alpha,
    )
