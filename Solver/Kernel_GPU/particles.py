from pathlib import Path
from typing import Any
import math

import numpy as np
from numba import cuda

import Solver.Kernel_GPU.kernel_config as kernel_config


def load_particle_sources(
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
                times = np.asarray(archive["times"], dtype=np.float32)
                offsets = np.asarray(archive["offsets"], dtype=np.uint32)
                positions = np.asarray(archive["positions"], dtype=np.float32).reshape(
                    -1, 3
                )
                sizes = np.asarray(archive["sizes"], dtype=np.float32)
                velocities = np.asarray(
                    archive["velocities"], dtype=np.float32
                ).reshape(-1, 3)

            if reference_matrix_data is not None:
                transform_particle_frames_to_reference(
                    times,
                    offsets,
                    positions,
                    velocities,
                    reference_matrix_data,
                )

            frame_counts = np.diff(offsets).astype(np.int64, copy=False)
            max_frame_count = int(frame_counts.max()) if frame_counts.size else 0
            buffer_shape = (max_frame_count, 3)

            source_entries.append(
                {
                    "times": times,
                    "offsets": offsets,
                    "positions": positions,
                    "sizes": sizes,
                    "velocities": velocities,
                    "current_positions_device": cuda.device_array(
                        buffer_shape, dtype=np.float32
                    ),
                    "next_positions_device": cuda.device_array(
                        buffer_shape, dtype=np.float32
                    ),
                    "current_velocities_device": cuda.device_array(
                        buffer_shape, dtype=np.float32
                    ),
                    "next_velocities_device": cuda.device_array(
                        buffer_shape, dtype=np.float32
                    ),
                    "previous_sample_positions_device": cuda.device_array(
                        buffer_shape, dtype=np.float32
                    ),
                    "current_sample_positions_device": cuda.device_array(
                        buffer_shape, dtype=np.float32
                    ),
                    "sample_time": None,
                    "sample_count": 0,
                    "loaded_frame_index": -1,
                    "radius": np.float32(particle_input.get("radius", 0.0)),
                    "velocity_transfer": np.float32(
                        particle_input.get("velocity_transfer", 1.0)
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
    """Convert exported world-space particle samples into reference space."""
    from Solver.Kernel_GPU.update_masks import get_matrix_data

    reference_times, reference_matrices, reference_rates = reference_matrix_data
    for frame_index, time_value in enumerate(times):
        start = int(offsets[frame_index])
        end = int(offsets[frame_index + 1])
        if start >= end:
            continue
        reference_matrix, _ = get_matrix_data(
            reference_times,
            reference_matrices,
            reference_rates,
            float(time_value),
        )
        inverse = np.linalg.inv(reference_matrix).astype(np.float32)
        positions[start:end] = positions[start:end] @ inverse[:3, :3].T + inverse[:3, 3]
        velocities[start:end] = velocities[start:end] @ inverse[:3, :3].T


def particle_frame(entry: dict, time_value: float) -> tuple[int, int, float]:
    """
    Upload the current and next frames when the active frame interval changes.
    This avoids holding all particle data in memory for the whole simulation.
    """
    times = entry["times"]
    frame_index = int(np.searchsorted(times, time_value, side="right") - 1)
    frame_index = min(max(frame_index, 0), times.size - 1)
    offsets = entry["offsets"]
    current_start = int(offsets[frame_index])
    current_end = int(offsets[frame_index + 1])
    current_count = current_end - current_start
    next_frame_index = min(frame_index + 1, times.size - 1)
    next_start = int(offsets[next_frame_index])
    next_end = int(offsets[next_frame_index + 1])
    next_count = next_end - next_start

    if entry["loaded_frame_index"] != frame_index:
        if current_count:
            entry["current_positions_device"][:current_count].copy_to_device(
                entry["positions"][current_start:current_end]
            )
            entry["current_velocities_device"][:current_count].copy_to_device(
                entry["velocities"][current_start:current_end]
            )
        if next_count:
            entry["next_positions_device"][:next_count].copy_to_device(
                entry["positions"][next_start:next_end]
            )
            entry["next_velocities_device"][:next_count].copy_to_device(
                entry["velocities"][next_start:next_end]
            )
        entry["loaded_frame_index"] = frame_index

    alpha = 0.0

    if frame_index + 1 < times.size:
        duration = float(times[frame_index + 1] - times[frame_index])
        if duration > 0.0:
            alpha = min(
                max((float(time_value) - float(times[frame_index])) / duration, 0.0),
                1.0,
            )

    return current_count, next_count, alpha


def particle_motion_samples(
    entry: dict, time_value: float
) -> tuple[Any, int, Any, int]:
    """Return particle positions at the previous and current solver times."""
    count, next_count, alpha = particle_frame(entry, time_value)

    if entry["sample_time"] != time_value:
        previous_positions = entry["current_sample_positions_device"]
        current_positions = entry["previous_sample_positions_device"]
        previous_count = entry["sample_count"]

        if count:
            sample_interpolated_vectors[(count + 127) // 128, 128](
                current_positions,
                entry["current_positions_device"],
                entry["next_positions_device"],
                count,
                next_count,
                np.float32(alpha),
            )

        # At the first solver sample there is no path to sweep yet.
        if entry["sample_time"] is None:
            previous_positions, current_positions = (
                current_positions,
                previous_positions,
            )
            if count:
                sample_interpolated_vectors[(count + 127) // 128, 128](
                    current_positions,
                    entry["current_positions_device"],
                    entry["next_positions_device"],
                    count,
                    next_count,
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
        entry.get("previous_sample_count", entry["sample_count"]),
        entry["current_sample_positions_device"],
        entry["sample_count"],
    )


def update_source_tile_mask(
    source_tile_mask: Any,
    particle_sources: list[list[dict]],
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
                particle_motion_samples(entry, time_value)
            )
            if count <= 0:
                continue

            mark_particle_tiles[count, threads](
                source_tile_mask,
                previous_positions,
                current_positions,
                count,
                previous_count,
                entry["radius"],
                np.float32(delta),
                np.float32(origin[0]),
                np.float32(origin[1]),
                np.float32(origin[2]),
            )


def add_particles_to_source_masks(
    source_masks: list[Any],
    particle_sources: list[list[dict]],
    time_value: float,
    delta: float,
    origin: tuple[float, float, float],
    tile_map: Any,
) -> None:
    """
    Union current particle spheres into their corresponding source masks.
    """
    threads = 128

    for source_mask, source_entries in zip(source_masks, particle_sources):
        for entry in source_entries:
            previous_positions, previous_count, current_positions, count = (
                particle_motion_samples(entry, time_value)
            )
            if count <= 0:
                continue

            rasterize_particle_spheres[count, threads](
                source_mask,
                tile_map,
                previous_positions,
                current_positions,
                count,
                previous_count,
                entry["radius"],
                np.float32(delta),
                np.float32(origin[0]),
                np.float32(origin[1]),
                np.float32(origin[2]),
            )


@cuda.jit(device=True, inline=True, cache=True)
def interpolated_particle_vector(
    current_values, next_values, sample_index, next_count, alpha
):
    """
    Linearly interpolate one three-component particle value between two frames.
    """
    px = current_values[sample_index, 0]
    py = current_values[sample_index, 1]
    pz = current_values[sample_index, 2]

    if sample_index < next_count:
        px += (next_values[sample_index, 0] - px) * alpha
        py += (next_values[sample_index, 1] - py) * alpha
        pz += (next_values[sample_index, 2] - pz) * alpha

    return px, py, pz


@cuda.jit(cache=True)
def sample_interpolated_vectors(
    output, current_values, next_values, count, next_count, alpha
):
    """Materialize interpolated vectors for one solver-time sample."""
    sample_index = cuda.grid(1)
    if sample_index >= count:
        return
    x, y, z = interpolated_particle_vector(
        current_values, next_values, sample_index, next_count, alpha
    )
    output[sample_index, 0] = x
    output[sample_index, 1] = y
    output[sample_index, 2] = z


@cuda.jit(device=True, inline=True, cache=True)
def point_segment_distance_squared(px, py, pz, ax, ay, az, bx, by, bz):
    """Squared distance from a point to a finite 3D segment."""
    abx = bx - ax
    aby = by - ay
    abz = bz - az
    apx = px - ax
    apy = py - ay
    apz = pz - az
    length_squared = abx * abx + aby * aby + abz * abz
    t = 0.0
    if length_squared > 0.0:
        t = (apx * abx + apy * aby + apz * abz) / length_squared
        t = min(max(t, 0.0), 1.0)
    dx = px - (ax + t * abx)
    dy = py - (ay + t * aby)
    dz = pz - (az + t * abz)
    return dx * dx + dy * dy + dz * dz


@cuda.jit(device=True, inline=True, cache=True)
def particle_grid_bounds(
    px, py, pz, radius, spacing, origin_x, origin_y, origin_z, size_i, size_j, size_k
):
    """
    Return clamped grid-index bounds for a particle sphere.
    """
    min_i = max(int(math.floor((px - radius - origin_x) / spacing)), 0)
    min_j = max(int(math.floor((py - radius - origin_y) / spacing)), 0)
    min_k = max(int(math.floor((pz - radius - origin_z) / spacing)), 0)
    max_i = min(int(math.floor((px + radius - origin_x) / spacing)), size_i - 1)
    max_j = min(int(math.floor((py + radius - origin_y) / spacing)), size_j - 1)
    max_k = min(int(math.floor((pz + radius - origin_z) / spacing)), size_k - 1)
    return min_i, min_j, min_k, max_i, max_j, max_k


@cuda.jit(device=True, inline=True, cache=True)
def linear_to_grid_index(linear_index, min_i, min_j, min_k, count_j, count_k):
    """
    Decode a linear index into coordinates inside a bounded 3D grid region.
    """
    entries_per_i = count_j * count_k
    local_i = linear_index // entries_per_i
    remainder = linear_index - local_i * entries_per_i
    local_j = remainder // count_k
    local_k = remainder - local_j * count_k
    return min_i + local_i, min_j + local_j, min_k + local_k


@cuda.jit(cache=True)
def mark_particle_tiles(
    tile_mask,
    previous_positions,
    current_positions,
    count,
    previous_count,
    radius,
    delta,
    origin_x,
    origin_y,
    origin_z,
):
    """
    Mark every coarse tile in a particle's swept-sphere bounds.
    """
    sample_index = cuda.blockIdx.x
    if sample_index >= count:
        return

    px = current_positions[sample_index, 0]
    py = current_positions[sample_index, 1]
    pz = current_positions[sample_index, 2]
    previous_px = px
    previous_py = py
    previous_pz = pz
    if sample_index < previous_count:
        previous_px = previous_positions[sample_index, 0]
        previous_py = previous_positions[sample_index, 1]
        previous_pz = previous_positions[sample_index, 2]
    tile_world_size = delta * kernel_config.TILE_SIZE
    min_ti, min_tj, min_tk, max_ti, max_tj, max_tk = particle_grid_bounds(
        min(px, previous_px),
        min(py, previous_py),
        min(pz, previous_pz),
        radius,
        tile_world_size,
        origin_x,
        origin_y,
        origin_z,
        tile_mask.shape[0],
        tile_mask.shape[1],
        tile_mask.shape[2],
    )
    _, _, _, path_max_i, path_max_j, path_max_k = particle_grid_bounds(
        max(px, previous_px),
        max(py, previous_py),
        max(pz, previous_pz),
        radius,
        tile_world_size,
        origin_x,
        origin_y,
        origin_z,
        tile_mask.shape[0],
        tile_mask.shape[1],
        tile_mask.shape[2],
    )
    max_ti = path_max_i
    max_tj = path_max_j
    max_tk = path_max_k
    if min_ti > max_ti or min_tj > max_tj or min_tk > max_tk:
        return

    count_i = max_ti - min_ti + 1
    count_j = max_tj - min_tj + 1
    count_k = max_tk - min_tk + 1
    tile_count = count_i * count_j * count_k

    linear_index = cuda.threadIdx.x

    while linear_index < tile_count:
        ti, tj, tk = linear_to_grid_index(
            linear_index, min_ti, min_tj, min_tk, count_j, count_k
        )
        tile_center_x = origin_x + (ti + 0.5) * tile_world_size
        tile_center_y = origin_y + (tj + 0.5) * tile_world_size
        tile_center_z = origin_z + (tk + 0.5) * tile_world_size
        # A center-to-segment test enlarged by the tile's half diagonal is a
        # conservative capsule/AABB intersection and cannot miss needed tiles.
        tile_reach = radius + 0.8660254037844386 * tile_world_size
        if (
            point_segment_distance_squared(
                tile_center_x,
                tile_center_y,
                tile_center_z,
                previous_px,
                previous_py,
                previous_pz,
                px,
                py,
                pz,
            )
            <= tile_reach * tile_reach
        ):
            tile_mask[ti, tj, tk] = True
        linear_index += cuda.blockDim.x


@cuda.jit(cache=True)
def rasterize_particle_spheres(
    source_mask,
    tile_map,
    previous_positions,
    current_positions,
    count,
    previous_count,
    radius,
    delta,
    origin_x,
    origin_y,
    origin_z,
):
    """
    Union particle swept spheres into a sparse source mask.
    """
    sample_index = cuda.blockIdx.x
    if sample_index >= count:
        return

    px = current_positions[sample_index, 0]
    py = current_positions[sample_index, 1]
    pz = current_positions[sample_index, 2]
    previous_px = px
    previous_py = py
    previous_pz = pz
    if sample_index < previous_count:
        previous_px = previous_positions[sample_index, 0]
        previous_py = previous_positions[sample_index, 1]
        previous_pz = previous_positions[sample_index, 2]
    tile_size = kernel_config.TILE_SIZE
    min_i, min_j, min_k, max_i, max_j, max_k = particle_grid_bounds(
        min(px, previous_px),
        min(py, previous_py),
        min(pz, previous_pz),
        radius,
        delta,
        origin_x,
        origin_y,
        origin_z,
        tile_map.shape[0] * tile_size,
        tile_map.shape[1] * tile_size,
        tile_map.shape[2] * tile_size,
    )
    _, _, _, path_max_i, path_max_j, path_max_k = particle_grid_bounds(
        max(px, previous_px),
        max(py, previous_py),
        max(pz, previous_pz),
        radius,
        delta,
        origin_x,
        origin_y,
        origin_z,
        tile_map.shape[0] * tile_size,
        tile_map.shape[1] * tile_size,
        tile_map.shape[2] * tile_size,
    )
    max_i = path_max_i
    max_j = path_max_j
    max_k = path_max_k
    if min_i > max_i or min_j > max_j or min_k > max_k:
        return

    radius_squared = radius * radius
    count_i = max_i - min_i + 1
    count_j = max_j - min_j + 1
    count_k = max_k - min_k + 1
    cell_count = count_i * count_j * count_k

    linear_index = cuda.threadIdx.x

    while linear_index < cell_count:
        i, j, k = linear_to_grid_index(
            linear_index, min_i, min_j, min_k, count_j, count_k
        )
        cell_x = origin_x + (i + 0.5) * delta
        cell_y = origin_y + (j + 0.5) * delta
        cell_z = origin_z + (k + 0.5) * delta

        if (
            point_segment_distance_squared(
                cell_x,
                cell_y,
                cell_z,
                previous_px,
                previous_py,
                previous_pz,
                px,
                py,
                pz,
            )
            <= radius_squared
        ):
            ti = i // tile_size
            tj = j // tile_size
            tk = k // tile_size
            pool_index = tile_map[ti, tj, tk]
            if pool_index >= 0:
                source_mask[
                    pool_index,
                    i - ti * tile_size,
                    j - tj * tile_size,
                    k - tk * tile_size,
                ] = True
        linear_index += cuda.blockDim.x


def reset_particle_velocity(
    u: Any,
    v: Any,
    w: Any,
    source_entries: list[dict],
    time_value: float,
    delta: float,
    origin: tuple[float, float, float],
    tile_map: Any,
) -> None:
    """Reset particle-covered cells before the unchanged additive transfer."""
    threads = 128
    for entry in source_entries:
        count, next_count, alpha = particle_frame(entry, time_value)
        if count <= 0:
            continue

        reset_particle_velocity_kernel[count, threads](
            u,
            v,
            w,
            tile_map,
            entry["current_positions_device"],
            entry["next_positions_device"],
            count,
            next_count,
            np.float32(alpha),
            entry["radius"],
            np.float32(delta),
            np.float32(origin[0]),
            np.float32(origin[1]),
            np.float32(origin[2]),
        )


@cuda.jit(cache=True)
def reset_particle_velocity_kernel(
    u,
    v,
    w,
    tile_map,
    current_positions,
    next_positions,
    count,
    next_count,
    alpha,
    radius,
    delta,
    origin_x,
    origin_y,
    origin_z,
):
    """Set velocity to zero only in cells covered by particle spheres."""
    sample_index = cuda.blockIdx.x
    if sample_index >= count:
        return

    px, py, pz = interpolated_particle_vector(
        current_positions, next_positions, sample_index, next_count, alpha
    )
    tile_size = kernel_config.TILE_SIZE
    min_i, min_j, min_k, max_i, max_j, max_k = particle_grid_bounds(
        px,
        py,
        pz,
        radius,
        delta,
        origin_x,
        origin_y,
        origin_z,
        tile_map.shape[0] * tile_size,
        tile_map.shape[1] * tile_size,
        tile_map.shape[2] * tile_size,
    )
    if min_i > max_i or min_j > max_j or min_k > max_k:
        return

    radius_squared = radius * radius
    count_j = max_j - min_j + 1
    count_k = max_k - min_k + 1
    cell_count = (max_i - min_i + 1) * count_j * count_k

    linear_index = cuda.threadIdx.x
    while linear_index < cell_count:
        i, j, k = linear_to_grid_index(
            linear_index, min_i, min_j, min_k, count_j, count_k
        )
        dx = origin_x + (i + 0.5) * delta - px
        dy = origin_y + (j + 0.5) * delta - py
        dz = origin_z + (k + 0.5) * delta - pz
        if dx * dx + dy * dy + dz * dz <= radius_squared:
            ti = i // tile_size
            tj = j // tile_size
            tk = k // tile_size
            pool_index = tile_map[ti, tj, tk]
            if pool_index >= 0:
                index = (
                    pool_index,
                    i - ti * tile_size,
                    j - tj * tile_size,
                    k - tk * tile_size,
                )
                u[index] = 0.0
                v[index] = 0.0
                w[index] = 0.0

        linear_index += cuda.blockDim.x


def transfer_particle_velocities(
    u: Any,
    v: Any,
    w: Any,
    particle_sources: list[list[dict]],
    time_value: float,
    delta: float,
    origin: tuple[float, float, float],
    tile_map: Any,
) -> None:
    """
    Add interpolated particle velocities to cells covered by particle spheres.
    """
    threads = 128
    for source_entries in particle_sources:
        for entry in source_entries:
            velocity_transfer = entry["velocity_transfer"]
            if velocity_transfer == 0.0:
                continue

            count, next_count, alpha = particle_frame(entry, time_value)
            if count <= 0:
                continue

            transfer_particle_velocities_gpu[count, threads](
                u,
                v,
                w,
                tile_map,
                entry["current_positions_device"],
                entry["next_positions_device"],
                entry["current_velocities_device"],
                entry["next_velocities_device"],
                count,
                next_count,
                np.float32(alpha),
                entry["radius"],
                velocity_transfer,
                np.float32(delta),
                np.float32(origin[0]),
                np.float32(origin[1]),
                np.float32(origin[2]),
            )


@cuda.jit(cache=True)
def transfer_particle_velocities_gpu(
    u,
    v,
    w,
    tile_map,
    current_positions,
    next_positions,
    current_velocities,
    next_velocities,
    count,
    next_count,
    alpha,
    radius,
    velocity_transfer,
    delta,
    origin_x,
    origin_y,
    origin_z,
):
    """
    Atomically add scaled, interpolated particle velocity inside each sphere.
    """
    sample_index = cuda.blockIdx.x
    if sample_index >= count:
        return

    px, py, pz = interpolated_particle_vector(
        current_positions, next_positions, sample_index, next_count, alpha
    )
    vx, vy, vz = interpolated_particle_vector(
        current_velocities, next_velocities, sample_index, next_count, alpha
    )
    vx *= velocity_transfer
    vy *= velocity_transfer
    vz *= velocity_transfer

    tile_size = kernel_config.TILE_SIZE
    min_i, min_j, min_k, max_i, max_j, max_k = particle_grid_bounds(
        px,
        py,
        pz,
        radius,
        delta,
        origin_x,
        origin_y,
        origin_z,
        tile_map.shape[0] * tile_size,
        tile_map.shape[1] * tile_size,
        tile_map.shape[2] * tile_size,
    )
    if min_i > max_i or min_j > max_j or min_k > max_k:
        return

    radius_squared = radius * radius
    count_j = max_j - min_j + 1
    count_k = max_k - min_k + 1
    cell_count = (max_i - min_i + 1) * count_j * count_k

    linear_index = cuda.threadIdx.x
    while linear_index < cell_count:
        i, j, k = linear_to_grid_index(
            linear_index, min_i, min_j, min_k, count_j, count_k
        )
        dx = origin_x + (i + 0.5) * delta - px
        dy = origin_y + (j + 0.5) * delta - py
        dz = origin_z + (k + 0.5) * delta - pz
        if dx * dx + dy * dy + dz * dz <= radius_squared:
            ti = i // tile_size
            tj = j // tile_size
            tk = k // tile_size
            pool_index = tile_map[ti, tj, tk]
            if pool_index >= 0:
                index = (
                    pool_index,
                    i - ti * tile_size,
                    j - tj * tile_size,
                    k - tk * tile_size,
                )
                cuda.atomic.add(u, index, vx)
                cuda.atomic.add(v, index, vy)
                cuda.atomic.add(w, index, vz)

        linear_index += cuda.blockDim.x
