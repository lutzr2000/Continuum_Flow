from pathlib import Path
from typing import Any
import math

import numpy as np
from numba import cuda

import Solver.Kernel_GPU.kernel_config as kernel_config


def load_particle_sources(sources: list[dict], bake_path: str) -> list[list[dict]]:
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
                    "loaded_frame_index": -1,
                    "radius": np.float32(particle_input.get("radius", 0.0)),
                    "velocity_transfer": np.float32(
                        particle_input.get("velocity_transfer", 1.0)
                    ),
                }
            )
        particle_sources.append(source_entries)

    return particle_sources


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
            count, next_count, alpha = particle_frame(entry, time_value)
            if count <= 0:
                continue

            mark_particle_tiles[count, threads](
                source_tile_mask,
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
            count, next_count, alpha = particle_frame(entry, time_value)
            if count <= 0:
                continue

            rasterize_particle_spheres[count, threads](
                source_mask,
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
    """
    Mark every coarse tile intersected by an interpolated particle sphere.
    """
    sample_index = cuda.blockIdx.x
    if sample_index >= count:
        return

    px, py, pz = interpolated_particle_vector(
        current_positions, next_positions, sample_index, next_count, alpha
    )
    tile_world_size = delta * kernel_config.TILE_SIZE
    min_ti, min_tj, min_tk, max_ti, max_tj, max_tk = particle_grid_bounds(
        px,
        py,
        pz,
        radius,
        tile_world_size,
        origin_x,
        origin_y,
        origin_z,
        tile_mask.shape[0],
        tile_mask.shape[1],
        tile_mask.shape[2],
    )
    if min_ti > max_ti or min_tj > max_tj or min_tk > max_tk:
        return

    radius_squared = radius * radius
    count_i = max_ti - min_ti + 1
    count_j = max_tj - min_tj + 1
    count_k = max_tk - min_tk + 1
    tile_count = count_i * count_j * count_k

    linear_index = cuda.threadIdx.x

    while linear_index < tile_count:
        ti, tj, tk = linear_to_grid_index(
            linear_index, min_ti, min_tj, min_tk, count_j, count_k
        )
        tile_min_x = origin_x + ti * tile_world_size
        tile_min_y = origin_y + tj * tile_world_size
        tile_min_z = origin_z + tk * tile_world_size
        closest_x = min(max(px, tile_min_x), tile_min_x + tile_world_size)
        closest_y = min(max(py, tile_min_y), tile_min_y + tile_world_size)
        closest_z = min(max(pz, tile_min_z), tile_min_z + tile_world_size)
        dx = px - closest_x
        dy = py - closest_y
        dz = pz - closest_z

        if dx * dx + dy * dy + dz * dz <= radius_squared:
            tile_mask[ti, tj, tk] = True
        linear_index += cuda.blockDim.x


@cuda.jit(cache=True)
def rasterize_particle_spheres(
    source_mask,
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
    """
    Union interpolated particle spheres into a sparse source mask.
    """
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
    count_i = max_i - min_i + 1
    count_j = max_j - min_j + 1
    count_k = max_k - min_k + 1
    cell_count = count_i * count_j * count_k

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
                source_mask[
                    pool_index,
                    i - ti * tile_size,
                    j - tj * tile_size,
                    k - tk * tile_size,
                ] = True
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
