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

            next_indices = build_next_sample_indices(offsets)

            source_entries.append(
                {
                    "times": times,
                    "offsets": offsets,
                    "positions": positions,
                    "sizes": sizes,
                    "velocities": velocities,
                    "positions_device": cuda.to_device(positions),
                    "next_indices_device": cuda.to_device(next_indices),
                    "radius": np.float32(particle_input.get("radius", 0.0)),
                }
            )
        particle_sources.append(source_entries)

    return particle_sources


def build_next_sample_indices(offsets: np.ndarray) -> np.ndarray:
    """
    Map each flattened sample to the same collection index in the following frame.
    """
    total_samples = int(offsets[-1]) if offsets.size else 0
    next_indices = np.full(total_samples, -1, dtype=np.int32)

    for frame_index in range(max(0, offsets.size - 2)):
        start = int(offsets[frame_index])
        end = int(offsets[frame_index + 1])
        next_start = int(offsets[frame_index + 1])
        next_end = int(offsets[frame_index + 2])
        shared_count = min(end - start, next_end - next_start)

        for local_index in range(shared_count):
            next_indices[start + local_index] = next_start + local_index

    return next_indices


def frame_sample_range(entry: dict, time_value: float) -> tuple[int, int, float]:
    """
    Return the flattened sample range at or immediately before a simulation time.
    """
    times = entry["times"]
    frame_index = int(np.searchsorted(times, time_value, side="right") - 1)
    frame_index = min(max(frame_index, 0), times.size - 1)
    offsets = entry["offsets"]
    alpha = 0.0

    if frame_index + 1 < times.size:
        duration = float(times[frame_index + 1] - times[frame_index])
        if duration > 0.0:
            alpha = min(
                max((float(time_value) - float(times[frame_index])) / duration, 0.0),
                1.0,
            )

    return int(offsets[frame_index]), int(offsets[frame_index + 1]), alpha


def update_source_tile_mask(
    source_tile_mask: Any,
    particle_sources: list[list[dict]],
    time_value: float,
    delta: float,
    origin: tuple[float, float, float],
) -> None:
    """
    Mark every coarse tile touched by a current particle sphere.
    """
    threads = 128

    for source_entries in particle_sources:
        for entry in source_entries:
            start, end, alpha = frame_sample_range(entry, time_value)
            count = end - start
            if count <= 0:
                continue

            mark_particle_tiles[count, threads](
                source_tile_mask,
                entry["positions_device"],
                entry["next_indices_device"],
                start,
                end,
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
    Union current particle spheres into their corresponding sparse source masks.
    """
    threads = 128

    for source_mask, source_entries in zip(source_masks, particle_sources):
        for entry in source_entries:
            start, end, alpha = frame_sample_range(entry, time_value)
            count = end - start
            if count <= 0:
                continue

            rasterize_particle_spheres[count, threads](
                source_mask,
                tile_map,
                entry["positions_device"],
                entry["next_indices_device"],
                start,
                end,
                np.float32(alpha),
                entry["radius"],
                np.float32(delta),
                np.float32(origin[0]),
                np.float32(origin[1]),
                np.float32(origin[2]),
            )


@cuda.jit(device=True, inline=True, cache=True)
def interpolated_particle_position(positions, next_indices, sample_index, alpha):
    px = positions[sample_index, 0]
    py = positions[sample_index, 1]
    pz = positions[sample_index, 2]
    next_index = next_indices[sample_index]

    if next_index >= 0:
        px += (positions[next_index, 0] - px) * alpha
        py += (positions[next_index, 1] - py) * alpha
        pz += (positions[next_index, 2] - pz) * alpha

    return px, py, pz


@cuda.jit(device=True, inline=True, cache=True)
def particle_grid_bounds(
    px, py, pz, radius, spacing, origin_x, origin_y, origin_z, size_i, size_j, size_k
):
    min_i = max(int(math.floor((px - radius - origin_x) / spacing)), 0)
    min_j = max(int(math.floor((py - radius - origin_y) / spacing)), 0)
    min_k = max(int(math.floor((pz - radius - origin_z) / spacing)), 0)
    max_i = min(int(math.floor((px + radius - origin_x) / spacing)), size_i - 1)
    max_j = min(int(math.floor((py + radius - origin_y) / spacing)), size_j - 1)
    max_k = min(int(math.floor((pz + radius - origin_z) / spacing)), size_k - 1)
    return min_i, min_j, min_k, max_i, max_j, max_k


@cuda.jit(device=True, inline=True, cache=True)
def linear_to_grid_index(linear_index, min_i, min_j, min_k, count_j, count_k):
    entries_per_i = count_j * count_k
    local_i = linear_index // entries_per_i
    remainder = linear_index - local_i * entries_per_i
    local_j = remainder // count_k
    local_k = remainder - local_j * count_k
    return min_i + local_i, min_j + local_j, min_k + local_k


@cuda.jit(cache=True)
def mark_particle_tiles(
    tile_mask,
    positions,
    next_indices,
    start,
    end,
    alpha,
    radius,
    delta,
    origin_x,
    origin_y,
    origin_z,
):
    sample_index = start + cuda.blockIdx.x
    if sample_index >= end:
        return

    px, py, pz = interpolated_particle_position(
        positions, next_indices, sample_index, alpha
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
    positions,
    next_indices,
    start,
    end,
    alpha,
    radius,
    delta,
    origin_x,
    origin_y,
    origin_z,
):
    sample_index = start + cuda.blockIdx.x
    if sample_index >= end:
        return

    px, py, pz = interpolated_particle_position(
        positions, next_indices, sample_index, alpha
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
