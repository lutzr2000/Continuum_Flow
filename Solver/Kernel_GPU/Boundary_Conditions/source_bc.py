import math

import numpy as np
from numba import cuda

import Solver.General.sources as general_sources
import Solver.Kernel_GPU.Boundary_Conditions.obstacles as obstacles
from Solver.Kernel_GPU.kernel_config import THREADS_PER_BLOCK_3D, volume_blocks_per_grid


@cuda.jit(cache=True)
def _clear_source_fields_kernel(
    source_mask,
    source_velocity_mask,
    source_temperature,
    source_smoke,
    source_fuel,
    source_velocity_x,
    source_velocity_y,
    source_velocity_z,
):
    """
    Update the full source masks and target fields by clearing all cells.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = source_mask.shape

    if i >= nx or j >= ny or k >= nz:
        return

    source_mask[i, j, k] = False
    source_velocity_mask[i, j, k] = False
    source_temperature[i, j, k] = 0.0
    source_smoke[i, j, k] = 0.0
    source_fuel[i, j, k] = 0.0
    source_velocity_x[i, j, k] = 0.0
    source_velocity_y[i, j, k] = 0.0
    source_velocity_z[i, j, k] = 0.0


@cuda.jit(cache=True)
def _sample_source_entry_cuda(
    source_mask,
    source_velocity_mask,
    source_temperature,
    source_smoke,
    source_fuel,
    source_velocity_x,
    source_velocity_y,
    source_velocity_z,
    base,
    delta,
    ox,
    oy,
    oz,
    ix0,
    iy0,
    iz0,
    sx,
    sy,
    sz,
    base_ox,
    base_oy,
    base_oz,
    m00,
    m01,
    m02,
    m03,
    m10,
    m11,
    m12,
    m13,
    m20,
    m21,
    m22,
    m23,
    temperature_value,
    smoke_value,
    fuel_value,
    has_velocity_target,
    velocity_x_value,
    velocity_y_value,
    velocity_z_value,
):
    """
    Update one source region by backward-sampling a source object on the GPU.
    """
    di, dj, dk = cuda.grid(3)
    if di >= sx or dj >= sy or dk >= sz:
        return

    i = ix0 + di
    j = iy0 + dj
    k = iz0 + dk

    x = ox + i * delta
    y = oy + j * delta
    z = oz + k * delta

    bx = m00 * x + m01 * y + m02 * z + m03
    by = m10 * x + m11 * y + m12 * z + m13
    bz = m20 * x + m21 * y + m22 * z + m23

    bi = int(math.floor((bx - base_ox) / delta + 0.5))
    bj = int(math.floor((by - base_oy) / delta + 0.5))
    bk = int(math.floor((bz - base_oz) / delta + 0.5))

    bn_x, bn_y, bn_z = base.shape
    if not (0 <= bi < bn_x and 0 <= bj < bn_y and 0 <= bk < bn_z and base[bi, bj, bk]):
        return

    source_mask[i, j, k] = True

    if source_temperature[i, j, k] < temperature_value:
        source_temperature[i, j, k] = temperature_value
    if source_smoke[i, j, k] < smoke_value:
        source_smoke[i, j, k] = smoke_value
    if source_fuel[i, j, k] < fuel_value:
        source_fuel[i, j, k] = fuel_value

    if has_velocity_target:
        source_velocity_mask[i, j, k] = True
        source_velocity_x[i, j, k] = velocity_x_value
        source_velocity_y[i, j, k] = velocity_y_value
        source_velocity_z[i, j, k] = velocity_z_value


def build_source_data(domain_cfg, source_entries):
    """
    build persistent source target fields from exported source nodes.

    Multiple source nodes are merged with a max operation per scalar field.
    Velocity targets are written directly so later overlapping sources override
    earlier ones component-wise.

    """
    return general_sources.build_source_data(domain_cfg, source_entries, obstacles)


def prepare_source_data_for_gpu(source_data):
    """
    Upload static local voxel masks for dynamic source sampling on the GPU.
    """
    if not source_data.get("is_animated", False):
        source_data["gpu_ready"] = False
        return source_data
    for runtime_entry in source_data.get("runtime_entries", ()):
        obstacles.prepare_dynamic_runtime_for_gpu(runtime_entry["runtime"])
    source_data["gpu_ready"] = True
    return source_data


def update_source_data_gpu(source_data, gpu_fields, time_value):
    """
    Rebuild dynamic source masks and fields directly on the GPU.
    """
    if not source_data.get("is_animated", False):
        source_data["last_has_source"] = bool(np.any(source_data["mask"]))
        return source_data
    if not source_data.get("gpu_ready"):
        prepare_source_data_for_gpu(source_data)

    source_mask = gpu_fields["source_mask"]
    source_velocity_mask = gpu_fields["source_velocity_mask"]
    source_temperature = gpu_fields["source_temperature"]
    source_smoke = gpu_fields["source_smoke"]
    source_fuel = gpu_fields["source_fuel"]
    source_velocity_x = gpu_fields["source_velocity_x"]
    source_velocity_y = gpu_fields["source_velocity_y"]
    source_velocity_z = gpu_fields["source_velocity_z"]
    _clear_source_fields_kernel[
        volume_blocks_per_grid(source_mask.shape, THREADS_PER_BLOCK_3D),
        THREADS_PER_BLOCK_3D,
    ](
        source_mask,
        source_velocity_mask,
        source_temperature,
        source_smoke,
        source_fuel,
        source_velocity_x,
        source_velocity_y,
        source_velocity_z,
    )

    any_source = False
    for runtime_entry in source_data.get("runtime_entries", ()):
        runtime = runtime_entry["runtime"]
        shape = runtime["shape"]
        origin = np.asarray(runtime["origin"], dtype=np.float32)
        delta = np.float32(runtime["delta"])

        for obj in runtime.get("objects", ()):
            local_mask_device = obj.get("local_mask_device")
            if local_mask_device is None:
                continue

            state = obstacles._resolve_dynamic_object_state(
                obj, time_value, delta, origin, shape
            )
            if not state["active"]:
                continue
            ix0, ix1, iy0, iy1, iz0, iz1 = state["index_bounds"]

            any_source = True
            resolved_velocity = general_sources.resolve_runtime_entry_velocity(
                runtime_entry,
                time_value,
                obstacles,
            )
            inv = state["inv"]
            sx = int(ix1 - ix0 + 1)
            sy = int(iy1 - iy0 + 1)
            sz = int(iz1 - iz0 + 1)
            base_origin = obj["local_origin"]
            blocks = volume_blocks_per_grid((sx, sy, sz), THREADS_PER_BLOCK_3D)

            _sample_source_entry_cuda[blocks, THREADS_PER_BLOCK_3D](
                source_mask,
                source_velocity_mask,
                source_temperature,
                source_smoke,
                source_fuel,
                source_velocity_x,
                source_velocity_y,
                source_velocity_z,
                local_mask_device,
                delta,
                np.float32(origin[0]),
                np.float32(origin[1]),
                np.float32(origin[2]),
                int(ix0),
                int(iy0),
                int(iz0),
                sx,
                sy,
                sz,
                np.float32(base_origin[0]),
                np.float32(base_origin[1]),
                np.float32(base_origin[2]),
                np.float32(inv[0, 0]),
                np.float32(inv[0, 1]),
                np.float32(inv[0, 2]),
                np.float32(inv[0, 3]),
                np.float32(inv[1, 0]),
                np.float32(inv[1, 1]),
                np.float32(inv[1, 2]),
                np.float32(inv[1, 3]),
                np.float32(inv[2, 0]),
                np.float32(inv[2, 1]),
                np.float32(inv[2, 2]),
                np.float32(inv[2, 3]),
                np.float32(runtime_entry["temperature"]),
                np.float32(runtime_entry["smoke"]),
                np.float32(runtime_entry["fuel"]),
                bool(runtime_entry["has_velocity_target"]),
                np.float32(resolved_velocity[0]),
                np.float32(resolved_velocity[1]),
                np.float32(resolved_velocity[2]),
            )

    source_data["last_has_source"] = bool(any_source)
    return source_data


def update_source_data(source_data, time_value):
    """
    Rebuild source masks and persistent source target fields for the current time.
    """
    source_active_mask = source_data["mask"]
    velocity_active_mask = source_data["velocity_mask"]
    temperature_field = source_data["temperature"]
    smoke_field = source_data["smoke"]
    fuel_field = source_data["fuel"]
    velocity_x_field = source_data["velocity_x"]
    velocity_y_field = source_data["velocity_y"]
    velocity_z_field = source_data["velocity_z"]

    source_active_mask.fill(False)
    velocity_active_mask.fill(False)
    temperature_field.fill(0.0)
    smoke_field.fill(0.0)
    fuel_field.fill(0.0)
    velocity_x_field.fill(0.0)
    velocity_y_field.fill(0.0)
    velocity_z_field.fill(0.0)

    for runtime_entry in source_data.get("runtime_entries", ()):
        source_mask = obstacles.update_dynamic_mask(
            runtime_entry["runtime"],
            time_value,
            out_mask=runtime_entry["mask"],
        )
        if not np.any(source_mask):
            continue

        resolved_velocity = general_sources.resolve_runtime_entry_velocity(
            runtime_entry,
            time_value,
            obstacles,
        )
        source_active_mask |= source_mask
        temperature_field[source_mask] = np.maximum(
            temperature_field[source_mask],
            runtime_entry["temperature"],
        )
        smoke_field[source_mask] = np.maximum(
            smoke_field[source_mask],
            runtime_entry["smoke"],
        )
        fuel_field[source_mask] = np.maximum(
            fuel_field[source_mask],
            runtime_entry["fuel"],
        )

        if runtime_entry["has_velocity_target"]:
            velocity_active_mask[source_mask] = True
            velocity_x_field[source_mask] = resolved_velocity[0]
            velocity_y_field[source_mask] = resolved_velocity[1]
            velocity_z_field[source_mask] = resolved_velocity[2]

    return source_data


@cuda.jit(cache=True)
def source_bc_kernel(
    u,
    v,
    w,
    T,
    smoke,
    fuel,
    source_mask,
    source_velocity_mask,
    source_temperature,
    source_smoke,
    source_fuel,
    source_velocity_x,
    source_velocity_y,
    source_velocity_z,
    dt,
    apply_velocity,
    apply_scalars,
):
    """
    Apply source velocity/temperature and inject smoke/fuel rates on the GPU.

    Each thread checks one source cell, sets the configured source velocity,
    keeps temperature above the configured minimum and injects smoke/fuel
    according to the authored per-second emission rates.

    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = source_mask.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if not source_mask[i, j, k]:
        return

    source_temperature_value = source_temperature[i, j, k]
    # Boost authored source rates so the emitted volume reads denser visually.
    source_smoke_rate = 10.0 * source_smoke[i, j, k]
    source_fuel_rate = 10.0 * source_fuel[i, j, k]

    if apply_velocity and source_velocity_mask[i, j, k]:
        u[i, j, k] = source_velocity_x[i, j, k]
        v[i, j, k] = source_velocity_y[i, j, k]
        w[i, j, k] = source_velocity_z[i, j, k]

    if apply_scalars:
        if T[i, j, k] < source_temperature_value:
            T[i, j, k] = source_temperature_value
        smoke[i, j, k] = min(max(smoke[i, j, k] + dt * source_smoke_rate, 0.0), 100.0)
        fuel[i, j, k] = min(max(fuel[i, j, k] + dt * source_fuel_rate, 0.0), 100.0)
