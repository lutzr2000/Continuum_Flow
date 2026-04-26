import math

import numpy as np
from numba import cuda

import Kernel.Boundary_Conditions.Obstacles as Obstacles
from Kernel.Kernel_Config import THREADS_PER_BLOCK_3D, volume_blocks_per_grid


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
    ox, oy, oz,
    ix0, iy0, iz0,
    sx, sy, sz,
    base_ox, base_oy, base_oz,
    m00, m01, m02, m03,
    m10, m11, m12, m13,
    m20, m21, m22, m23,
    temperature_value,
    smoke_value,
    fuel_value,
    has_velocity_target,
    velocity_x_value,
    velocity_y_value,
    velocity_z_value,
):
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

    Args:
        domain_cfg (dict): exported domain configuration
        source_entries (list[dict]): exported source node configurations
    Returns:
        dict: source mask and persistent source target fields
    """
    nx = int(domain_cfg["grid"]["nx"])
    ny = int(domain_cfg["grid"]["ny"])
    nz = int(domain_cfg["grid"]["nz"])
    delta = float(domain_cfg["resolution"])
    origin_x = -0.5 * nx * delta
    origin_y = -0.5 * ny * delta
    origin_z = 0.0

    temperature_field = np.zeros((nx, ny, nz), dtype=np.float32)
    smoke_field = np.zeros((nx, ny, nz), dtype=np.float32)
    fuel_field = np.zeros((nx, ny, nz), dtype=np.float32)
    velocity_x_field = np.zeros((nx, ny, nz), dtype=np.float32)
    velocity_y_field = np.zeros((nx, ny, nz), dtype=np.float32)
    velocity_z_field = np.zeros((nx, ny, nz), dtype=np.float32)
    source_active_mask = np.zeros((nx, ny, nz), dtype=np.bool_)
    velocity_active_mask = np.zeros((nx, ny, nz), dtype=np.bool_)
    runtime_entries = []

    for source_entry in source_entries:
        if source_entry.get("shape") != "mesh":
            continue

        mesh_cfg = source_entry.get("mesh", {})
        mesh_objects = mesh_cfg.get("objects", ())
        if not mesh_objects:
            continue

        source_runtime = Obstacles.build_dynamic_runtime(
            nx, ny, nz, delta, mesh_objects,
            origin_x=origin_x, origin_y=origin_y, origin_z=origin_z,
        )
        source_mask = Obstacles.update_dynamic_mask(source_runtime, 0.0)

        velocity = source_entry.get("velocity", (0.0, 0.0, 0.0))
        velocity_x = np.float32(velocity[0] if len(velocity) > 0 else 0.0)
        velocity_y = np.float32(velocity[1] if len(velocity) > 1 else 0.0)
        velocity_z = np.float32(velocity[2] if len(velocity) > 2 else 0.0)
        has_velocity_target = bool(
            velocity_x != 0.0 or velocity_y != 0.0 or velocity_z != 0.0
        )

        runtime_entries.append(
            {
                "runtime": source_runtime,
                "mask": source_mask.copy(),
                "temperature": np.float32(source_entry.get("temperature", 0.0)),
                "smoke": np.float32(source_entry.get("smoke", 0.0)),
                "fuel": np.float32(source_entry.get("fuel", 0.0)),
                "velocity_x": velocity_x,
                "velocity_y": velocity_y,
                "velocity_z": velocity_z,
                "has_velocity_target": has_velocity_target,
            }
        )

        if not np.any(source_mask):
            continue

        source_active_mask |= source_mask
        temperature_field[source_mask] = np.maximum(
            temperature_field[source_mask],
            np.float32(source_entry.get("temperature", 0.0)),
        )
        smoke_field[source_mask] = np.maximum(
            smoke_field[source_mask],
            np.float32(source_entry.get("smoke", 0.0)),
        )
        fuel_field[source_mask] = np.maximum(
            fuel_field[source_mask],
            np.float32(source_entry.get("fuel", 0.0)),
        )
        if has_velocity_target:
            velocity_active_mask[source_mask] = True
            velocity_x_field[source_mask] = velocity_x
            velocity_y_field[source_mask] = velocity_y
            velocity_z_field[source_mask] = velocity_z

    return {
        "mask": source_active_mask,
        "velocity_mask": velocity_active_mask,
        "temperature": temperature_field,
        "smoke": smoke_field,
        "fuel": fuel_field,
        "velocity_x": velocity_x_field,
        "velocity_y": velocity_y_field,
        "velocity_z": velocity_z_field,
        "runtime_entries": runtime_entries,
    }


def prepare_source_data_for_gpu(source_data):
    """Upload static local voxel masks for dynamic source sampling on the GPU."""
    for runtime_entry in source_data.get("runtime_entries", ()):
        Obstacles.prepare_dynamic_runtime_for_gpu(runtime_entry["runtime"])
    source_data["gpu_ready"] = True
    return source_data


def update_source_data_gpu(source_data, device_state, time_value):
    """Rebuild dynamic source masks and fields directly on the GPU."""
    if not source_data.get("gpu_ready"):
        prepare_source_data_for_gpu(source_data)

    source_mask = device_state["source_mask"]
    source_velocity_mask = device_state["source_velocity_mask"]
    source_temperature = device_state["source_temperature"]
    source_smoke = device_state["source_smoke"]
    source_fuel = device_state["source_fuel"]
    source_velocity_x = device_state["source_velocity_x"]
    source_velocity_y = device_state["source_velocity_y"]
    source_velocity_z = device_state["source_velocity_z"]

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

            matrix = Obstacles._matrix_at(obj["transform_series"], time_value)
            bounds = Obstacles._transform_bounds(obj["local_bounds_min"], obj["local_bounds_max"], matrix)
            ix0, ix1, iy0, iy1, iz0, iz1 = Obstacles._bounds_to_indices(
                bounds[0], bounds[1], delta, origin, shape=shape
            )
            if ix0 > ix1 or iy0 > iy1 or iz0 > iz1:
                continue

            any_source = True
            inv = Obstacles._as_f32(np.linalg.inv(matrix))
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
                np.float32(origin[0]), np.float32(origin[1]), np.float32(origin[2]),
                int(ix0), int(iy0), int(iz0),
                sx, sy, sz,
                np.float32(base_origin[0]), np.float32(base_origin[1]), np.float32(base_origin[2]),
                np.float32(inv[0, 0]), np.float32(inv[0, 1]), np.float32(inv[0, 2]), np.float32(inv[0, 3]),
                np.float32(inv[1, 0]), np.float32(inv[1, 1]), np.float32(inv[1, 2]), np.float32(inv[1, 3]),
                np.float32(inv[2, 0]), np.float32(inv[2, 1]), np.float32(inv[2, 2]), np.float32(inv[2, 3]),
                np.float32(runtime_entry["temperature"]),
                np.float32(runtime_entry["smoke"]),
                np.float32(runtime_entry["fuel"]),
                bool(runtime_entry["has_velocity_target"]),
                np.float32(runtime_entry["velocity_x"]),
                np.float32(runtime_entry["velocity_y"]),
                np.float32(runtime_entry["velocity_z"]),
            )

    source_data["last_has_source"] = bool(any_source)
    return source_data


def update_source_data(source_data, time_value):
    """Rebuild source masks and persistent source target fields for the current time."""
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
        source_mask = Obstacles.update_dynamic_mask(
            runtime_entry["runtime"],
            time_value,
            out_mask=runtime_entry["mask"],
        )
        if not np.any(source_mask):
            continue

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
            velocity_x_field[source_mask] = runtime_entry["velocity_x"]
            velocity_y_field[source_mask] = runtime_entry["velocity_y"]
            velocity_z_field[source_mask] = runtime_entry["velocity_z"]

    return source_data


@cuda.jit
def _source_bc_kernel(
    u, v, w, T, smoke, fuel,
    source_mask, source_velocity_mask,
    source_temperature, source_smoke, source_fuel,
    source_velocity_x, source_velocity_y, source_velocity_z,
):
    """
    clamps source regions to persistent source maxima on the GPU.

    Each thread checks one source cell, sets the configured source velocity,
    and raises temperature, smoke and fuel to the configured source values
    when the source mask is active there.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        T (device array): temperature field
        smoke (device array): smoke field
        fuel (device array): fuel field
        source_mask (device array): boolean source mask
        source_velocity_mask (device array): boolean mask for cells with imposed velocity
        source_temperature (device array): source temperature targets
        source_smoke (device array): source smoke targets
        source_fuel (device array): source fuel targets
        source_velocity_x (device array): source x-velocity targets
        source_velocity_y (device array): source y-velocity targets
        source_velocity_z (device array): source z-velocity targets
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = source_mask.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if not source_mask[i, j, k]:
        return

    source_temperature_value = source_temperature[i, j, k]
    source_smoke_value = source_smoke[i, j, k]
    source_fuel_value = source_fuel[i, j, k]

    if source_velocity_mask[i, j, k]:
        u[i, j, k] = source_velocity_x[i, j, k]
        v[i, j, k] = source_velocity_y[i, j, k]
        w[i, j, k] = source_velocity_z[i, j, k]

    if T[i, j, k] < source_temperature_value:
        T[i, j, k] = source_temperature_value
    if smoke[i, j, k] < source_smoke_value:
        smoke[i, j, k] = source_smoke_value
    if fuel[i, j, k] < source_fuel_value:
        fuel[i, j, k] = source_fuel_value


def source_bc(
    u, v, w, T, smoke, fuel,
    source_mask, source_velocity_mask,
    source_temperature, source_smoke, source_fuel,
    source_velocity_x, source_velocity_y, source_velocity_z,
    threadsperblock=None,
):
    """
    applies all source boundary conditions to the GPU field state.

    Velocity is imposed directly and temperature, smoke and fuel are clamped to
    their persistent source target values inside active source cells.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        T (device array): temperature field
        smoke (device array): smoke field
        fuel (device array): fuel field
        source_mask (device array): boolean source mask
        source_velocity_mask (device array): boolean mask for cells with imposed velocity
        source_temperature (device array): source temperature targets
        source_smoke (device array): source smoke targets
        source_fuel (device array): source fuel targets
        source_velocity_x (device array): source x-velocity targets
        source_velocity_y (device array): source y-velocity targets
        source_velocity_z (device array): source z-velocity targets
        threadsperblock (tuple[int, int, int], optional): CUDA block shape for volume kernels
    Returns:
        tuple: updated velocity, temperature, smoke and fuel fields
    """
    if threadsperblock is None:
        threadsperblock = THREADS_PER_BLOCK_3D

    blockspergrid = volume_blocks_per_grid(source_mask.shape, threadsperblock)

    _source_bc_kernel[blockspergrid, threadsperblock](
        u, v, w, T, smoke, fuel,
        source_mask, source_velocity_mask,
        source_temperature, source_smoke, source_fuel,
        source_velocity_x, source_velocity_y, source_velocity_z,
    )
    return u, v, w, T, smoke, fuel
