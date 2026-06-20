import math

import numpy as np
from numba import cuda

import Solver.General.sources as general_sources
import Solver.Kernel_GPU.Boundary_Conditions.obstacles as obstacles
from Solver.Kernel_GPU.kernel_config import THREADS_PER_BLOCK_3D, volume_blocks_per_grid


@cuda.jit(cache=True)
def _clear_source_mask_kernel(source_mask):
    """
    Clear one aggregate or per-source mask on the GPU.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = source_mask.shape

    if i >= nx or j >= ny or k >= nz:
        return

    source_mask[i, j, k] = False


@cuda.jit(cache=True)
def _sample_source_entry_cuda(
    source_mask,
    entry_mask,
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

    entry_mask[i, j, k] = True
    source_mask[i, j, k] = True


@cuda.jit(device=True, inline=True, cache=True)
def resolve_source_cell_targets(
    i,
    j,
    k,
    source_entry_masks,
    source_temperature_values,
    source_smoke_values,
    source_fuel_values,
    source_velocity_enabled,
    source_velocity_x_values,
    source_velocity_y_values,
    source_velocity_z_values,
):
    """
    Resolve the authored source targets for one cell by scanning all source masks.
    """
    source_temperature_value = 0.0
    source_smoke_value = 0.0
    source_fuel_value = 0.0
    velocity_x_value = 0.0
    velocity_y_value = 0.0
    velocity_z_value = 0.0
    has_velocity_target = False

    source_count = source_entry_masks.shape[0]
    for source_idx in range(source_count):
        if not source_entry_masks[source_idx, i, j, k]:
            continue

        temperature_value = source_temperature_values[source_idx]
        smoke_value = source_smoke_values[source_idx]
        fuel_value = source_fuel_values[source_idx]

        if source_temperature_value < temperature_value:
            source_temperature_value = temperature_value
        if source_smoke_value < smoke_value:
            source_smoke_value = smoke_value
        if source_fuel_value < fuel_value:
            source_fuel_value = fuel_value

        if source_velocity_enabled[source_idx]:
            has_velocity_target = True
            velocity_x_value = source_velocity_x_values[source_idx]
            velocity_y_value = source_velocity_y_values[source_idx]
            velocity_z_value = source_velocity_z_values[source_idx]

    return (
        source_temperature_value,
        source_smoke_value,
        source_fuel_value,
        has_velocity_target,
        velocity_x_value,
        velocity_y_value,
        velocity_z_value,
    )


@cuda.jit(device=True, inline=True, cache=True)
def accumulate_source_extra_pressure(
    i,
    j,
    k,
    source_entry_masks,
    source_extra_pressure_values,
):
    """
    Accumulate the authored extra pressure from all sources covering one cell.
    """
    extra_pressure_term = 0.0
    source_count = source_entry_masks.shape[0]

    for source_idx in range(source_count):
        if source_entry_masks[source_idx, i, j, k]:
            extra_pressure_term += source_extra_pressure_values[source_idx]

    return extra_pressure_term


def build_source_data(domain_cfg, source_entries):
    """
    Build per-source masks and authored values from exported source nodes.
    """
    return general_sources.build_source_data(domain_cfg, source_entries, obstacles)


def prepare_source_data_for_gpu(source_data):
    """
    Prepare animated source runtime data so it can be reused on the GPU.
    """
    if source_data.get("has_dynamic_masks", False):
        for runtime_entry in source_data.get("runtime_entries", ()):
            obstacles.prepare_dynamic_runtime_for_gpu(runtime_entry["runtime"])
    source_data["gpu_ready"] = True
    return source_data


def update_source_data_gpu(source_data, gpu_fields, time_value):
    """
    Rebuild dynamic source masks directly on the GPU and sync compact source values.
    """
    if not source_data.get("gpu_ready"):
        prepare_source_data_for_gpu(source_data)

    if not source_data.get("has_dynamic_masks", False):
        source_data["last_has_source"] = bool(np.any(source_data["mask"]))
        return source_data

    source_mask = gpu_fields["source_mask"]
    _clear_source_mask_kernel[
        volume_blocks_per_grid(source_mask.shape, THREADS_PER_BLOCK_3D),
        THREADS_PER_BLOCK_3D,
    ](source_mask)

    any_source = False
    for source_idx, runtime_entry in enumerate(source_data.get("runtime_entries", ())):
        runtime = runtime_entry["runtime"]
        shape = runtime["shape"]
        origin = np.asarray(runtime["origin"], dtype=np.float32)
        delta = np.float32(runtime["delta"])
        entry_mask = gpu_fields["source_entry_masks"][source_idx]

        _clear_source_mask_kernel[
            volume_blocks_per_grid(entry_mask.shape, THREADS_PER_BLOCK_3D),
            THREADS_PER_BLOCK_3D,
        ](entry_mask)

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
            inv = state["inv"]
            sx = int(ix1 - ix0 + 1)
            sy = int(iy1 - iy0 + 1)
            sz = int(iz1 - iz0 + 1)
            base_origin = obj["local_origin"]
            blocks = volume_blocks_per_grid((sx, sy, sz), THREADS_PER_BLOCK_3D)

            _sample_source_entry_cuda[blocks, THREADS_PER_BLOCK_3D](
                source_mask,
                entry_mask,
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
            )

    source_data["last_has_source"] = bool(any_source)
    return source_data


def update_source_data(source_data, time_value):
    """
    Rebuild source masks for the current time without allocating per-cell value fields.
    """
    if not source_data.get("has_dynamic_masks", False):
        return general_sources.rebuild_source_mask(source_data)

    source_active_mask = source_data["mask"]
    source_active_mask.fill(False)
    any_source = False

    for runtime_entry in source_data.get("runtime_entries", ()):
        source_mask = obstacles.update_dynamic_mask(
            runtime_entry["runtime"],
            time_value,
            out_mask=runtime_entry["mask"],
        )
        if not np.any(source_mask):
            continue

        source_active_mask |= source_mask
        any_source = True

    source_data["last_has_source"] = bool(any_source)
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
    temperature_value,
    smoke_value,
    fuel_value,
    velocity_x_value,
    velocity_y_value,
    velocity_z_value,
    dt,
):
    """
    Apply source velocity/temperature and inject smoke/fuel rates on the GPU.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = source_mask.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if not source_mask[i, j, k]:
        return

    if velocity_x_value != 0 and velocity_y_value != 0 and velocity_z_value != 0:
        u[i, j, k] = velocity_x_value
        v[i, j, k] = velocity_y_value
        w[i, j, k] = velocity_z_value

    T[i, j, k] = temperature_value

    if smoke_value != 0:
        smoke[i, j, k] = min(
            max(smoke[i, j, k] + dt * 10.0 * smoke_value, 0.0),
            100.0,
        )
    if fuel_value != 0:
        fuel[i, j, k] = min(
            max(fuel[i, j, k] + dt * 10.0 * fuel_value, 0.0),
            100.0,
        )
