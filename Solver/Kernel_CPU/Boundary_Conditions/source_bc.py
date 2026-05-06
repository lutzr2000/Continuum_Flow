import numpy as np
from numba import njit, prange

import Solver.General.obstacles as obstacles
import Solver.General.sources as general_sources


@njit(cache=True, parallel=True)
def _clear_source_fields_cpu(
    source_mask,
    source_velocity_mask,
    source_temperature,
    source_smoke,
    source_fuel,
    source_velocity_x,
    source_velocity_y,
    source_velocity_z,
):
    """Clear all persistent source target fields in one parallel volume sweep."""
    total_size = source_mask.size
    source_mask_flat = source_mask.reshape(total_size)
    source_velocity_mask_flat = source_velocity_mask.reshape(total_size)
    source_temperature_flat = source_temperature.reshape(total_size)
    source_smoke_flat = source_smoke.reshape(total_size)
    source_fuel_flat = source_fuel.reshape(total_size)
    source_velocity_x_flat = source_velocity_x.reshape(total_size)
    source_velocity_y_flat = source_velocity_y.reshape(total_size)
    source_velocity_z_flat = source_velocity_z.reshape(total_size)

    for idx in prange(total_size):
        source_mask_flat[idx] = False
        source_velocity_mask_flat[idx] = False
        source_temperature_flat[idx] = 0.0
        source_smoke_flat[idx] = 0.0
        source_fuel_flat[idx] = 0.0
        source_velocity_x_flat[idx] = 0.0
        source_velocity_y_flat[idx] = 0.0
        source_velocity_z_flat[idx] = 0.0


@njit(cache=True, parallel=True)
def _sample_source_entry_cpu(
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
    ix0, ix1, iy0, iy1, iz0, iz1,
    base_ox, base_oy, base_oz,
    inv,
    temperature_value,
    smoke_value,
    fuel_value,
    has_velocity_target,
    velocity_x_value,
    velocity_y_value,
    velocity_z_value,
):
    """Sample one dynamic source entry into the persistent source target fields."""
    bn_x, bn_y, bn_z = base.shape
    sx = ix1 - ix0 + 1
    sy = iy1 - iy0 + 1
    sz = iz1 - iz0 + 1

    for n in prange(sx * sy * sz):
        i = ix0 + n // (sy * sz)
        r = n % (sy * sz)
        j = iy0 + r // sz
        k = iz0 + r % sz

        x = np.float32(ox + i * delta)
        y = np.float32(oy + j * delta)
        z = np.float32(oz + k * delta)

        bx = inv[0, 0] * x + inv[0, 1] * y + inv[0, 2] * z + inv[0, 3]
        by = inv[1, 0] * x + inv[1, 1] * y + inv[1, 2] * z + inv[1, 3]
        bz = inv[2, 0] * x + inv[2, 1] * y + inv[2, 2] * z + inv[2, 3]

        bi = int(np.floor((bx - base_ox) / delta + 0.5))
        bj = int(np.floor((by - base_oy) / delta + 0.5))
        bk = int(np.floor((bz - base_oz) / delta + 0.5))

        if not (0 <= bi < bn_x and 0 <= bj < bn_y and 0 <= bk < bn_z and base[bi, bj, bk]):
            continue

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
    Build persistent source target fields from exported source nodes.

    Multiple source nodes are merged with a max operation per scalar field.
    Velocity targets are written directly so later overlapping sources override
    earlier ones component-wise.
    """
    return general_sources.build_source_data(domain_cfg, source_entries, obstacles)


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

    _clear_source_fields_cpu(
        source_active_mask,
        velocity_active_mask,
        temperature_field,
        smoke_field,
        fuel_field,
        velocity_x_field,
        velocity_y_field,
        velocity_z_field,
    )

    any_source = False
    for runtime_entry in source_data.get("runtime_entries", ()):
        runtime = runtime_entry["runtime"]
        shape = runtime["shape"]
        origin = np.asarray(runtime["origin"], dtype=np.float32)
        delta = np.float32(runtime["delta"])

        for obj in runtime.get("objects", ()):
            state = obstacles._resolve_dynamic_object_state(obj, time_value, delta, origin, shape)
            if not state["active"]:
                continue

            any_source = True
            ix0, ix1, iy0, iy1, iz0, iz1 = state["index_bounds"]
            base_origin = obj["local_origin"]
            _sample_source_entry_cpu(
                source_active_mask,
                velocity_active_mask,
                temperature_field,
                smoke_field,
                fuel_field,
                velocity_x_field,
                velocity_y_field,
                velocity_z_field,
                obj["local_mask"],
                delta,
                np.float32(origin[0]), np.float32(origin[1]), np.float32(origin[2]),
                int(ix0), int(ix1), int(iy0), int(iy1), int(iz0), int(iz1),
                np.float32(base_origin[0]), np.float32(base_origin[1]), np.float32(base_origin[2]),
                state["inv"],
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


@njit(cache=True, parallel=True)
def _source_bc_kernel_cpu(
    u, v, w, T, smoke, fuel,
    source_mask, source_velocity_mask,
    source_temperature, source_smoke, source_fuel,
    source_velocity_x, source_velocity_y, source_velocity_z,
):
    """Clamp active source cells to their persistent CPU source target fields."""
    total_size = source_mask.size
    u_flat = u.reshape(total_size)
    v_flat = v.reshape(total_size)
    w_flat = w.reshape(total_size)
    T_flat = T.reshape(total_size)
    smoke_flat = smoke.reshape(total_size)
    fuel_flat = fuel.reshape(total_size)
    source_mask_flat = source_mask.reshape(total_size)
    source_velocity_mask_flat = source_velocity_mask.reshape(total_size)
    source_temperature_flat = source_temperature.reshape(total_size)
    source_smoke_flat = source_smoke.reshape(total_size)
    source_fuel_flat = source_fuel.reshape(total_size)
    source_velocity_x_flat = source_velocity_x.reshape(total_size)
    source_velocity_y_flat = source_velocity_y.reshape(total_size)
    source_velocity_z_flat = source_velocity_z.reshape(total_size)

    for idx in prange(total_size):
        if not source_mask_flat[idx]:
            continue

        if source_velocity_mask_flat[idx]:
            u_flat[idx] = source_velocity_x_flat[idx]
            v_flat[idx] = source_velocity_y_flat[idx]
            w_flat[idx] = source_velocity_z_flat[idx]

        source_temperature_value = source_temperature_flat[idx]
        source_smoke_value = source_smoke_flat[idx]
        source_fuel_value = source_fuel_flat[idx]

        if T_flat[idx] < source_temperature_value:
            T_flat[idx] = source_temperature_value
        if smoke_flat[idx] < source_smoke_value:
            smoke_flat[idx] = source_smoke_value
        if fuel_flat[idx] < source_fuel_value:
            fuel_flat[idx] = source_fuel_value


def source_bc(
    u, v, w, T, smoke, fuel,
    source_mask, source_velocity_mask,
    source_temperature, source_smoke, source_fuel,
    source_velocity_x, source_velocity_y, source_velocity_z,
):
    """
    Apply all source boundary conditions to the CPU field state.

    Velocity is imposed directly and temperature, smoke and fuel are clamped to
    their persistent source target values inside active source cells.
    """
    if source_mask.size == 0 or not np.any(source_mask):
        return u, v, w, T, smoke, fuel

    _source_bc_kernel_cpu(
        u, v, w, T, smoke, fuel,
        source_mask, source_velocity_mask,
        source_temperature, source_smoke, source_fuel,
        source_velocity_x, source_velocity_y, source_velocity_z,
    )
    return u, v, w, T, smoke, fuel
