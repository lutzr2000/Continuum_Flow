import numpy as np
from numba import njit, prange

import Solver.General.obstacles as obstacles
import Solver.General.sources as general_sources


@njit(cache=True, parallel=True)
def _sample_source_entry_cpu(
    source_mask,
    entry_mask,
    base,
    delta,
    ox,
    oy,
    oz,
    ix0,
    ix1,
    iy0,
    iy1,
    iz0,
    iz1,
    base_ox,
    base_oy,
    base_oz,
    inv,
):
    """
    Sample one dynamic source entry into its own boolean mask and the aggregate mask.
    """
    entry_mask.fill(False)

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

        if not (
            0 <= bi < bn_x and 0 <= bj < bn_y and 0 <= bk < bn_z and base[bi, bj, bk]
        ):
            continue

        entry_mask[i, j, k] = True
        source_mask[i, j, k] = True


def build_source_data(domain_cfg, source_entries):
    """
    Build per-source masks and authored values from exported source nodes.
    """
    return general_sources.build_source_data(domain_cfg, source_entries, obstacles)


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
        runtime = runtime_entry["runtime"]
        shape = runtime["shape"]
        origin = np.asarray(runtime["origin"], dtype=np.float32)
        delta = np.float32(runtime["delta"])
        entry_mask = runtime_entry["mask"]

        for obj in runtime.get("objects", ()):
            state = obstacles._resolve_dynamic_object_state(
                obj, time_value, delta, origin, shape
            )
            if not state["active"]:
                entry_mask.fill(False)
                continue

            ix0, ix1, iy0, iy1, iz0, iz1 = state["index_bounds"]
            base_origin = obj["local_origin"]
            _sample_source_entry_cpu(
                source_active_mask,
                entry_mask,
                obj["local_mask"],
                delta,
                np.float32(origin[0]),
                np.float32(origin[1]),
                np.float32(origin[2]),
                int(ix0),
                int(ix1),
                int(iy0),
                int(iy1),
                int(iz0),
                int(iz1),
                np.float32(base_origin[0]),
                np.float32(base_origin[1]),
                np.float32(base_origin[2]),
                state["inv"],
            )

            if np.any(entry_mask):
                any_source = True

    source_data["last_has_source"] = bool(any_source)
    return source_data


@njit(cache=True, parallel=True)
def _source_bc_kernel_cpu(
    u,
    v,
    w,
    T,
    smoke,
    fuel,
    source_mask,
    source_entry_masks,
    source_temperature_values,
    source_smoke_values,
    source_fuel_values,
    source_velocity_enabled,
    source_velocity_x_values,
    source_velocity_y_values,
    source_velocity_z_values,
    dt,
    apply_velocity,
    apply_scalars,
):
    """
    Apply source velocity/temperature and inject smoke/fuel rates on the CPU.
    """
    total_size = source_mask.size
    u_flat = u.reshape(total_size)
    v_flat = v.reshape(total_size)
    w_flat = w.reshape(total_size)
    T_flat = T.reshape(total_size)
    smoke_flat = smoke.reshape(total_size)
    fuel_flat = fuel.reshape(total_size)
    source_mask_flat = source_mask.reshape(total_size)
    source_count = source_entry_masks.shape[0]
    nx, ny, nz = source_mask.shape

    for idx in prange(total_size):
        if not source_mask_flat[idx]:
            continue

        i = idx // (ny * nz)
        remainder = idx % (ny * nz)
        j = remainder // nz
        k = remainder % nz

        source_temperature_value = 0.0
        source_smoke_value = 0.0
        source_fuel_value = 0.0
        velocity_x_value = 0.0
        velocity_y_value = 0.0
        velocity_z_value = 0.0
        has_velocity_target = False

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

        if apply_velocity and has_velocity_target:
            u_flat[idx] = velocity_x_value
            v_flat[idx] = velocity_y_value
            w_flat[idx] = velocity_z_value

        if apply_scalars:
            if T_flat[idx] < source_temperature_value:
                T_flat[idx] = source_temperature_value

            smoke_flat[idx] = min(
                max(smoke_flat[idx] + dt * 10.0 * source_smoke_value, 0.0),
                100.0,
            )
            fuel_flat[idx] = min(
                max(fuel_flat[idx] + dt * 10.0 * source_fuel_value, 0.0),
                100.0,
            )


def source_bc(
    u,
    v,
    w,
    T,
    smoke,
    fuel,
    source_mask,
    source_entry_masks,
    source_temperature_values,
    source_smoke_values,
    source_fuel_values,
    source_velocity_enabled,
    source_velocity_x_values,
    source_velocity_y_values,
    source_velocity_z_values,
    dt,
    apply_velocity=True,
    apply_scalars=True,
):
    """
    Apply all source boundary conditions to the CPU field state.
    """
    if source_mask.size == 0 or not np.any(source_mask):
        return u, v, w, T, smoke, fuel

    _source_bc_kernel_cpu(
        u,
        v,
        w,
        T,
        smoke,
        fuel,
        source_mask,
        source_entry_masks,
        source_temperature_values,
        source_smoke_values,
        source_fuel_values,
        source_velocity_enabled,
        source_velocity_x_values,
        source_velocity_y_values,
        source_velocity_z_values,
        dt,
        apply_velocity,
        apply_scalars,
    )
    return u, v, w, T, smoke, fuel
