import numpy as np


def _as_f32_vector(values):
    """
    Return a contiguous float32 vector with exactly three components.
    """
    vector = np.zeros(3, dtype=np.float32)
    if values is None:
        return vector
    source = np.asarray(values, dtype=np.float32).reshape(-1)
    limit = min(int(source.size), 3)
    if limit > 0:
        vector[:limit] = source[:limit]
    return vector


def _rotation_from_world_matrix(matrix):
    """
    Extract the closest pure rotation from an affine world matrix.
    """
    linear = np.asarray(matrix, dtype=np.float32)[:3, :3]
    try:
        u, _, vh = np.linalg.svd(
            linear.astype(np.float64, copy=False), full_matrices=False
        )
        rotation = u @ vh
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vh
        return np.asarray(rotation, dtype=np.float32)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float32)


def resolve_local_space_velocity(local_velocity, matrix_world):
    """
    Convert one local-space velocity vector into world space using object orientation.
    """
    local_velocity = _as_f32_vector(local_velocity)
    rotation = _rotation_from_world_matrix(matrix_world)
    return np.asarray(rotation @ local_velocity, dtype=np.float32)


def resolve_runtime_entry_velocity(runtime_entry, time_value, obstacles_backend):
    """
    Resolve the authored source velocity into world space for the current object state.
    """
    authored_velocity = _as_f32_vector(
        (
            runtime_entry.get(
                "authored_velocity_x", runtime_entry.get("velocity_x", 0.0)
            ),
            runtime_entry.get(
                "authored_velocity_y", runtime_entry.get("velocity_y", 0.0)
            ),
            runtime_entry.get(
                "authored_velocity_z", runtime_entry.get("velocity_z", 0.0)
            ),
        )
    )
    if runtime_entry.get("velocity_space", "WORLD") != "LOCAL":
        return authored_velocity

    runtime = runtime_entry.get("runtime") or {}
    objects = runtime.get("objects", ())
    if not objects:
        return authored_velocity

    obj = objects[0]
    state = obstacles_backend._resolve_dynamic_object_state(
        obj,
        time_value,
        np.float32(runtime["delta"]),
        np.asarray(runtime["origin"], dtype=np.float32),
        runtime["shape"],
    )
    return resolve_local_space_velocity(authored_velocity, state["matrix"])


def apply_initial_source_conditions(u, v, w, T, source_data):
    """
    Apply source-authored initial velocity and temperature directly to solver fields.
    """
    for runtime_entry in source_data.get("runtime_entries", ()):
        source_mask = runtime_entry.get("mask")
        if source_mask is None or not np.any(source_mask):
            continue

        if runtime_entry.get("has_velocity_target", False):
            u[source_mask] = runtime_entry.get("initial_velocity_x", 0.0)
            v[source_mask] = runtime_entry.get("initial_velocity_y", 0.0)
            w[source_mask] = runtime_entry.get("initial_velocity_z", 0.0)

        T[source_mask] = np.maximum(T[source_mask], runtime_entry.get("temperature", 0.0))

    return u, v, w, T


def build_runtime_entry_payload(source_data, time_value, obstacles_backend):
    """
    Convert runtime entries into compact per-source masks and value arrays.
    """
    source_mask = np.asarray(source_data["mask"])
    source_shape = source_mask.shape
    runtime_entries = tuple(source_data.get("runtime_entries", ()))
    source_count = len(runtime_entries)

    entry_masks = np.zeros((source_count,) + source_shape, dtype=np.bool_)
    temperature_values = np.zeros(source_count, dtype=np.float32)
    smoke_values = np.zeros(source_count, dtype=np.float32)
    fuel_values = np.zeros(source_count, dtype=np.float32)
    extra_pressure_values = np.zeros(source_count, dtype=np.float32)
    velocity_enabled = np.zeros(source_count, dtype=np.bool_)
    velocity_x_values = np.zeros(source_count, dtype=np.float32)
    velocity_y_values = np.zeros(source_count, dtype=np.float32)
    velocity_z_values = np.zeros(source_count, dtype=np.float32)

    for source_idx, runtime_entry in enumerate(runtime_entries):
        entry_masks[source_idx] = np.asarray(runtime_entry["mask"], dtype=np.bool_)
        temperature_values[source_idx] = np.float32(runtime_entry.get("temperature", 0.0))
        smoke_values[source_idx] = np.float32(runtime_entry.get("smoke", 0.0))
        fuel_values[source_idx] = np.float32(runtime_entry.get("fuel", 0.0))
        extra_pressure_values[source_idx] = np.float32(
            runtime_entry.get("extra_pressure", 0.0)
        )

        has_velocity_target = bool(runtime_entry.get("has_velocity_target", False))
        velocity_enabled[source_idx] = has_velocity_target
        if has_velocity_target:
            resolved_velocity = resolve_runtime_entry_velocity(
                runtime_entry,
                time_value,
                obstacles_backend,
            )
            velocity_x_values[source_idx] = np.float32(resolved_velocity[0])
            velocity_y_values[source_idx] = np.float32(resolved_velocity[1])
            velocity_z_values[source_idx] = np.float32(resolved_velocity[2])

    return {
        "source_mask": source_mask,
        "entry_masks": entry_masks,
        "temperature_values": temperature_values,
        "smoke_values": smoke_values,
        "fuel_values": fuel_values,
        "extra_pressure_values": extra_pressure_values,
        "velocity_enabled": velocity_enabled,
        "velocity_x_values": velocity_x_values,
        "velocity_y_values": velocity_y_values,
        "velocity_z_values": velocity_z_values,
    }


def rebuild_source_mask(source_data):
    """
    Rebuild the aggregate source mask from all per-source masks.
    """
    aggregate_mask = source_data["mask"]
    aggregate_mask.fill(False)

    any_source = False
    for runtime_entry in source_data.get("runtime_entries", ()):
        entry_mask = runtime_entry.get("mask")
        if entry_mask is None or not np.any(entry_mask):
            continue
        aggregate_mask |= entry_mask
        any_source = True

    source_data["last_has_source"] = bool(any_source)
    return source_data


def build_source_data(domain_cfg, source_entries, obstacles_backend):
    """
    Build persistent source target fields from exported source nodes.

    Multiple source nodes are merged with a max operation per scalar field.
    Velocity targets are written directly so later overlapping sources override
    earlier ones component-wise.
    """
    nx = int(domain_cfg["grid"]["nx"])
    ny = int(domain_cfg["grid"]["ny"])
    nz = int(domain_cfg["grid"]["nz"])
    delta = float(domain_cfg["resolution"])
    origin_x = -0.5 * nx * delta
    origin_y = -0.5 * ny * delta
    origin_z = 0.0

    source_active_mask = np.zeros((nx, ny, nz), dtype=np.bool_)
    runtime_entries = []
    has_animation = False
    has_dynamic_masks = False

    for source_entry in source_entries:
        if source_entry.get("shape") != "mesh":
            continue

        mesh_cfg = source_entry.get("mesh", {})
        mesh_objects = mesh_cfg.get("objects", ())
        if not mesh_objects:
            continue

        velocity = _as_f32_vector(source_entry.get("velocity", (0.0, 0.0, 0.0)))
        velocity_space = str(source_entry.get("velocity_space", "WORLD"))
        source_animations = dict(source_entry.get("animations", {}))
        source_temperature = np.float32(source_entry.get("temperature", 0.0))
        source_smoke = np.float32(source_entry.get("smoke", 0.0))
        source_fuel = np.float32(source_entry.get("fuel", 0.0))
        source_extra_pressure = np.float32(source_entry.get("extra_pressure", 0.0))
        has_velocity_target = bool(np.any(velocity != 0.0))

        for mesh_object in mesh_objects:
            source_runtime = obstacles_backend.build_dynamic_runtime(
                nx,
                ny,
                nz,
                delta,
                [mesh_object],
                origin_x=origin_x,
                origin_y=origin_y,
                origin_z=origin_z,
            )
            source_mask = obstacles_backend.update_dynamic_mask(source_runtime, 0.0)
            runtime_entry = {
                "runtime": source_runtime,
                "mask": source_mask.copy(),
                "temperature": source_temperature,
                "smoke": source_smoke,
                "fuel": source_fuel,
                "extra_pressure": source_extra_pressure,
                "velocity_space": velocity_space,
                "authored_velocity_x": np.float32(velocity[0]),
                "authored_velocity_y": np.float32(velocity[1]),
                "authored_velocity_z": np.float32(velocity[2]),
                "has_velocity_target": has_velocity_target,
                "animations": source_animations.copy(),
            }
            runtime_entries.append(runtime_entry)
            has_animation = has_animation or bool(
                source_runtime.get("is_animated", False) or source_animations
            )
            has_dynamic_masks = has_dynamic_masks or bool(
                source_runtime.get("is_animated", False)
            )

            if not np.any(source_mask):
                continue

            resolved_velocity = resolve_runtime_entry_velocity(
                runtime_entry, 0.0, obstacles_backend
            )
            runtime_entry["initial_velocity_x"] = np.float32(resolved_velocity[0])
            runtime_entry["initial_velocity_y"] = np.float32(resolved_velocity[1])
            runtime_entry["initial_velocity_z"] = np.float32(resolved_velocity[2])
            source_active_mask |= source_mask

    return {
        "mask": source_active_mask,
        "runtime_entries": runtime_entries,
        "is_animated": bool(has_animation),
        "has_dynamic_masks": bool(has_dynamic_masks),
        "last_has_source": bool(np.any(source_active_mask)),
    }
