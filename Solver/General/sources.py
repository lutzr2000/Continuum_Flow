import numpy as np


def _as_f32_vector(values):
    """Return a contiguous float32 vector with exactly three components."""
    vector = np.zeros(3, dtype=np.float32)
    if values is None:
        return vector
    source = np.asarray(values, dtype=np.float32).reshape(-1)
    limit = min(int(source.size), 3)
    if limit > 0:
        vector[:limit] = source[:limit]
    return vector


def _rotation_from_world_matrix(matrix):
    """Extract the closest pure rotation from an affine world matrix."""
    linear = np.asarray(matrix, dtype=np.float32)[:3, :3]
    try:
        u, _, vh = np.linalg.svd(linear.astype(np.float64, copy=False), full_matrices=False)
        rotation = u @ vh
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vh
        return np.asarray(rotation, dtype=np.float32)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float32)


def resolve_local_space_velocity(local_velocity, matrix_world):
    """Convert one local-space velocity vector into world space using object orientation."""
    local_velocity = _as_f32_vector(local_velocity)
    rotation = _rotation_from_world_matrix(matrix_world)
    return np.asarray(rotation @ local_velocity, dtype=np.float32)


def resolve_runtime_entry_velocity(runtime_entry, time_value, obstacles_backend):
    """Resolve the authored source velocity into world space for the current object state."""
    authored_velocity = _as_f32_vector(
        (
            runtime_entry.get("authored_velocity_x", runtime_entry.get("velocity_x", 0.0)),
            runtime_entry.get("authored_velocity_y", runtime_entry.get("velocity_y", 0.0)),
            runtime_entry.get("authored_velocity_z", runtime_entry.get("velocity_z", 0.0)),
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

    temperature_field = np.zeros((nx, ny, nz), dtype=np.float32)
    smoke_field = np.zeros((nx, ny, nz), dtype=np.float32)
    fuel_field = np.zeros((nx, ny, nz), dtype=np.float32)
    extra_pressure_field = np.zeros((nx, ny, nz), dtype=np.float32)
    velocity_x_field = np.zeros((nx, ny, nz), dtype=np.float32)
    velocity_y_field = np.zeros((nx, ny, nz), dtype=np.float32)
    velocity_z_field = np.zeros((nx, ny, nz), dtype=np.float32)
    source_active_mask = np.zeros((nx, ny, nz), dtype=np.bool_)
    velocity_active_mask = np.zeros((nx, ny, nz), dtype=np.bool_)
    runtime_entries = []
    has_animation = False

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

            if not np.any(source_mask):
                continue

            resolved_velocity = resolve_runtime_entry_velocity(runtime_entry, 0.0, obstacles_backend)
            source_active_mask |= source_mask
            temperature_field[source_mask] = np.maximum(
                temperature_field[source_mask],
                source_temperature,
            )
            smoke_field[source_mask] = np.maximum(
                smoke_field[source_mask],
                source_smoke,
            )
            fuel_field[source_mask] = np.maximum(
                fuel_field[source_mask],
                source_fuel,
            )
            extra_pressure_field[source_mask] += source_extra_pressure
            if has_velocity_target:
                velocity_active_mask[source_mask] = True
                velocity_x_field[source_mask] = resolved_velocity[0]
                velocity_y_field[source_mask] = resolved_velocity[1]
                velocity_z_field[source_mask] = resolved_velocity[2]

    return {
        "mask": source_active_mask,
        "velocity_mask": velocity_active_mask,
        "temperature": temperature_field,
        "smoke": smoke_field,
        "fuel": fuel_field,
        "extra_pressure": extra_pressure_field,
        "velocity_x": velocity_x_field,
        "velocity_y": velocity_y_field,
        "velocity_z": velocity_z_field,
        "runtime_entries": runtime_entries,
        "is_animated": bool(has_animation),
    }
