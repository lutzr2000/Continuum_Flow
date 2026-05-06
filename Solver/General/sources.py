import numpy as np


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

        source_runtime = obstacles_backend.build_dynamic_runtime(
            nx, ny, nz, delta, mesh_objects,
            origin_x=origin_x, origin_y=origin_y, origin_z=origin_z,
        )
        has_animation = has_animation or bool(
            source_runtime.get("is_animated", False) or source_entry.get("animations")
        )
        source_mask = obstacles_backend.update_dynamic_mask(source_runtime, 0.0)

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
                "animations": dict(source_entry.get("animations", {})),
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
        "is_animated": bool(has_animation),
    }
