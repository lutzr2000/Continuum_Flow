import numpy as np

import Solver.General.helper_functions as helper_functions


def rebuild_boundary_data(simulation_params, obstacle_builder, source_builder):
    """Rebuild obstacle/source runtime data through backend-specific builders."""
    domain_cfg = {
        "grid": {
            "nx": simulation_params["NX"],
            "ny": simulation_params["NY"],
            "nz": simulation_params["NZ"],
        },
        "resolution": simulation_params["DELTA"],
    }

    simulation_cfg = simulation_params.get("_simulation_cfg") or {}
    obstacle_entries = simulation_cfg.get("obstacles", [])
    source_entries = simulation_cfg.get("sources", [])

    obstacle_data = obstacle_builder(domain_cfg, obstacle_entries)
    source_field_data = source_builder(domain_cfg, source_entries)

    simulation_params["obstacle_data"] = obstacle_data
    simulation_params["obstacle_mask"] = obstacle_data["mask"]
    simulation_params["source_field_data"] = source_field_data
    simulation_params["HAS_OBSTACLE"] = bool(np.any(obstacle_data["mask"]))
    simulation_params["HAS_SOURCE"] = bool(np.any(source_field_data["mask"]))
    simulation_params["HAS_DYNAMIC_BOUNDARIES"] = bool(
        obstacle_data.get("is_animated", False) or
        source_field_data.get("is_animated", False)
    )
    return simulation_params


def build_initial_host_state(
    simulation_params,
    obstacle_prepare=None,
    source_prepare=None,
):
    """
    Build the shared host-side simulation state before CPU/GPU backend upload.

    Optional prepare callbacks can mutate backend-specific obstacle/source
    runtime data before the arrays are mirrored into solver fields.
    """
    precision_dtype = np.dtype(simulation_params["PRECISION"])
    nx = simulation_params["NX"]
    ny = simulation_params["NY"]
    nz = simulation_params["NZ"]

    u = np.full((nx, ny, nz), simulation_params["INITIAL_U"], dtype=precision_dtype)
    v = np.full((nx, ny, nz), simulation_params["INITIAL_V"], dtype=precision_dtype)
    w = np.full((nx, ny, nz), simulation_params["INITIAL_W"], dtype=precision_dtype)

    p = np.zeros((nx, ny, nz), dtype=precision_dtype)
    T = np.full((nx, ny, nz), simulation_params["T_REFERENCE"], dtype=precision_dtype)
    smoke = np.zeros((nx, ny, nz), dtype=precision_dtype)
    fuel = np.zeros((nx, ny, nz), dtype=precision_dtype)
    flame = np.zeros((nx, ny, nz), dtype=precision_dtype)

    force_field_data = simulation_params["force_field_data"]
    turbulence_data = force_field_data["turbulence"]

    obstacle_data = simulation_params.get("obstacle_data", {"mask": simulation_params["obstacle_mask"]})
    if obstacle_prepare is not None:
        obstacle_prepare(obstacle_data)

    obstacle_mask_host = np.asarray(obstacle_data["mask"])
    obstacle_velocity_x_host = np.asarray(obstacle_data["velocity_x"], dtype=precision_dtype)
    obstacle_velocity_y_host = np.asarray(obstacle_data["velocity_y"], dtype=precision_dtype)
    obstacle_velocity_z_host = np.asarray(obstacle_data["velocity_z"], dtype=precision_dtype)

    source_field_data = simulation_params["source_field_data"]
    if source_prepare is not None:
        source_prepare(source_field_data)

    source_mask_host = np.asarray(source_field_data["mask"])
    source_velocity_mask_host = np.asarray(source_field_data["velocity_mask"])
    source_temperature_host = np.asarray(source_field_data["temperature"], dtype=precision_dtype)
    source_smoke_host = np.asarray(source_field_data["smoke"], dtype=precision_dtype)
    source_fuel_host = np.asarray(source_field_data["fuel"], dtype=precision_dtype)
    source_velocity_x_host = np.asarray(source_field_data["velocity_x"], dtype=precision_dtype)
    source_velocity_y_host = np.asarray(source_field_data["velocity_y"], dtype=precision_dtype)
    source_velocity_z_host = np.asarray(source_field_data["velocity_z"], dtype=precision_dtype)

    u[source_velocity_mask_host] = source_velocity_x_host[source_velocity_mask_host]
    v[source_velocity_mask_host] = source_velocity_y_host[source_velocity_mask_host]
    w[source_velocity_mask_host] = source_velocity_z_host[source_velocity_mask_host]

    T = np.maximum(T, source_temperature_host)
    smoke = np.maximum(smoke, source_smoke_host)
    fuel = np.maximum(fuel, source_fuel_host)

    return {
        "precision_dtype": precision_dtype,
        "shape": (nx, ny, nz),
        "u": u,
        "v": v,
        "w": w,
        "p": p,
        "T": T,
        "smoke": smoke,
        "fuel": fuel,
        "flame": flame,
        "force_field_data": force_field_data,
        "turbulence_data": turbulence_data,
        "obstacle_mask": obstacle_mask_host,
        "obstacle_velocity_x": obstacle_velocity_x_host,
        "obstacle_velocity_y": obstacle_velocity_y_host,
        "obstacle_velocity_z": obstacle_velocity_z_host,
        "source_mask": source_mask_host,
        "source_velocity_mask": source_velocity_mask_host,
        "source_temperature": source_temperature_host,
        "source_smoke": source_smoke_host,
        "source_fuel": source_fuel_host,
        "source_velocity_x": source_velocity_x_host,
        "source_velocity_y": source_velocity_y_host,
        "source_velocity_z": source_velocity_z_host,
    }


def build_solver_constants(simulation_params, precision_dtype, force_field_data):
    """Build the backend-agnostic scalar solver constants block."""
    return {
        "RHO": precision_dtype.type(simulation_params["RHO"]),
        "NU": precision_dtype.type(simulation_params["NU"]),
        "TEMPERATURE_DISSIPATION_RATE": precision_dtype.type(simulation_params["TEMPERATURE_DISSIPATION_RATE"]),
        "TEMPERATURE_PRODUCTION_RATE": precision_dtype.type(simulation_params["TEMPERATURE_PRODUCTION_RATE"]),
        "SMOKE_DISSIPATION_RATE": precision_dtype.type(simulation_params["SMOKE_DISSIPATION_RATE"]),
        "SMOKE_PRODUCTION_RATE": precision_dtype.type(simulation_params["SMOKE_PRODUCTION_RATE"]),
        "FUEL_BURN_RATE": precision_dtype.type(simulation_params["FUEL_BURN_RATE"]),
        "FUEL_IGNITION_TEMPERATURE": precision_dtype.type(simulation_params["FUEL_IGNITION_TEMPERATURE"]),
        "T_REFERENCE": precision_dtype.type(simulation_params["T_REFERENCE"]),
        "SOURCE_TEMPERATURE_MAX": precision_dtype.type(simulation_params["SOURCE_TEMPERATURE_MAX"]),
        "BUOANCY_FACTOR": precision_dtype.type(simulation_params["BUOANCY_FACTOR"]),
        "EXPANSION_RATE": precision_dtype.type(simulation_params["EXPANSION_RATE"]),
        "VORTICITY": precision_dtype.type(simulation_params["VORTICITY"]),
        "DELTA": precision_dtype.type(simulation_params["DELTA"]),
        "NX": simulation_params["NX"],
        "NY": simulation_params["NY"],
        "NZ": simulation_params["NZ"],
        "FORCE_X_MAX": precision_dtype.type(force_field_data["max_abs"][0]),
        "FORCE_Y_MAX": precision_dtype.type(force_field_data["max_abs"][1]),
        "FORCE_Z_MAX": precision_dtype.type(force_field_data["max_abs"][2]),
        "HAS_SOURCE": simulation_params["HAS_SOURCE"],
        "HAS_OBSTACLE": simulation_params["HAS_OBSTACLE"],
        "HAS_FORCE": simulation_params["HAS_FORCE"],
    }


def update_animated_constants(simulation_params, constants, time_value):
    """Update animated scalar solver constants for the current simulation time."""
    animation_state = simulation_params.get("ANIMATION_STATE")
    if not animation_state or not animation_state.get("enabled", False):
        return

    for constant_name, series in animation_state.get("constants", {}).items():
        constants[constant_name] = np.float32(
            helper_functions._interpolate_animation_series(series, time_value)
        )


def update_animated_source_force_values(
    simulation_params,
    solver_fields,
    time_value,
    source_updater,
    source_sync=None,
):
    """Update animated source targets and animated constant-force values."""
    source_field_data = simulation_params.get("source_field_data")
    if source_field_data is not None:
        source_changed = False
        for runtime_entry in source_field_data.get("runtime_entries", ()):
            if "_base_temperature" not in runtime_entry:
                runtime_entry["_base_temperature"] = np.float32(runtime_entry.get("temperature", 0.0))
                runtime_entry["_base_smoke"] = np.float32(runtime_entry.get("smoke", 0.0))
                runtime_entry["_base_fuel"] = np.float32(runtime_entry.get("fuel", 0.0))
                runtime_entry["_base_velocity_x"] = np.float32(runtime_entry.get("authored_velocity_x", runtime_entry.get("velocity_x", 0.0)))
                runtime_entry["_base_velocity_y"] = np.float32(runtime_entry.get("authored_velocity_y", runtime_entry.get("velocity_y", 0.0)))
                runtime_entry["_base_velocity_z"] = np.float32(runtime_entry.get("authored_velocity_z", runtime_entry.get("velocity_z", 0.0)))

            next_temperature = runtime_entry["_base_temperature"]
            next_smoke = runtime_entry["_base_smoke"]
            next_fuel = runtime_entry["_base_fuel"]
            next_velocity_x = runtime_entry["_base_velocity_x"]
            next_velocity_y = runtime_entry["_base_velocity_y"]
            next_velocity_z = runtime_entry["_base_velocity_z"]

            temperature_series = helper_functions._cached_animation_series(runtime_entry, "temperature", np.float32)
            if temperature_series is not None:
                next_temperature = np.float32(
                    helper_functions._interpolate_animation_series(temperature_series, time_value)
                )

            smoke_series = helper_functions._cached_animation_series(runtime_entry, "smoke", np.float32)
            if smoke_series is not None:
                next_smoke = np.float32(
                    helper_functions._interpolate_animation_series(smoke_series, time_value)
                )

            fuel_series = helper_functions._cached_animation_series(runtime_entry, "fuel", np.float32)
            if fuel_series is not None:
                next_fuel = np.float32(
                    helper_functions._interpolate_animation_series(fuel_series, time_value)
                )

            velocity_series = helper_functions._cached_animation_series(runtime_entry, "velocity", np.float32)
            if velocity_series is not None:
                velocity_value = np.asarray(
                    helper_functions._interpolate_animation_series(velocity_series, time_value),
                    dtype=np.float32,
                ).reshape(-1)
                if velocity_value.size > 0:
                    next_velocity_x = np.float32(velocity_value[0])
                if velocity_value.size > 1:
                    next_velocity_y = np.float32(velocity_value[1])
                if velocity_value.size > 2:
                    next_velocity_z = np.float32(velocity_value[2])

            if (
                next_temperature != runtime_entry.get("temperature") or
                next_smoke != runtime_entry.get("smoke") or
                next_fuel != runtime_entry.get("fuel") or
                next_velocity_x != runtime_entry.get("authored_velocity_x", runtime_entry.get("velocity_x")) or
                next_velocity_y != runtime_entry.get("authored_velocity_y", runtime_entry.get("velocity_y")) or
                next_velocity_z != runtime_entry.get("authored_velocity_z", runtime_entry.get("velocity_z"))
            ):
                runtime_entry["temperature"] = next_temperature
                runtime_entry["smoke"] = next_smoke
                runtime_entry["fuel"] = next_fuel
                runtime_entry["authored_velocity_x"] = next_velocity_x
                runtime_entry["authored_velocity_y"] = next_velocity_y
                runtime_entry["authored_velocity_z"] = next_velocity_z
                runtime_entry["has_velocity_target"] = bool(
                    next_velocity_x != 0.0 or next_velocity_y != 0.0 or next_velocity_z != 0.0
                )
                source_changed = True

        if source_changed or source_field_data.get("is_animated", False):
            source_updater(source_field_data, solver_fields, time_value)
            if source_sync is not None:
                source_sync(source_field_data, solver_fields)

    animated_force = {"x": 0.0, "y": 0.0, "z": 0.0}
    animation_state = simulation_params.get("ANIMATION_STATE") or {}
    for axis_name, series in animation_state.get("constant_force", {}).items():
        animated_force[axis_name] = float(
            helper_functions._interpolate_animation_series(series, time_value)
        )

    return animated_force
