import numpy as np

import Solver.General.helper_functions as helper_functions
import Solver.Kernel_CPU.Boundary_Conditions.obstacle_bc as obstacle_bc
import Solver.Kernel_CPU.Boundary_Conditions.source_bc as source_bc


def rebuild_cpu_boundary_data(simulation_params):
    """Rebuild obstacle/source runtime data so the CPU solver uses CPU-side modules only."""
    domain_cfg = {
        "grid": {
            "nx": simulation_params["NX"],
            "ny": simulation_params["NY"],
            "nz": simulation_params["NZ"],
        },
        "resolution": simulation_params["DELTA"],
    }

    simulation_cfg = (simulation_params.get("_simulation_cfg") or {})
    obstacle_entries = simulation_cfg.get("obstacles", [])
    source_entries = simulation_cfg.get("sources", [])

    obstacle_data = obstacle_bc.build_obstacle_data(domain_cfg, obstacle_entries)
    source_field_data = source_bc.build_source_data(domain_cfg, source_entries)

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


def upload_simulation_state_to_cpu(simulation_params):
    """
    Allocate the persistent CPU solver fields and work buffers.

    The setup mirrors the GPU state layout closely so the solver loop can stay
    structurally similar between CPU and GPU backends.
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
    Fx_base = np.asarray(force_field_data["Fx_base"], dtype=precision_dtype)
    Fy_base = np.asarray(force_field_data["Fy_base"], dtype=precision_dtype)
    Fz_base = np.asarray(force_field_data["Fz_base"], dtype=precision_dtype)
    Fx = np.asarray(force_field_data["Fx"], dtype=precision_dtype)
    Fy = np.asarray(force_field_data["Fy"], dtype=precision_dtype)
    Fz = np.asarray(force_field_data["Fz"], dtype=precision_dtype)
    point_divergence = np.asarray(force_field_data["point_divergence"], dtype=precision_dtype)
    turbulence_data = force_field_data["turbulence"]

    obstacle_data = simulation_params["obstacle_data"]
    obstacle_mask_host = np.asarray(obstacle_data["mask"])
    obstacle_velocity_x_host = np.asarray(obstacle_data["velocity_x"], dtype=precision_dtype)
    obstacle_velocity_y_host = np.asarray(obstacle_data["velocity_y"], dtype=precision_dtype)
    obstacle_velocity_z_host = np.asarray(obstacle_data["velocity_z"], dtype=precision_dtype)

    source_field_data = simulation_params["source_field_data"]
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

    cpu_fields = {
        "u": u,
        "v": v,
        "w": w,
        "u_work": np.empty((nx, ny, nz), dtype=precision_dtype),
        "v_work": np.empty((nx, ny, nz), dtype=precision_dtype),
        "w_work": np.empty((nx, ny, nz), dtype=precision_dtype),
        "p": p,
        "pressure_rhs": np.empty((nx, ny, nz), dtype=precision_dtype),
        "T": T,
        "temperature_work": np.empty((nx, ny, nz), dtype=precision_dtype),
        "smoke": smoke,
        "smoke_work": np.empty((nx, ny, nz), dtype=precision_dtype),
        "fuel": fuel,
        "fuel_work": np.empty((nx, ny, nz), dtype=precision_dtype),
        "flame": flame,
        "flame_work": np.empty((nx, ny, nz), dtype=precision_dtype),
        "vorticity_x": np.empty((nx, ny, nz), dtype=precision_dtype),
        "vorticity_y": np.empty((nx, ny, nz), dtype=precision_dtype),
        "vorticity_z": np.empty((nx, ny, nz), dtype=precision_dtype),
        "vorticity_magnitude": np.empty((nx, ny, nz), dtype=precision_dtype),
        "Fx": Fx.copy(),
        "Fy": Fy.copy(),
        "Fz": Fz.copy(),
        "point_divergence": point_divergence,
        "Fx_base": Fx_base,
        "Fy_base": Fy_base,
        "Fz_base": Fz_base,
        "turbulence_Fx_a": np.asarray(turbulence_data["Fx_a"], dtype=precision_dtype),
        "turbulence_Fy_a": np.asarray(turbulence_data["Fy_a"], dtype=precision_dtype),
        "turbulence_Fz_a": np.asarray(turbulence_data["Fz_a"], dtype=precision_dtype),
        "turbulence_Fx_b": np.asarray(turbulence_data["Fx_b"], dtype=precision_dtype),
        "turbulence_Fy_b": np.asarray(turbulence_data["Fy_b"], dtype=precision_dtype),
        "turbulence_Fz_b": np.asarray(turbulence_data["Fz_b"], dtype=precision_dtype),
        "turbulence_cos_coeffs": np.ones(len(turbulence_data["angular_frequencies"]), dtype=precision_dtype),
        "turbulence_sin_coeffs": np.zeros(len(turbulence_data["angular_frequencies"]), dtype=precision_dtype),
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
        "velocity_maxima": np.zeros(3, dtype=np.float32),
    }

    cpu_constants = {
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

    return cpu_fields, cpu_constants


def update_animated_cpu_constants(simulation_params, cpu_constants, time_value):
    """Update animated scalar solver constants for the current simulation time."""
    animation_state = simulation_params.get("ANIMATION_STATE")
    if not animation_state or not animation_state.get("enabled", False):
        return

    for constant_name, series in animation_state.get("constants", {}).items():
        cpu_constants[constant_name] = np.float32(
            helper_functions._interpolate_animation_series(series, time_value)
        )


def _cached_animation_series(container, property_name, dtype):
    """Return one cached runtime animation series from an exported animation entry."""
    cache = container.setdefault("_animation_series_cache", {})
    if property_name in cache:
        return cache[property_name]

    series = helper_functions._animation_series_to_arrays(
        (container.get("animations") or {}).get(property_name),
        dtype,
    )
    cache[property_name] = series
    return series


def update_animated_source_force_values(simulation_params, cpu_fields, time_value):
    """Update animated source targets and animated constant-force values."""
    source_field_data = simulation_params.get("source_field_data")
    if source_field_data is not None:
        source_changed = False
        for runtime_entry in source_field_data.get("runtime_entries", ()):
            if "_base_temperature" not in runtime_entry:
                runtime_entry["_base_temperature"] = np.float32(runtime_entry.get("temperature", 0.0))
                runtime_entry["_base_smoke"] = np.float32(runtime_entry.get("smoke", 0.0))
                runtime_entry["_base_fuel"] = np.float32(runtime_entry.get("fuel", 0.0))
                runtime_entry["_base_velocity_x"] = np.float32(runtime_entry.get("velocity_x", 0.0))
                runtime_entry["_base_velocity_y"] = np.float32(runtime_entry.get("velocity_y", 0.0))
                runtime_entry["_base_velocity_z"] = np.float32(runtime_entry.get("velocity_z", 0.0))

            next_temperature = runtime_entry["_base_temperature"]
            next_smoke = runtime_entry["_base_smoke"]
            next_fuel = runtime_entry["_base_fuel"]
            next_velocity_x = runtime_entry["_base_velocity_x"]
            next_velocity_y = runtime_entry["_base_velocity_y"]
            next_velocity_z = runtime_entry["_base_velocity_z"]

            temperature_series = _cached_animation_series(runtime_entry, "temperature", np.float32)
            if temperature_series is not None:
                next_temperature = np.float32(
                    helper_functions._interpolate_animation_series(temperature_series, time_value)
                )

            smoke_series = _cached_animation_series(runtime_entry, "smoke", np.float32)
            if smoke_series is not None:
                next_smoke = np.float32(
                    helper_functions._interpolate_animation_series(smoke_series, time_value)
                )

            fuel_series = _cached_animation_series(runtime_entry, "fuel", np.float32)
            if fuel_series is not None:
                next_fuel = np.float32(
                    helper_functions._interpolate_animation_series(fuel_series, time_value)
                )

            velocity_series = _cached_animation_series(runtime_entry, "velocity", np.float32)
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
                next_velocity_x != runtime_entry.get("velocity_x") or
                next_velocity_y != runtime_entry.get("velocity_y") or
                next_velocity_z != runtime_entry.get("velocity_z")
            ):
                runtime_entry["temperature"] = next_temperature
                runtime_entry["smoke"] = next_smoke
                runtime_entry["fuel"] = next_fuel
                runtime_entry["velocity_x"] = next_velocity_x
                runtime_entry["velocity_y"] = next_velocity_y
                runtime_entry["velocity_z"] = next_velocity_z
                runtime_entry["has_velocity_target"] = bool(
                    next_velocity_x != 0.0 or next_velocity_y != 0.0 or next_velocity_z != 0.0
                )
                source_changed = True

        if source_changed or source_field_data.get("is_animated", False):
            source_bc.update_source_data(source_field_data, time_value)
            cpu_fields["source_mask"] = source_field_data["mask"]
            cpu_fields["source_velocity_mask"] = source_field_data["velocity_mask"]
            cpu_fields["source_temperature"] = source_field_data["temperature"]
            cpu_fields["source_smoke"] = source_field_data["smoke"]
            cpu_fields["source_fuel"] = source_field_data["fuel"]
            cpu_fields["source_velocity_x"] = source_field_data["velocity_x"]
            cpu_fields["source_velocity_y"] = source_field_data["velocity_y"]
            cpu_fields["source_velocity_z"] = source_field_data["velocity_z"]

    animated_force = {"x": 0.0, "y": 0.0, "z": 0.0}
    animation_state = simulation_params.get("ANIMATION_STATE") or {}
    for axis_name, series in animation_state.get("constant_force", {}).items():
        animated_force[axis_name] = float(
            helper_functions._interpolate_animation_series(series, time_value)
        )

    return animated_force


def update_dynamic_boundary_data_on_cpu(simulation_params, cpu_fields, cpu_constants, time_value):
    """Update animated obstacle/source masks and fields on the CPU."""
    if not simulation_params.get("HAS_DYNAMIC_BOUNDARIES", False):
        return

    obstacle_data = simulation_params.get("obstacle_data")
    if obstacle_data is not None and obstacle_data.get("runtime") is not None and obstacle_data.get("is_animated", False):
        obstacle_bc.update_obstacle_mask(obstacle_data, time_value)
        cpu_fields["obstacle_mask"] = obstacle_data["mask"]
        cpu_fields["obstacle_velocity_x"] = obstacle_data["velocity_x"]
        cpu_fields["obstacle_velocity_y"] = obstacle_data["velocity_y"]
        cpu_fields["obstacle_velocity_z"] = obstacle_data["velocity_z"]
        cpu_constants["HAS_OBSTACLE"] = bool(np.any(obstacle_data["mask"]))

    source_field_data = simulation_params.get("source_field_data")
    if source_field_data is not None and source_field_data.get("is_animated", False):
        source_bc.update_source_data(source_field_data, time_value)
        cpu_fields["source_mask"] = source_field_data["mask"]
        cpu_fields["source_velocity_mask"] = source_field_data["velocity_mask"]
        cpu_fields["source_temperature"] = source_field_data["temperature"]
        cpu_fields["source_smoke"] = source_field_data["smoke"]
        cpu_fields["source_fuel"] = source_field_data["fuel"]
        cpu_fields["source_velocity_x"] = source_field_data["velocity_x"]
        cpu_fields["source_velocity_y"] = source_field_data["velocity_y"]
        cpu_fields["source_velocity_z"] = source_field_data["velocity_z"]
        cpu_constants["HAS_SOURCE"] = bool(np.any(source_field_data["mask"]))
