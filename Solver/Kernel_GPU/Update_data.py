import numpy as np
from numba import cuda

import Solver.General.Helper_Functions as Helper_Functions
import Solver.Kernel_GPU.Boundary_Conditions.obstacles as Obstacles
import Solver.Kernel_GPU.Boundary_Conditions.Source_BC as source_bc


def upload_simulation_state_to_gpu(simulation_params):
    """
    Allocate the simulation fields on the host and upload persistent arrays to the GPU.
    """
    # Resolve the shared grid shape and numeric precision once up front.
    precision_dtype = np.dtype(simulation_params["PRECISION"])
    nx = simulation_params["NX"]
    ny = simulation_params["NY"]
    nz = simulation_params["NZ"]

    # Allocate the base host-side simulation fields from the solver config.
    u = np.full((nx, ny, nz), simulation_params["INITIAL_U"], dtype=precision_dtype)
    v = np.full((nx, ny, nz), simulation_params["INITIAL_V"], dtype=precision_dtype)
    w = np.full((nx, ny, nz), simulation_params["INITIAL_W"], dtype=precision_dtype)

    p = np.zeros((nx, ny, nz), dtype=precision_dtype)
    T = np.full((nx, ny, nz), simulation_params["T_REFERENCE"], dtype=precision_dtype)
    smoke = np.zeros((nx, ny, nz), dtype=precision_dtype)
    fuel = np.zeros((nx, ny, nz), dtype=precision_dtype)
    flame = np.zeros((nx, ny, nz), dtype=precision_dtype)

    # Prepare persistent force-field inputs that are reused across time steps.
    force_field_data = simulation_params["force_field_data"]
    Fx_base = np.asarray(force_field_data["Fx_base"], dtype=precision_dtype)
    Fy_base = np.asarray(force_field_data["Fy_base"], dtype=precision_dtype)
    Fz_base = np.asarray(force_field_data["Fz_base"], dtype=precision_dtype)
    Fx = np.asarray(force_field_data["Fx"], dtype=precision_dtype)
    Fy = np.asarray(force_field_data["Fy"], dtype=precision_dtype)
    Fz = np.asarray(force_field_data["Fz"], dtype=precision_dtype)
    point_divergence = np.asarray(force_field_data["point_divergence"], dtype=precision_dtype)
    turbulence_data = force_field_data["turbulence"]

    # Prepare obstacle runtime data and the host arrays mirrored on the GPU.
    obstacle_data = simulation_params.get("obstacle_data", {"mask": simulation_params["obstacle_mask"]})
    if obstacle_data.get("runtime") is not None and obstacle_data.get("is_animated", False):
        Obstacles.prepare_dynamic_runtime_for_gpu(obstacle_data["runtime"])
    obstacle_mask_host = np.asarray(obstacle_data["mask"])
    obstacle_velocity_x_host = np.asarray(obstacle_data["velocity_x"], dtype=precision_dtype)
    obstacle_velocity_y_host = np.asarray(obstacle_data["velocity_y"], dtype=precision_dtype)
    obstacle_velocity_z_host = np.asarray(obstacle_data["velocity_z"], dtype=precision_dtype)

    # Prepare source runtime data and the host arrays mirrored on the GPU.
    source_field_data = simulation_params["source_field_data"]
    if source_field_data.get("is_animated", False):
        source_bc.prepare_source_data_for_gpu(source_field_data)
    source_mask_host = np.asarray(source_field_data["mask"])
    source_velocity_mask_host = np.asarray(source_field_data["velocity_mask"])
    source_temperature_host = np.asarray(source_field_data["temperature"], dtype=precision_dtype)
    source_smoke_host = np.asarray(source_field_data["smoke"], dtype=precision_dtype)
    source_fuel_host = np.asarray(source_field_data["fuel"], dtype=precision_dtype)
    source_velocity_x_host = np.asarray(source_field_data["velocity_x"], dtype=precision_dtype)
    source_velocity_y_host = np.asarray(source_field_data["velocity_y"], dtype=precision_dtype)
    source_velocity_z_host = np.asarray(source_field_data["velocity_z"], dtype=precision_dtype)

    # Apply source values to the initial host state before the first GPU upload.
    u[source_velocity_mask_host] = source_velocity_x_host[source_velocity_mask_host]
    v[source_velocity_mask_host] = source_velocity_y_host[source_velocity_mask_host]
    w[source_velocity_mask_host] = source_velocity_z_host[source_velocity_mask_host]

    T = np.maximum(T, source_temperature_host)
    smoke = np.maximum(smoke, source_smoke_host)
    fuel = np.maximum(fuel, source_fuel_host)

    # Upload persistent fields and allocate all work buffers on the device.
    gpu_fields = {
        "u": cuda.to_device(u),
        "v": cuda.to_device(v),
        "w": cuda.to_device(w),
        "u_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "v_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "w_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "p": cuda.to_device(p),
        "pressure_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "pressure_rhs": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "T": cuda.to_device(T),
        "temperature_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "smoke": cuda.to_device(smoke),
        "smoke_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "fuel": cuda.to_device(fuel),
        "fuel_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "flame": cuda.to_device(flame),
        "flame_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "vorticity_x": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "vorticity_y": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "vorticity_z": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "vorticity_magnitude": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "Fx": cuda.to_device(Fx),
        "Fy": cuda.to_device(Fy),
        "Fz": cuda.to_device(Fz),
        "point_divergence": cuda.to_device(point_divergence),
        "Fx_base": cuda.to_device(Fx_base),
        "Fy_base": cuda.to_device(Fy_base),
        "Fz_base": cuda.to_device(Fz_base),
        "turbulence_Fx_a": cuda.to_device(np.asarray(turbulence_data["Fx_a"], dtype=precision_dtype)),
        "turbulence_Fy_a": cuda.to_device(np.asarray(turbulence_data["Fy_a"], dtype=precision_dtype)),
        "turbulence_Fz_a": cuda.to_device(np.asarray(turbulence_data["Fz_a"], dtype=precision_dtype)),
        "turbulence_Fx_b": cuda.to_device(np.asarray(turbulence_data["Fx_b"], dtype=precision_dtype)),
        "turbulence_Fy_b": cuda.to_device(np.asarray(turbulence_data["Fy_b"], dtype=precision_dtype)),
        "turbulence_Fz_b": cuda.to_device(np.asarray(turbulence_data["Fz_b"], dtype=precision_dtype)),
        "turbulence_cos_coeffs": cuda.to_device(np.ones(len(turbulence_data["angular_frequencies"]), dtype=precision_dtype)),
        "turbulence_sin_coeffs": cuda.to_device(np.zeros(len(turbulence_data["angular_frequencies"]), dtype=precision_dtype)),
        "obstacle_mask": cuda.to_device(obstacle_mask_host),
        "obstacle_velocity_x": cuda.to_device(obstacle_velocity_x_host),
        "obstacle_velocity_y": cuda.to_device(obstacle_velocity_y_host),
        "obstacle_velocity_z": cuda.to_device(obstacle_velocity_z_host),
        "source_mask": cuda.to_device(source_mask_host),
        "source_velocity_mask": cuda.to_device(source_velocity_mask_host),
        "source_temperature": cuda.to_device(source_temperature_host),
        "source_smoke": cuda.to_device(source_smoke_host),
        "source_fuel": cuda.to_device(source_fuel_host),
        "source_velocity_x": cuda.to_device(source_velocity_x_host),
        "source_velocity_y": cuda.to_device(source_velocity_y_host),
        "source_velocity_z": cuda.to_device(source_velocity_z_host),
        "velocity_maxima": cuda.device_array(3, dtype=np.float32),
    }

    # Pack scalar constants separately so kernels can access a compact settings block.
    gpu_constants = {
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

    return gpu_fields, gpu_constants


def update_animated_gpu_constants(simulation_params, gpu_constants, time_value):
    """Update animated scalar solver constants for the current simulation time."""
    animation_state = simulation_params.get("ANIMATION_STATE")
    if not animation_state:
        return
    if not animation_state.get("enabled", False):
        return

    for constant_name, series in animation_state.get("constants", {}).items():
        gpu_constants[constant_name] = np.float32(
            Helper_Functions._interpolate_animation_series(series, time_value)
        )


def _cached_animation_series(container, property_name, dtype):
    """Return one cached runtime animation series from an exported animation entry."""
    cache = container.setdefault("_animation_series_cache", {})
    if property_name in cache:
        return cache[property_name]

    series = Helper_Functions._animation_series_to_arrays(
        (container.get("animations") or {}).get(property_name),
        dtype,
    )
    cache[property_name] = series
    return series


def update_animated_source_force_values(simulation_params, gpu_fields, time_value):
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
                    Helper_Functions._interpolate_animation_series(temperature_series, time_value)
                )

            smoke_series = _cached_animation_series(runtime_entry, "smoke", np.float32)
            if smoke_series is not None:
                next_smoke = np.float32(
                    Helper_Functions._interpolate_animation_series(smoke_series, time_value)
                )

            fuel_series = _cached_animation_series(runtime_entry, "fuel", np.float32)
            if fuel_series is not None:
                next_fuel = np.float32(
                    Helper_Functions._interpolate_animation_series(fuel_series, time_value)
                )

            velocity_series = _cached_animation_series(runtime_entry, "velocity", np.float32)
            if velocity_series is not None:
                velocity_value = np.asarray(
                    Helper_Functions._interpolate_animation_series(velocity_series, time_value),
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
            source_bc.update_source_data_gpu(source_field_data, gpu_fields, time_value)

    animated_force = {"x": 0.0, "y": 0.0, "z": 0.0}
    animation_state = simulation_params.get("ANIMATION_STATE") or {}
    for axis_name, series in animation_state.get("constant_force", {}).items():
        animated_force[axis_name] = float(
            Helper_Functions._interpolate_animation_series(series, time_value)
        )

    return animated_force


def update_dynamic_boundary_data_on_gpu(simulation_params, gpu_fields, gpu_constants, time_value):
    """Update animated obstacle/source masks on the host and upload them to the GPU."""
    if not simulation_params.get("HAS_DYNAMIC_BOUNDARIES", False):
        return

    obstacle_data = simulation_params.get("obstacle_data")
    if obstacle_data is not None and obstacle_data.get("runtime") is not None and obstacle_data.get("is_animated", False):
        Obstacles.update_dynamic_obstacle_data_gpu(
            obstacle_data["runtime"],
            time_value,
            gpu_fields["obstacle_mask"],
            gpu_fields["obstacle_velocity_x"],
            gpu_fields["obstacle_velocity_y"],
            gpu_fields["obstacle_velocity_z"],
        )
        gpu_constants["HAS_OBSTACLE"] = bool(obstacle_data["runtime"].get("last_has_obstacle", False))

    source_field_data = simulation_params.get("source_field_data")
    if source_field_data is not None and source_field_data.get("is_animated", False):
        gpu_constants["HAS_SOURCE"] = bool(source_field_data.get("last_has_source", False))
