import numpy as np
from numba import cuda

import Solver.Kernel_GPU.Boundary_Conditions.obstacles as Obstacles
import Solver.Kernel_GPU.Boundary_Conditions.source_bc as source_bc


def upload_simulation_state_to_gpu(simulation_params):
    """
    Allocate the simulation fields on the host and upload persistent arrays to the GPU.
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

    obstacle_data = simulation_params.get("obstacle_data", {"mask": simulation_params["obstacle_mask"]})
    if obstacle_data.get("runtime") is not None and obstacle_data.get("is_animated", False):
        Obstacles.prepare_dynamic_runtime_for_gpu(obstacle_data["runtime"])
    obstacle_mask_host = np.asarray(obstacle_data["mask"])
    obstacle_velocity_x_host = np.asarray(obstacle_data["velocity_x"], dtype=precision_dtype)
    obstacle_velocity_y_host = np.asarray(obstacle_data["velocity_y"], dtype=precision_dtype)
    obstacle_velocity_z_host = np.asarray(obstacle_data["velocity_z"], dtype=precision_dtype)
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

    u[source_velocity_mask_host] = source_velocity_x_host[source_velocity_mask_host]
    v[source_velocity_mask_host] = source_velocity_y_host[source_velocity_mask_host]
    w[source_velocity_mask_host] = source_velocity_z_host[source_velocity_mask_host]

    T = np.maximum(T, source_temperature_host)
    smoke = np.maximum(smoke, source_smoke_host)
    fuel = np.maximum(fuel, source_fuel_host)

    device_state = {
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

    return device_state, gpu_constants


def update_dynamic_boundary_data_on_gpu(simulation_params, device_state, gpu_constants, time_value):
    """Update animated obstacle/source masks on the host and upload them to the GPU."""
    if not simulation_params.get("HAS_DYNAMIC_BOUNDARIES", False):
        return

    obstacle_data = simulation_params.get("obstacle_data")
    if obstacle_data is not None and obstacle_data.get("runtime") is not None and obstacle_data.get("is_animated", False):
        Obstacles.update_dynamic_obstacle_data_gpu(
            obstacle_data["runtime"],
            time_value,
            device_state["obstacle_mask"],
            device_state["obstacle_velocity_x"],
            device_state["obstacle_velocity_y"],
            device_state["obstacle_velocity_z"],
        )
        gpu_constants["HAS_OBSTACLE"] = bool(obstacle_data["runtime"].get("last_has_obstacle", False))

    source_field_data = simulation_params.get("source_field_data")
    if source_field_data is not None and source_field_data.get("is_animated", False):
        source_bc.update_source_data_gpu(source_field_data, device_state, time_value)
        gpu_constants["HAS_SOURCE"] = bool(source_field_data.get("last_has_source", False))
