import numpy as np
from numba import cuda

import Solver.General.update_data as general_update_data
import Solver.Kernel_GPU.Boundary_Conditions.obstacles as obstacles
import Solver.Kernel_GPU.Boundary_Conditions.source_bc as source_bc


def _prepare_gpu_obstacle_data(obstacle_data):
    if obstacle_data.get("runtime") is not None and obstacle_data.get("is_animated", False):
        obstacles.prepare_dynamic_runtime_for_gpu(obstacle_data["runtime"])


def _prepare_gpu_source_data(source_field_data):
    if source_field_data.get("is_animated", False):
        source_bc.prepare_source_data_for_gpu(source_field_data)


def upload_simulation_state_to_gpu(simulation_params):
    """
    Allocate the simulation fields on the host and upload persistent arrays to the GPU.
    """
    host_state = general_update_data.build_initial_host_state(
        simulation_params,
        obstacle_prepare=_prepare_gpu_obstacle_data,
        source_prepare=_prepare_gpu_source_data,
    )
    precision_dtype = host_state["precision_dtype"]
    nx, ny, nz = host_state["shape"]
    force_field_data = host_state["force_field_data"]
    turbulence_data = host_state["turbulence_data"]

    gpu_fields = {
        # Primary velocity state and scratch buffers.
        "u": cuda.to_device(host_state["u"]),
        "v": cuda.to_device(host_state["v"]),
        "w": cuda.to_device(host_state["w"]),
        "u_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "v_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "w_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),

        # Pressure solve state.
        "p": cuda.to_device(host_state["p"]),
        "pressure_rhs": cuda.device_array((nx, ny, nz), dtype=precision_dtype),

        # Advected scalar fields and their work buffers.
        "T": cuda.to_device(host_state["T"]),
        "temperature_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "smoke": cuda.to_device(host_state["smoke"]),
        "smoke_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "fuel": cuda.to_device(host_state["fuel"]),
        "fuel_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "flame": cuda.to_device(host_state["flame"]),
        "flame_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),

        # Vorticity diagnostics for confinement forces.
        "vorticity_x": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "vorticity_y": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "vorticity_z": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "vorticity_magnitude": cuda.device_array((nx, ny, nz), dtype=precision_dtype),

        # Static and animated body-force inputs.
        "Fx": cuda.to_device(np.asarray(force_field_data["Fx"], dtype=precision_dtype)),
        "Fy": cuda.to_device(np.asarray(force_field_data["Fy"], dtype=precision_dtype)),
        "Fz": cuda.to_device(np.asarray(force_field_data["Fz"], dtype=precision_dtype)),
        "point_divergence": cuda.to_device(np.asarray(force_field_data["point_divergence"], dtype=precision_dtype)),
        "Fx_base": cuda.to_device(np.asarray(force_field_data["Fx_base"], dtype=precision_dtype)),
        "Fy_base": cuda.to_device(np.asarray(force_field_data["Fy_base"], dtype=precision_dtype)),
        "Fz_base": cuda.to_device(np.asarray(force_field_data["Fz_base"], dtype=precision_dtype)),
        "turbulence_Fx_a": cuda.to_device(np.asarray(turbulence_data["Fx_a"], dtype=precision_dtype)),
        "turbulence_Fy_a": cuda.to_device(np.asarray(turbulence_data["Fy_a"], dtype=precision_dtype)),
        "turbulence_Fz_a": cuda.to_device(np.asarray(turbulence_data["Fz_a"], dtype=precision_dtype)),
        "turbulence_Fx_b": cuda.to_device(np.asarray(turbulence_data["Fx_b"], dtype=precision_dtype)),
        "turbulence_Fy_b": cuda.to_device(np.asarray(turbulence_data["Fy_b"], dtype=precision_dtype)),
        "turbulence_Fz_b": cuda.to_device(np.asarray(turbulence_data["Fz_b"], dtype=precision_dtype)),
        "turbulence_cos_coeffs": cuda.to_device(np.ones(len(turbulence_data["angular_frequencies"]), dtype=precision_dtype)),
        "turbulence_sin_coeffs": cuda.to_device(np.zeros(len(turbulence_data["angular_frequencies"]), dtype=precision_dtype)),

        # Dynamic obstacle mask and obstacle wall velocities.
        "obstacle_mask": cuda.to_device(host_state["obstacle_mask"]),
        "obstacle_velocity_x": cuda.to_device(host_state["obstacle_velocity_x"]),
        "obstacle_velocity_y": cuda.to_device(host_state["obstacle_velocity_y"]),
        "obstacle_velocity_z": cuda.to_device(host_state["obstacle_velocity_z"]),

        # Dynamic source masks and source target values.
        "source_mask": cuda.to_device(host_state["source_mask"]),
        "source_velocity_mask": cuda.to_device(host_state["source_velocity_mask"]),
        "source_temperature": cuda.to_device(host_state["source_temperature"]),
        "source_smoke": cuda.to_device(host_state["source_smoke"]),
        "source_fuel": cuda.to_device(host_state["source_fuel"]),
        "source_velocity_x": cuda.to_device(host_state["source_velocity_x"]),
        "source_velocity_y": cuda.to_device(host_state["source_velocity_y"]),
        "source_velocity_z": cuda.to_device(host_state["source_velocity_z"]),

        # Temporary timestep maxima storage.
        "velocity_maxima": cuda.device_array(3, dtype=np.float32),
    }

    gpu_constants = general_update_data.build_solver_constants(
        simulation_params,
        precision_dtype,
        force_field_data,
    )

    return gpu_fields, gpu_constants
def _update_gpu_source_data(source_field_data, gpu_fields, time_value):
    source_bc.update_source_data_gpu(source_field_data, gpu_fields, time_value)


def _sync_gpu_force_fields(force_field_data, gpu_fields):
    gpu_fields["point_divergence"].copy_to_device(force_field_data["point_divergence"])


def update_animated_source_force_values(simulation_params, gpu_fields, time_value):
    """Update animated source targets and animated constant-force values."""
    return general_update_data.update_animated_source_force_values(
        simulation_params,
        gpu_fields,
        time_value,
        source_updater=_update_gpu_source_data,
        force_updater=_sync_gpu_force_fields,
    )


def update_dynamic_boundary_data_on_gpu(simulation_params, gpu_fields, gpu_constants, time_value):
    """Update animated obstacle/source masks on the host and upload them to the GPU."""
    if not simulation_params.get("HAS_DYNAMIC_BOUNDARIES", False):
        return

    obstacle_data = simulation_params.get("obstacle_data")
    if obstacle_data is not None and obstacle_data.get("runtime") is not None and obstacle_data.get("is_animated", False):
        obstacles.update_dynamic_obstacle_data_gpu(
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
