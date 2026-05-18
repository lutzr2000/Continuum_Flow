import numpy as np
from numba import cuda

import Solver.General.update_data as general_update_data
import Solver.Kernel_GPU.Boundary_Conditions.obstacles as obstacles
import Solver.Kernel_GPU.Boundary_Conditions.source_bc as source_bc
import Solver.Kernel_GPU.kernel_config as kernel_config

GPU_FIELD_DTYPE = np.dtype(np.float32)


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
    nx, ny, nz = host_state["shape"]
    force_field_data = host_state["force_field_data"]
    turbulence_data = host_state["turbulence_data"]
    active_tile_shape = kernel_config.active_tile_shape((nx, ny, nz))

    gpu_fields = {
        # Primary velocity state and scratch buffers.
        "u": cuda.to_device(np.asarray(host_state["u"], dtype=GPU_FIELD_DTYPE)),
        "v": cuda.to_device(np.asarray(host_state["v"], dtype=GPU_FIELD_DTYPE)),
        "w": cuda.to_device(np.asarray(host_state["w"], dtype=GPU_FIELD_DTYPE)),
        "u_work": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "v_work": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "w_work": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "u_tmp": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "v_tmp": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "w_tmp": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),

        # Pressure solve state.
        "p": cuda.to_device(np.asarray(host_state["p"], dtype=GPU_FIELD_DTYPE)),
        "pressure_rhs": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "pressure_rhs_partial_sums": cuda.device_array(
            kernel_config.MAX_REDUCTION_BLOCKS,
            dtype=np.float32,
        ),
        "pressure_rhs_sum": cuda.device_array(1, dtype=np.float32),

        # Advected scalar fields and their work buffers.
        "T": cuda.to_device(np.asarray(host_state["T"], dtype=GPU_FIELD_DTYPE)),
        "temperature_work": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "temperature_tmp": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "smoke": cuda.to_device(np.asarray(host_state["smoke"], dtype=GPU_FIELD_DTYPE)),
        "smoke_work": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "smoke_tmp": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "fuel": cuda.to_device(np.asarray(host_state["fuel"], dtype=GPU_FIELD_DTYPE)),
        "fuel_work": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "fuel_tmp": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "flame": cuda.to_device(np.asarray(host_state["flame"], dtype=GPU_FIELD_DTYPE)),
        "flame_work": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "scalar_active_tiles": cuda.device_array(active_tile_shape, dtype=np.bool_),
        "scalar_active_tiles_dilated": cuda.device_array(active_tile_shape, dtype=np.bool_),

        # Vorticity diagnostics for confinement forces.
        "vorticity_x": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "vorticity_y": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "vorticity_z": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "vorticity_magnitude": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),

        # Static and animated body-force inputs.
        "Fx": cuda.to_device(np.asarray(force_field_data["Fx"], dtype=GPU_FIELD_DTYPE)),
        "Fy": cuda.to_device(np.asarray(force_field_data["Fy"], dtype=GPU_FIELD_DTYPE)),
        "Fz": cuda.to_device(np.asarray(force_field_data["Fz"], dtype=GPU_FIELD_DTYPE)),
        "point_divergence": cuda.to_device(np.asarray(force_field_data["point_divergence"], dtype=GPU_FIELD_DTYPE)),
        "Fx_base": cuda.to_device(np.asarray(force_field_data["Fx_base"], dtype=GPU_FIELD_DTYPE)),
        "Fy_base": cuda.to_device(np.asarray(force_field_data["Fy_base"], dtype=GPU_FIELD_DTYPE)),
        "Fz_base": cuda.to_device(np.asarray(force_field_data["Fz_base"], dtype=GPU_FIELD_DTYPE)),
        "turbulence_Fx_a": cuda.to_device(np.asarray(turbulence_data["Fx_a"], dtype=GPU_FIELD_DTYPE)),
        "turbulence_Fy_a": cuda.to_device(np.asarray(turbulence_data["Fy_a"], dtype=GPU_FIELD_DTYPE)),
        "turbulence_Fz_a": cuda.to_device(np.asarray(turbulence_data["Fz_a"], dtype=GPU_FIELD_DTYPE)),
        "turbulence_Fx_b": cuda.to_device(np.asarray(turbulence_data["Fx_b"], dtype=GPU_FIELD_DTYPE)),
        "turbulence_Fy_b": cuda.to_device(np.asarray(turbulence_data["Fy_b"], dtype=GPU_FIELD_DTYPE)),
        "turbulence_Fz_b": cuda.to_device(np.asarray(turbulence_data["Fz_b"], dtype=GPU_FIELD_DTYPE)),
        "turbulence_mix_factors": cuda.to_device(np.zeros(len(turbulence_data["angular_frequencies"]), dtype=GPU_FIELD_DTYPE)),

        # Dynamic obstacle mask and obstacle wall velocities.
        "obstacle_mask": cuda.to_device(host_state["obstacle_mask"]),
        "obstacle_velocity_x": cuda.to_device(np.asarray(host_state["obstacle_velocity_x"], dtype=GPU_FIELD_DTYPE)),
        "obstacle_velocity_y": cuda.to_device(np.asarray(host_state["obstacle_velocity_y"], dtype=GPU_FIELD_DTYPE)),
        "obstacle_velocity_z": cuda.to_device(np.asarray(host_state["obstacle_velocity_z"], dtype=GPU_FIELD_DTYPE)),

        # Dynamic source masks and source target values.
        "source_mask": cuda.to_device(host_state["source_mask"]),
        "source_velocity_mask": cuda.to_device(host_state["source_velocity_mask"]),
        "source_temperature": cuda.to_device(np.asarray(host_state["source_temperature"], dtype=GPU_FIELD_DTYPE)),
        "source_smoke": cuda.to_device(np.asarray(host_state["source_smoke"], dtype=GPU_FIELD_DTYPE)),
        "source_fuel": cuda.to_device(np.asarray(host_state["source_fuel"], dtype=GPU_FIELD_DTYPE)),
        "source_velocity_x": cuda.to_device(np.asarray(host_state["source_velocity_x"], dtype=GPU_FIELD_DTYPE)),
        "source_velocity_y": cuda.to_device(np.asarray(host_state["source_velocity_y"], dtype=GPU_FIELD_DTYPE)),
        "source_velocity_z": cuda.to_device(np.asarray(host_state["source_velocity_z"], dtype=GPU_FIELD_DTYPE)),

        # Temporary timestep maxima storage.
        "velocity_maxima": cuda.device_array(3, dtype=np.float32),
    }

    gpu_constants = general_update_data.build_solver_constants(
        simulation_params,
        GPU_FIELD_DTYPE,
        force_field_data,
    )

    return gpu_fields, gpu_constants


def _update_gpu_source_data(source_field_data, gpu_fields, time_value):
    source_bc.update_source_data_gpu(source_field_data, gpu_fields, time_value)


def _sync_gpu_force_fields(force_field_data, gpu_fields):
    gpu_fields["point_divergence"].copy_to_device(
        np.asarray(force_field_data["point_divergence"], dtype=GPU_FIELD_DTYPE)
    )


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
