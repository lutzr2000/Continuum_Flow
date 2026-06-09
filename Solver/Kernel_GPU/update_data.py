import numpy as np
from numba import cuda

import Solver.General.sources as general_sources
import Solver.General.update_data as general_update_data
import Solver.Kernel_GPU.Boundary_Conditions.obstacles as obstacles
import Solver.Kernel_GPU.Boundary_Conditions.source_bc as source_bc
import Solver.Kernel_GPU.kernel_config as kernel_config

GPU_FIELD_DTYPE = np.dtype(np.float32)


def _create_dummy_obstacle_velocity_fields(shape):
    """
    Create placeholder device arrays for the zero-velocity obstacle path.

    These buffers are never read when obstacle wall velocities are disabled, but
    they still need the same 3D rank as the real fields so Numba can type-check
    the obstacle kernel successfully.
    """
    zero = np.zeros(shape, dtype=GPU_FIELD_DTYPE)
    return cuda.to_device(zero), cuda.to_device(zero), cuda.to_device(zero)


def _prepare_gpu_obstacle_data(obstacle_data):
    """
    Prepare animated obstacle runtime data so it can be reused on the GPU.
    """
    if obstacle_data.get("runtime") is not None and obstacle_data.get(
        "is_animated", False
    ):
        obstacles.prepare_dynamic_runtime_for_gpu(obstacle_data["runtime"])


def _prepare_gpu_source_data(source_field_data):
    """
    Prepare animated source field data so it can be reused on the GPU.
    """
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
    source_payload = general_sources.build_runtime_entry_payload(
        simulation_params["source_field_data"],
        0.0,
        obstacles,
    )
    active_tile_shape = kernel_config.active_tile_shape((nx, ny, nz))

    # Reuse three full-volume scratch buffers across vorticity, velocity
    # MacCormack prediction, pressure RHS assembly and scalar MacCormack
    # prediction. These phases run sequentially in one timestep, so their
    # lifetimes do not overlap.
    scratch_x = cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE)
    scratch_y = cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE)
    scratch_z = cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE)
    depart_x = cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE)
    depart_y = cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE)
    depart_z = cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE)
    (
        obstacle_velocity_x,
        obstacle_velocity_y,
        obstacle_velocity_z,
    ) = _create_dummy_obstacle_velocity_fields((nx, ny, nz))
    if host_state["obstacle_has_velocity"]:
        obstacle_velocity_x = cuda.to_device(
            np.asarray(host_state["obstacle_velocity_x"], dtype=GPU_FIELD_DTYPE)
        )
        obstacle_velocity_y = cuda.to_device(
            np.asarray(host_state["obstacle_velocity_y"], dtype=GPU_FIELD_DTYPE)
        )
        obstacle_velocity_z = cuda.to_device(
            np.asarray(host_state["obstacle_velocity_z"], dtype=GPU_FIELD_DTYPE)
        )

    gpu_fields = {
        # Primary velocity state and scratch buffers.
        "u": cuda.to_device(np.asarray(host_state["u"], dtype=GPU_FIELD_DTYPE)),
        "v": cuda.to_device(np.asarray(host_state["v"], dtype=GPU_FIELD_DTYPE)),
        "w": cuda.to_device(np.asarray(host_state["w"], dtype=GPU_FIELD_DTYPE)),
        "u_work": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "v_work": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "w_work": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "u_tmp": scratch_x,
        "v_tmp": scratch_y,
        "w_tmp": scratch_z,
        "depart_x": depart_x,
        "depart_y": depart_y,
        "depart_z": depart_z,
        # Pressure solve state.
        "p": cuda.to_device(np.asarray(host_state["p"], dtype=GPU_FIELD_DTYPE)),
        "pressure_rhs": scratch_x,
        "pressure_rhs_partial_sums": cuda.device_array(
            kernel_config.MAX_REDUCTION_BLOCKS,
            dtype=np.float32,
        ),
        "pressure_rhs_sum": cuda.device_array(1, dtype=np.float32),
        # Advected scalar fields and their work buffers.
        "T": cuda.to_device(np.asarray(host_state["T"], dtype=GPU_FIELD_DTYPE)),
        "temperature_work": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "temperature_tmp": scratch_x,
        "smoke": cuda.to_device(np.asarray(host_state["smoke"], dtype=GPU_FIELD_DTYPE)),
        "smoke_work": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "smoke_tmp": scratch_y,
        "fuel": cuda.to_device(np.asarray(host_state["fuel"], dtype=GPU_FIELD_DTYPE)),
        "fuel_work": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        "fuel_tmp": scratch_z,
        "flame": cuda.to_device(np.asarray(host_state["flame"], dtype=GPU_FIELD_DTYPE)),
        "scalar_active_tiles": cuda.device_array(active_tile_shape, dtype=np.bool_),
        "scalar_active_tiles_dilated": cuda.device_array(
            active_tile_shape, dtype=np.bool_
        ),
        # Vorticity diagnostics for confinement forces.
        "vorticity_x": scratch_x,
        "vorticity_y": scratch_y,
        "vorticity_z": scratch_z,
        "vorticity_magnitude": cuda.device_array((nx, ny, nz), dtype=GPU_FIELD_DTYPE),
        # Static and animated body-force inputs.
        "Fx": cuda.to_device(np.asarray(force_field_data["Fx"], dtype=GPU_FIELD_DTYPE)),
        "Fy": cuda.to_device(np.asarray(force_field_data["Fy"], dtype=GPU_FIELD_DTYPE)),
        "Fz": cuda.to_device(np.asarray(force_field_data["Fz"], dtype=GPU_FIELD_DTYPE)),
        "point_divergence": cuda.to_device(
            np.asarray(force_field_data["point_divergence"], dtype=GPU_FIELD_DTYPE)
        ),
        "Fx_base": cuda.to_device(
            np.asarray(force_field_data["Fx_base"], dtype=GPU_FIELD_DTYPE)
        ),
        "Fy_base": cuda.to_device(
            np.asarray(force_field_data["Fy_base"], dtype=GPU_FIELD_DTYPE)
        ),
        "Fz_base": cuda.to_device(
            np.asarray(force_field_data["Fz_base"], dtype=GPU_FIELD_DTYPE)
        ),
        "turbulence_Fx": cuda.to_device(
            np.asarray(turbulence_data["Fx"], dtype=GPU_FIELD_DTYPE)
        ),
        "turbulence_Fy": cuda.to_device(
            np.asarray(turbulence_data["Fy"], dtype=GPU_FIELD_DTYPE)
        ),
        "turbulence_Fz": cuda.to_device(
            np.asarray(turbulence_data["Fz"], dtype=GPU_FIELD_DTYPE)
        ),
        "turbulence_signed_amplitudes": cuda.to_device(
            np.zeros(len(turbulence_data["angular_frequencies"]), dtype=GPU_FIELD_DTYPE)
        ),
        # Dynamic obstacle mask and obstacle wall velocities.
        "obstacle_mask": cuda.to_device(host_state["obstacle_mask"]),
        "obstacle_velocity_x": obstacle_velocity_x,
        "obstacle_velocity_y": obstacle_velocity_y,
        "obstacle_velocity_z": obstacle_velocity_z,
        # Dynamic source masks and compact per-source authored values.
        "source_mask": cuda.to_device(host_state["source_mask"]),
        "source_entry_masks": cuda.to_device(source_payload["entry_masks"]),
        "source_temperature_values": cuda.to_device(
            np.asarray(source_payload["temperature_values"], dtype=GPU_FIELD_DTYPE)
        ),
        "source_smoke_values": cuda.to_device(
            np.asarray(source_payload["smoke_values"], dtype=GPU_FIELD_DTYPE)
        ),
        "source_fuel_values": cuda.to_device(
            np.asarray(source_payload["fuel_values"], dtype=GPU_FIELD_DTYPE)
        ),
        "source_extra_pressure_values": cuda.to_device(
            np.asarray(source_payload["extra_pressure_values"], dtype=GPU_FIELD_DTYPE)
        ),
        "source_velocity_enabled": cuda.to_device(source_payload["velocity_enabled"]),
        "source_velocity_x_values": cuda.to_device(
            np.asarray(source_payload["velocity_x_values"], dtype=GPU_FIELD_DTYPE)
        ),
        "source_velocity_y_values": cuda.to_device(
            np.asarray(source_payload["velocity_y_values"], dtype=GPU_FIELD_DTYPE)
        ),
        "source_velocity_z_values": cuda.to_device(
            np.asarray(source_payload["velocity_z_values"], dtype=GPU_FIELD_DTYPE)
        ),
        # Temporary timestep maxima storage.
        "velocity_maxima": cuda.device_array(3, dtype=np.float32),
    }

    gpu_constants = general_update_data.build_solver_constants(
        simulation_params,
        GPU_FIELD_DTYPE,
        force_field_data,
    )
    gpu_constants["HAS_OBSTACLE_VELOCITY"] = bool(host_state["obstacle_has_velocity"])

    return gpu_fields, gpu_constants


def _update_gpu_source_data(source_field_data, gpu_fields, time_value):
    """
    Update animated source masks and compact authored values on the GPU.
    """
    source_payload = general_sources.build_runtime_entry_payload(
        source_field_data,
        time_value,
        obstacles,
    )
    gpu_fields["source_temperature_values"].copy_to_device(
        np.asarray(source_payload["temperature_values"], dtype=GPU_FIELD_DTYPE)
    )
    gpu_fields["source_smoke_values"].copy_to_device(
        np.asarray(source_payload["smoke_values"], dtype=GPU_FIELD_DTYPE)
    )
    gpu_fields["source_fuel_values"].copy_to_device(
        np.asarray(source_payload["fuel_values"], dtype=GPU_FIELD_DTYPE)
    )
    gpu_fields["source_extra_pressure_values"].copy_to_device(
        np.asarray(source_payload["extra_pressure_values"], dtype=GPU_FIELD_DTYPE)
    )
    gpu_fields["source_velocity_enabled"].copy_to_device(source_payload["velocity_enabled"])
    gpu_fields["source_velocity_x_values"].copy_to_device(
        np.asarray(source_payload["velocity_x_values"], dtype=GPU_FIELD_DTYPE)
    )
    gpu_fields["source_velocity_y_values"].copy_to_device(
        np.asarray(source_payload["velocity_y_values"], dtype=GPU_FIELD_DTYPE)
    )
    gpu_fields["source_velocity_z_values"].copy_to_device(
        np.asarray(source_payload["velocity_z_values"], dtype=GPU_FIELD_DTYPE)
    )
    source_bc.update_source_data_gpu(source_field_data, gpu_fields, time_value)


def _sync_gpu_force_fields(force_field_data, gpu_fields):
    """
    Upload animated divergence data to the GPU force buffers.
    """
    gpu_fields["point_divergence"].copy_to_device(
        np.asarray(force_field_data["point_divergence"], dtype=GPU_FIELD_DTYPE)
    )


def update_animated_source_force_values(simulation_params, gpu_fields, time_value):
    """
    Update animated source targets and animated constant-force values.
    """
    return general_update_data.update_animated_source_force_values(
        simulation_params,
        gpu_fields,
        time_value,
        source_updater=_update_gpu_source_data,
        force_updater=_sync_gpu_force_fields,
    )


def update_dynamic_boundary_data_on_gpu(
    simulation_params, gpu_fields, gpu_constants, time_value
):
    """
    Update animated obstacle/source masks on the host and upload them to the GPU.
    """
    if not simulation_params.get("HAS_DYNAMIC_BOUNDARIES", False):
        return

    obstacle_data = simulation_params.get("obstacle_data")
    if (
        obstacle_data is not None
        and obstacle_data.get("runtime") is not None
        and obstacle_data.get("is_animated", False)
    ):
        has_obstacle_velocity = bool(gpu_constants.get("HAS_OBSTACLE_VELOCITY", False))
        obstacles.update_dynamic_obstacle_data_gpu(
            obstacle_data["runtime"],
            time_value,
            gpu_fields["obstacle_mask"],
            gpu_fields["obstacle_velocity_x"],
            gpu_fields["obstacle_velocity_y"],
            gpu_fields["obstacle_velocity_z"],
            write_velocity=has_obstacle_velocity,
        )
        gpu_constants["HAS_OBSTACLE"] = bool(
            obstacle_data["runtime"].get("last_has_obstacle", False)
        )

    source_field_data = simulation_params.get("source_field_data")
    if source_field_data is not None and source_field_data.get("is_animated", False):
        gpu_constants["HAS_SOURCE"] = bool(
            source_field_data.get("last_has_source", False)
        )
