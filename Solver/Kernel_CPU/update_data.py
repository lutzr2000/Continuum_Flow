import numpy as np

import Solver.General.sources as general_sources
import Solver.General.update_data as general_update_data
import Solver.Kernel_CPU.Boundary_Conditions.obstacle_bc as obstacle_bc
import Solver.Kernel_CPU.Boundary_Conditions.source_bc as source_bc
import Solver.Kernel_CPU.kernel_config as kernel_config


def _set_cpu_obstacle_velocity_fields(
    cpu_fields, obstacle_data, precision_dtype, allocate_velocity
):
    """
    Point CPU obstacle velocity fields either at the live host arrays or at None.
    """
    if allocate_velocity:
        cpu_fields["obstacle_velocity_x"] = np.asarray(
            obstacle_data["velocity_x"], dtype=precision_dtype
        )
        cpu_fields["obstacle_velocity_y"] = np.asarray(
            obstacle_data["velocity_y"], dtype=precision_dtype
        )
        cpu_fields["obstacle_velocity_z"] = np.asarray(
            obstacle_data["velocity_z"], dtype=precision_dtype
        )
    else:
        cpu_fields["obstacle_velocity_x"] = None
        cpu_fields["obstacle_velocity_y"] = None
        cpu_fields["obstacle_velocity_z"] = None
    return allocate_velocity


def rebuild_cpu_boundary_data(simulation_params):
    """Rebuild obstacle/source runtime data so the CPU solver uses CPU-side modules only."""
    return general_update_data.rebuild_boundary_data(
        simulation_params,
        obstacle_bc.build_obstacle_data,
        source_bc.build_source_data,
    )


def upload_simulation_state_to_cpu(simulation_params):
    """
    Allocate the persistent CPU solver fields and work buffers.

    The CPU solver now follows the same MacCormack-based predictor/corrector
    pipeline as the GPU backend, so the predictor buffers are required here as
    real working state rather than compatibility-only padding.
    """
    host_state = general_update_data.build_initial_host_state(simulation_params)
    precision_dtype = host_state["precision_dtype"]
    nx, ny, nz = host_state["shape"]
    force_field_data = host_state["force_field_data"]
    turbulence_data = host_state["turbulence_data"]
    source_payload = general_sources.build_runtime_entry_payload(
        simulation_params["source_field_data"],
        0.0,
        source_bc.obstacles,
    )
    active_tile_shape = kernel_config.active_tile_shape((nx, ny, nz))

    cpu_fields = {
        # Primary velocity state, corrected output buffers and predictor buffers.
        "u": host_state["u"],
        "v": host_state["v"],
        "w": host_state["w"],
        "u_work": np.empty((nx, ny, nz), dtype=precision_dtype),
        "v_work": np.empty((nx, ny, nz), dtype=precision_dtype),
        "w_work": np.empty((nx, ny, nz), dtype=precision_dtype),
        "u_tmp": np.empty((nx, ny, nz), dtype=precision_dtype),
        "v_tmp": np.empty((nx, ny, nz), dtype=precision_dtype),
        "w_tmp": np.empty((nx, ny, nz), dtype=precision_dtype),
        "depart_x": np.empty((nx, ny, nz), dtype=precision_dtype),
        "depart_y": np.empty((nx, ny, nz), dtype=precision_dtype),
        "depart_z": np.empty((nx, ny, nz), dtype=precision_dtype),

        # Pressure solve state.
        "p": host_state["p"],
        "pressure_rhs": np.empty((nx, ny, nz), dtype=precision_dtype),

        # Scalar state, corrected output buffers and predictor buffers.
        "T": host_state["T"],
        "temperature_work": np.empty((nx, ny, nz), dtype=precision_dtype),
        "temperature_tmp": np.empty((nx, ny, nz), dtype=precision_dtype),
        "smoke": host_state["smoke"],
        "smoke_work": np.empty((nx, ny, nz), dtype=precision_dtype),
        "smoke_tmp": np.empty((nx, ny, nz), dtype=precision_dtype),
        "fuel": host_state["fuel"],
        "fuel_work": np.empty((nx, ny, nz), dtype=precision_dtype),
        "fuel_tmp": np.empty((nx, ny, nz), dtype=precision_dtype),
        "flame": host_state["flame"],
        "flame_work": np.empty((nx, ny, nz), dtype=precision_dtype),

        # Vorticity diagnostics for confinement forces.
        "vorticity_x": np.empty((nx, ny, nz), dtype=precision_dtype),
        "vorticity_y": np.empty((nx, ny, nz), dtype=precision_dtype),
        "vorticity_z": np.empty((nx, ny, nz), dtype=precision_dtype),
        "vorticity_magnitude": np.empty((nx, ny, nz), dtype=precision_dtype),

        # Static and animated body-force inputs.
        "Fx": np.asarray(force_field_data["Fx"], dtype=precision_dtype).copy(),
        "Fy": np.asarray(force_field_data["Fy"], dtype=precision_dtype).copy(),
        "Fz": np.asarray(force_field_data["Fz"], dtype=precision_dtype).copy(),
        "point_divergence": np.asarray(force_field_data["point_divergence"], dtype=precision_dtype),
        "Fx_base": np.asarray(force_field_data["Fx_base"], dtype=precision_dtype),
        "Fy_base": np.asarray(force_field_data["Fy_base"], dtype=precision_dtype),
        "Fz_base": np.asarray(force_field_data["Fz_base"], dtype=precision_dtype),
        "turbulence_Fx": np.asarray(turbulence_data["Fx"], dtype=precision_dtype),
        "turbulence_Fy": np.asarray(turbulence_data["Fy"], dtype=precision_dtype),
        "turbulence_Fz": np.asarray(turbulence_data["Fz"], dtype=precision_dtype),
        "turbulence_amplitudes": np.asarray(turbulence_data["amplitudes"], dtype=precision_dtype),
        "turbulence_signed_amplitudes": np.zeros(
            len(turbulence_data["angular_frequencies"]), dtype=precision_dtype
        ),

        # Dynamic obstacle mask and obstacle wall velocities.
        "obstacle_mask": host_state["obstacle_mask"],
        "obstacle_velocity_x": host_state["obstacle_velocity_x"],
        "obstacle_velocity_y": host_state["obstacle_velocity_y"],
        "obstacle_velocity_z": host_state["obstacle_velocity_z"],

        # Dynamic source masks and compact per-source authored values.
        "source_mask": host_state["source_mask"],
        "source_entry_masks": source_payload["entry_masks"],
        "source_temperature_values": np.asarray(
            source_payload["temperature_values"], dtype=precision_dtype
        ),
        "source_smoke_values": np.asarray(
            source_payload["smoke_values"], dtype=precision_dtype
        ),
        "source_fuel_values": np.asarray(
            source_payload["fuel_values"], dtype=precision_dtype
        ),
        "source_extra_pressure_values": np.asarray(
            source_payload["extra_pressure_values"], dtype=precision_dtype
        ),
        "source_velocity_enabled": source_payload["velocity_enabled"],
        "source_velocity_x_values": np.asarray(
            source_payload["velocity_x_values"], dtype=precision_dtype
        ),
        "source_velocity_y_values": np.asarray(
            source_payload["velocity_y_values"], dtype=precision_dtype
        ),
        "source_velocity_z_values": np.asarray(
            source_payload["velocity_z_values"], dtype=precision_dtype
        ),

        # Sparse simulation masks.
        "scalar_active_tiles": np.zeros(active_tile_shape, dtype=np.bool_),
        "scalar_active_tiles_dilated": np.zeros(active_tile_shape, dtype=np.bool_),

        # Temporary timestep maxima storage.
        "velocity_maxima": np.zeros(3, dtype=np.float32),
    }

    cpu_constants = general_update_data.build_solver_constants(
        simulation_params,
        precision_dtype,
        force_field_data,
    )
    cpu_constants["HAS_OBSTACLE_VELOCITY"] = bool(host_state["obstacle_has_velocity"])

    return cpu_fields, cpu_constants


def _sync_cpu_source_fields(source_field_data, cpu_fields, time_value=0.0):
    source_payload = general_sources.build_runtime_entry_payload(
        source_field_data,
        time_value,
        source_bc.obstacles,
    )
    cpu_fields["source_mask"] = source_payload["source_mask"]
    cpu_fields["source_entry_masks"] = source_payload["entry_masks"]
    cpu_fields["source_temperature_values"] = source_payload["temperature_values"]
    cpu_fields["source_smoke_values"] = source_payload["smoke_values"]
    cpu_fields["source_fuel_values"] = source_payload["fuel_values"]
    cpu_fields["source_extra_pressure_values"] = source_payload["extra_pressure_values"]
    cpu_fields["source_velocity_enabled"] = source_payload["velocity_enabled"]
    cpu_fields["source_velocity_x_values"] = source_payload["velocity_x_values"]
    cpu_fields["source_velocity_y_values"] = source_payload["velocity_y_values"]
    cpu_fields["source_velocity_z_values"] = source_payload["velocity_z_values"]

def _update_cpu_source_data(source_field_data, cpu_fields, time_value):
    source_bc.update_source_data(source_field_data, time_value)
    source_payload = general_sources.build_runtime_entry_payload(
        source_field_data,
        time_value,
        source_bc.obstacles,
    )
    cpu_fields["source_mask"] = source_payload["source_mask"]
    cpu_fields["source_entry_masks"] = source_payload["entry_masks"]
    cpu_fields["source_temperature_values"] = source_payload["temperature_values"]
    cpu_fields["source_smoke_values"] = source_payload["smoke_values"]
    cpu_fields["source_fuel_values"] = source_payload["fuel_values"]
    cpu_fields["source_extra_pressure_values"] = source_payload["extra_pressure_values"]
    cpu_fields["source_velocity_enabled"] = source_payload["velocity_enabled"]
    cpu_fields["source_velocity_x_values"] = source_payload["velocity_x_values"]
    cpu_fields["source_velocity_y_values"] = source_payload["velocity_y_values"]
    cpu_fields["source_velocity_z_values"] = source_payload["velocity_z_values"]


def _sync_cpu_force_fields(force_field_data, cpu_fields):
    cpu_fields["point_divergence"] = force_field_data["point_divergence"]
    cpu_fields["turbulence_amplitudes"] = force_field_data["turbulence"]["amplitudes"]


def update_animated_source_force_values(simulation_params, cpu_fields, time_value):
    """Update animated source targets and animated constant-force values."""
    return general_update_data.update_animated_source_force_values(
        simulation_params,
        cpu_fields,
        time_value,
        source_updater=_update_cpu_source_data,
        force_updater=_sync_cpu_force_fields,
    )


def update_dynamic_boundary_data_on_cpu(simulation_params, cpu_fields, cpu_constants, time_value):
    """Update animated obstacle/source masks and fields on the CPU."""
    if not simulation_params.get("HAS_DYNAMIC_BOUNDARIES", False):
        return

    obstacle_data = simulation_params.get("obstacle_data")
    if obstacle_data is not None and obstacle_data.get("runtime") is not None and obstacle_data.get("is_animated", False):
        obstacle_bc.update_obstacle_mask(obstacle_data, time_value)
        cpu_fields["obstacle_mask"] = obstacle_data["mask"]
        cpu_constants["HAS_OBSTACLE"] = bool(np.any(obstacle_data["mask"]))
        cpu_constants["HAS_OBSTACLE_VELOCITY"] = _set_cpu_obstacle_velocity_fields(
            cpu_fields,
            obstacle_data,
            np.dtype(simulation_params["PRECISION"]),
            bool(cpu_constants.get("HAS_OBSTACLE_VELOCITY", False)),
        )

    source_field_data = simulation_params.get("source_field_data")
    if source_field_data is not None and source_field_data.get("is_animated", False):
        _update_cpu_source_data(source_field_data, cpu_fields, time_value)
        cpu_constants["HAS_SOURCE"] = bool(np.any(source_field_data["mask"]))
