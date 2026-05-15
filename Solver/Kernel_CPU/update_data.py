import numpy as np

import Solver.General.update_data as general_update_data
import Solver.Kernel_CPU.Boundary_Conditions.obstacle_bc as obstacle_bc
import Solver.Kernel_CPU.Boundary_Conditions.source_bc as source_bc


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

    The setup mirrors the GPU state layout closely so the solver loop can stay
    structurally similar between CPU and GPU backends.
    """
    host_state = general_update_data.build_initial_host_state(simulation_params)
    precision_dtype = host_state["precision_dtype"]
    nx, ny, nz = host_state["shape"]
    force_field_data = host_state["force_field_data"]
    turbulence_data = host_state["turbulence_data"]

    cpu_fields = {
        # Primary velocity state and scratch buffers.
        "u": host_state["u"],
        "v": host_state["v"],
        "w": host_state["w"],
        "u_work": np.empty((nx, ny, nz), dtype=precision_dtype),
        "v_work": np.empty((nx, ny, nz), dtype=precision_dtype),
        "w_work": np.empty((nx, ny, nz), dtype=precision_dtype),
        "u_tmp": np.empty((nx, ny, nz), dtype=precision_dtype),
        "v_tmp": np.empty((nx, ny, nz), dtype=precision_dtype),
        "w_tmp": np.empty((nx, ny, nz), dtype=precision_dtype),

        # Pressure solve state.
        "p": host_state["p"],
        "pressure_rhs": np.empty((nx, ny, nz), dtype=precision_dtype),

        # Advected scalar fields and their work buffers.
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
        "turbulence_Fx_a": np.asarray(turbulence_data["Fx_a"], dtype=precision_dtype),
        "turbulence_Fy_a": np.asarray(turbulence_data["Fy_a"], dtype=precision_dtype),
        "turbulence_Fz_a": np.asarray(turbulence_data["Fz_a"], dtype=precision_dtype),
        "turbulence_Fx_b": np.asarray(turbulence_data["Fx_b"], dtype=precision_dtype),
        "turbulence_Fy_b": np.asarray(turbulence_data["Fy_b"], dtype=precision_dtype),
        "turbulence_Fz_b": np.asarray(turbulence_data["Fz_b"], dtype=precision_dtype),
        "turbulence_cos_coeffs": np.ones(len(turbulence_data["angular_frequencies"]), dtype=precision_dtype),
        "turbulence_sin_coeffs": np.zeros(len(turbulence_data["angular_frequencies"]), dtype=precision_dtype),

        # Dynamic obstacle mask and obstacle wall velocities.
        "obstacle_mask": host_state["obstacle_mask"],
        "obstacle_velocity_x": host_state["obstacle_velocity_x"],
        "obstacle_velocity_y": host_state["obstacle_velocity_y"],
        "obstacle_velocity_z": host_state["obstacle_velocity_z"],

        # Dynamic source masks and source target values.
        "source_mask": host_state["source_mask"],
        "source_velocity_mask": host_state["source_velocity_mask"],
        "source_temperature": host_state["source_temperature"],
        "source_smoke": host_state["source_smoke"],
        "source_fuel": host_state["source_fuel"],
        "source_velocity_x": host_state["source_velocity_x"],
        "source_velocity_y": host_state["source_velocity_y"],
        "source_velocity_z": host_state["source_velocity_z"],

        # Temporary timestep maxima storage.
        "velocity_maxima": np.zeros(3, dtype=np.float32),
    }

    cpu_constants = general_update_data.build_solver_constants(
        simulation_params,
        precision_dtype,
        force_field_data,
    )

    return cpu_fields, cpu_constants
def _sync_cpu_source_fields(source_field_data, cpu_fields):
    cpu_fields["source_mask"] = source_field_data["mask"]
    cpu_fields["source_velocity_mask"] = source_field_data["velocity_mask"]
    cpu_fields["source_temperature"] = source_field_data["temperature"]
    cpu_fields["source_smoke"] = source_field_data["smoke"]
    cpu_fields["source_fuel"] = source_field_data["fuel"]
    cpu_fields["source_velocity_x"] = source_field_data["velocity_x"]
    cpu_fields["source_velocity_y"] = source_field_data["velocity_y"]
    cpu_fields["source_velocity_z"] = source_field_data["velocity_z"]


def _update_cpu_source_data(source_field_data, _cpu_fields, time_value):
    source_bc.update_source_data(source_field_data, time_value)


def _sync_cpu_force_fields(force_field_data, cpu_fields):
    cpu_fields["point_divergence"] = force_field_data["point_divergence"]


def update_animated_source_force_values(simulation_params, cpu_fields, time_value):
    """Update animated source targets and animated constant-force values."""
    return general_update_data.update_animated_source_force_values(
        simulation_params,
        cpu_fields,
        time_value,
        source_updater=_update_cpu_source_data,
        source_sync=_sync_cpu_source_fields,
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
        cpu_fields["obstacle_velocity_x"] = obstacle_data["velocity_x"]
        cpu_fields["obstacle_velocity_y"] = obstacle_data["velocity_y"]
        cpu_fields["obstacle_velocity_z"] = obstacle_data["velocity_z"]
        cpu_constants["HAS_OBSTACLE"] = bool(np.any(obstacle_data["mask"]))

    source_field_data = simulation_params.get("source_field_data")
    if source_field_data is not None and source_field_data.get("is_animated", False):
        source_bc.update_source_data(source_field_data, time_value)
        _sync_cpu_source_fields(source_field_data, cpu_fields)
        cpu_constants["HAS_SOURCE"] = bool(np.any(source_field_data["mask"]))
