from time import perf_counter

import math
import os
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads

import Solver.General.helper_functions as helper_functions
import Solver.General.output_functions as output_functions
import Solver.General.update_data as general_update_data
import Solver.Kernel_CPU.Boundary_Conditions.domain_bc as BC
import Solver.Kernel_CPU.Boundary_Conditions.obstacle_bc as obstacle_bc
import Solver.Kernel_CPU.Boundary_Conditions.source_bc as source_bc
import Solver.Kernel_CPU.advection_schemes as advection_schemes
import Solver.Kernel_CPU.pressure_solve as pressure_solve
import Solver.Kernel_CPU.scalar_update as scalar_update
import Solver.Kernel_CPU.time_step as time_step
import Solver.Kernel_CPU.update_data as update_data


@njit(cache=True, parallel=True, fastmath=True)
def update_force_fields(
    Fx_base, Fy_base, Fz_base,
    turbulence_Fx_a, turbulence_Fy_a, turbulence_Fz_a,
    turbulence_Fx_b, turbulence_Fy_b, turbulence_Fz_b,
    turbulence_cos_coeffs, turbulence_sin_coeffs,
    turbulence_count, animated_force_x, animated_force_y, animated_force_z,
    Fx, Fy, Fz
):
    """Update body-force fields from base, animated and turbulence contributions."""
    nx, ny, nz = Fx.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                fx = Fx_base[i, j, k] + animated_force_x
                fy = Fy_base[i, j, k] + animated_force_y
                fz = Fz_base[i, j, k] + animated_force_z

                for turbulence_index in range(turbulence_count):
                    cos_coeff = turbulence_cos_coeffs[turbulence_index]
                    sin_coeff = turbulence_sin_coeffs[turbulence_index]
                    fx += (
                        cos_coeff * turbulence_Fx_a[turbulence_index, i, j, k] +
                        sin_coeff * turbulence_Fx_b[turbulence_index, i, j, k]
                    )
                    fy += (
                        cos_coeff * turbulence_Fy_a[turbulence_index, i, j, k] +
                        sin_coeff * turbulence_Fy_b[turbulence_index, i, j, k]
                    )
                    fz += (
                        cos_coeff * turbulence_Fz_a[turbulence_index, i, j, k] +
                        sin_coeff * turbulence_Fz_b[turbulence_index, i, j, k]
                    )

                Fx[i, j, k] = fx
                Fy[i, j, k] = fy
                Fz[i, j, k] = fz


@njit(cache=True, parallel=True, fastmath=True)
def buoyancy_approximation(T, Fz, buoyancy_factor, t_reference):
    """Accumulate Boussinesq buoyancy into the z-force field on the CPU."""
    nx, ny, nz = T.shape
    g = 9.81
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                Fz[i, j, k] += g * buoyancy_factor * (T[i, j, k] - t_reference)


@njit(cache=True, parallel=True, fastmath=True)
def apply_vorticity_confinement(
    obstacle_mask, omega_x, omega_y, omega_z, omega_magnitude, Fx, Fy, Fz, delta, vorticity_strength
):
    """Accumulate vorticity confinement forces into the body-force fields on the CPU."""
    nx, ny, nz = omega_magnitude.shape
    half_inv_delta = 0.5 / delta
    for i in prange(2, nx - 2):
        for j in range(2, ny - 2):
            for k in range(2, nz - 2):
                if obstacle_mask[i, j, k]:
                    continue

                grad_x = (omega_magnitude[i + 1, j, k] - omega_magnitude[i - 1, j, k]) * half_inv_delta
                grad_y = (omega_magnitude[i, j + 1, k] - omega_magnitude[i, j - 1, k]) * half_inv_delta
                grad_z = (omega_magnitude[i, j, k + 1] - omega_magnitude[i, j, k - 1]) * half_inv_delta

                grad_length = math.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z)
                if grad_length <= 1.0e-12:
                    continue

                nx_dir = grad_x / grad_length
                ny_dir = grad_y / grad_length
                nz_dir = grad_z / grad_length

                wx = omega_x[i, j, k]
                wy = omega_y[i, j, k]
                wz = omega_z[i, j, k]

                Fx[i, j, k] += vorticity_strength * (ny_dir * wz - nz_dir * wy)
                Fy[i, j, k] += vorticity_strength * (nz_dir * wx - nx_dir * wz)
                Fz[i, j, k] += vorticity_strength * (nx_dir * wy - ny_dir * wx)


def apply_all_BC(
    u, v, w, p, T, smoke, fuel, flame,
    dt,
    bc_config,
    has_obstacle, obstacle_mask, obstacle_velocity_x, obstacle_velocity_y, obstacle_velocity_z,
    has_source, source_mask, source_velocity_mask, source_temperature, source_smoke, source_fuel,
    source_velocity_x, source_velocity_y, source_velocity_z,
    apply_source_velocity=True,
    apply_source_scalars=True,
):
    """Apply domain, obstacle and source constraints in the fixed overwrite order."""
    u, v, w, p, T, smoke, fuel = BC.apply_all_BC(u, v, w, p, T, smoke, fuel, bc_config)

    if has_obstacle:
        u, v, w, smoke, fuel, flame = obstacle_bc.obstacle_bc(
            u, v, w, smoke, fuel, flame,
            obstacle_mask, obstacle_velocity_x, obstacle_velocity_y, obstacle_velocity_z,
        )

    if has_source:
        u, v, w, T, smoke, fuel = source_bc.source_bc(
            u, v, w, T, smoke, fuel,
            source_mask, source_velocity_mask,
            source_temperature, source_smoke, source_fuel,
            source_velocity_x, source_velocity_y, source_velocity_z,
            dt,
            apply_velocity=apply_source_velocity,
            apply_scalars=apply_source_scalars,
        )
    return u, v, w, p, T, smoke, fuel, flame


def _enqueue_host_output(
    write_queue,
    buffer_pool,
    buffered_variables,
    field_map,
    output_index,
    time_value,
    output_field_config,
):
    """Copy one CPU output frame into shared memory and enqueue it for writing."""
    fields = buffer_pool.get()
    for variable_name in buffered_variables:
        np.copyto(fields[variable_name]["array"], field_map[variable_name])
    write_queue.put((int(output_index), float(time_value), fields, output_field_config))


def _sync_host_field_views(host_fields, cpu_fields):
    """Point the exported host-field map at the current primary CPU arrays."""
    host_fields["u"] = cpu_fields["u"]
    host_fields["v"] = cpu_fields["v"]
    host_fields["w"] = cpu_fields["w"]
    host_fields["p"] = cpu_fields["p"]
    host_fields["T"] = cpu_fields["T"]
    host_fields["smoke"] = cpu_fields["smoke"]
    host_fields["fuel"] = cpu_fields["fuel"]
    host_fields["flame"] = cpu_fields["flame"]


def _initialise_solver(config):
    """Initialise the CPU solver state before the first simulation step."""
    total_start_time = perf_counter()
    timing_stats = {}
    step_count = 0
    output_frame_count = 0

    section_start = perf_counter()
    memory_tracker = helper_functions.MemoryUsageTracker("RAM", helper_functions._sample_process_memory_usage)
    simulation_params = helper_functions.apply_config(config)
    if config is not None:
        simulations = config.get("simulations") or []
        if simulations:
            simulation_params["_simulation_cfg"] = simulations[0]
    update_data.rebuild_cpu_boundary_data(simulation_params)
    cancel_flag_path = (((simulation_params.get("meta") or {}).get("cancel_flag_path")) or "").strip()

    available_cores = os.cpu_count() or 1
    reserved_writer_cores = max(0, int(simulation_params.get("OUTPUT_FORWARDER_COUNT", 0)))
    cpu_count = max(1, available_cores - 2 - reserved_writer_cores)
    set_num_threads(cpu_count)
    helper_functions._record_timing(timing_stats, "init_config", perf_counter() - section_start)

    print("################################################################")
    print("Initialise")
    print("Cell count: ", int(simulation_params["NX"] * simulation_params["NY"] * simulation_params["NZ"]))

    section_start = perf_counter()
    cpu_fields, cpu_constants = update_data.upload_simulation_state_to_cpu(simulation_params)
    helper_functions._record_timing(timing_stats, "init_upload_fields", perf_counter() - section_start)

    section_start = perf_counter()
    memory_tracker.sample()
    helper_functions._record_timing(timing_stats, "init_memory_sample", perf_counter() - section_start)

    turbulence_angular_frequencies = np.asarray(
        simulation_params["force_field_data"]["turbulence"]["angular_frequencies"],
        dtype=simulation_params["PRECISION"],
    )
    turbulence_count = int(turbulence_angular_frequencies.size)

    host_fields = {}
    _sync_host_field_views(host_fields, cpu_fields)

    section_start = perf_counter()
    update_data.update_dynamic_boundary_data_on_cpu(simulation_params, cpu_fields, cpu_constants, 0.0)
    helper_functions._record_timing(timing_stats, "init_dynamic_boundaries", perf_counter() - section_start)

    section_start = perf_counter()
    memory_tracker.sample()
    helper_functions._record_timing(timing_stats, "init_memory_sample", perf_counter() - section_start)

    section_start = perf_counter()
    u, v, w, p, T, smoke, fuel, flame = apply_all_BC(
        cpu_fields["u"], cpu_fields["v"], cpu_fields["w"], cpu_fields["p"],
        cpu_fields["T"], cpu_fields["smoke"], cpu_fields["fuel"], cpu_fields["flame"],
        0.0,
        simulation_params["BC_CONFIG"],
        cpu_constants["HAS_OBSTACLE"],
        cpu_fields["obstacle_mask"],
        cpu_fields["obstacle_velocity_x"],
        cpu_fields["obstacle_velocity_y"],
        cpu_fields["obstacle_velocity_z"],
        cpu_constants["HAS_SOURCE"],
        cpu_fields["source_mask"],
        cpu_fields["source_velocity_mask"],
        cpu_fields["source_temperature"],
        cpu_fields["source_smoke"],
        cpu_fields["source_fuel"],
        cpu_fields["source_velocity_x"],
        cpu_fields["source_velocity_y"],
        cpu_fields["source_velocity_z"],
    )
    cpu_fields["u"] = u
    cpu_fields["v"] = v
    cpu_fields["w"] = w
    cpu_fields["p"] = p
    cpu_fields["T"] = T
    cpu_fields["smoke"] = smoke
    cpu_fields["fuel"] = fuel
    cpu_fields["flame"] = flame
    _sync_host_field_views(host_fields, cpu_fields)
    helper_functions._record_timing(timing_stats, "init_apply_boundaries", perf_counter() - section_start)

    t = 0.0
    section_start = perf_counter()
    general_update_data.update_animated_constants(simulation_params, cpu_constants, t)
    animated_force = update_data.update_animated_source_force_values(
        simulation_params,
        cpu_fields,
        t,
    )
    fx_max, fy_max, fz_max = helper_functions.estimate_theoretical_force_maxima(
        cpu_constants,
        simulation_params["ANIMATION_STATE"],
    )
    helper_functions._record_timing(timing_stats, "init_animated_state", perf_counter() - section_start)

    write_queue = None
    writer_threads = None
    shared_memory_blocks = None
    solver_diverged = False
    cancel_requested = False

    section_start = perf_counter()
    dt, solver_diverged = time_step.compute_new_timestep_cpu(
        cpu_fields["u"], cpu_fields["v"], cpu_fields["w"], cpu_fields["velocity_maxima"], fx_max, fy_max, fz_max,
        cpu_constants["RHO"], cpu_constants["DELTA"], cpu_constants["NU"], simulation_params["CFL_MAX"],
        max_dt=1.0 / simulation_params["OUTPUT_FPS"],
    )
    helper_functions._record_timing(timing_stats, "init_timestep", perf_counter() - section_start)

    return {
        "total_start_time": total_start_time,
        "timing_stats": timing_stats,
        "step_count": step_count,
        "output_frame_count": output_frame_count,
        "memory_tracker": memory_tracker,
        "simulation_params": simulation_params,
        "cancel_flag_path": cancel_flag_path,
        "cpu_fields": cpu_fields,
        "cpu_constants": cpu_constants,
        "turbulence_angular_frequencies": turbulence_angular_frequencies,
        "turbulence_count": turbulence_count,
        "host_fields": host_fields,
        "t": t,
        "dt": dt,
        "fx_max": fx_max,
        "fy_max": fy_max,
        "fz_max": fz_max,
        "write_queue": write_queue,
        "writer_threads": writer_threads,
        "shared_memory_blocks": shared_memory_blocks,
        "solver_diverged": solver_diverged,
        "cancel_requested": cancel_requested,
        "animated_force": animated_force,
    }


def _run_time_step(state):
    """Advance the CPU solver by exactly one simulation timestep."""
    simulation_params = state["simulation_params"]
    cpu_fields = state["cpu_fields"]
    cpu_constants = state["cpu_constants"]
    timing_stats = state["timing_stats"]
    turbulence_angular_frequencies = state["turbulence_angular_frequencies"]
    turbulence_count = state["turbulence_count"]

    u = cpu_fields["u"]
    v = cpu_fields["v"]
    w = cpu_fields["w"]
    u_work = cpu_fields["u_work"]
    v_work = cpu_fields["v_work"]
    w_work = cpu_fields["w_work"]
    u_tmp = cpu_fields["u_tmp"]
    v_tmp = cpu_fields["v_tmp"]
    w_tmp = cpu_fields["w_tmp"]
    p = cpu_fields["p"]
    T = cpu_fields["T"]
    temperature_work = cpu_fields["temperature_work"]
    temperature_tmp = cpu_fields["temperature_tmp"]
    smoke = cpu_fields["smoke"]
    smoke_work = cpu_fields["smoke_work"]
    smoke_tmp = cpu_fields["smoke_tmp"]
    fuel = cpu_fields["fuel"]
    fuel_work = cpu_fields["fuel_work"]
    fuel_tmp = cpu_fields["fuel_tmp"]
    flame = cpu_fields["flame"]
    flame_work = cpu_fields["flame_work"]
    t = state["t"]
    dt = state["dt"]

    if simulation_params.get("HAS_DYNAMIC_BOUNDARIES", False):
        section_start = perf_counter()
        update_data.update_dynamic_boundary_data_on_cpu(simulation_params, cpu_fields, cpu_constants, t)
        helper_functions._record_timing(timing_stats, "loop_dynamic_boundaries", perf_counter() - section_start)

    section_start = perf_counter()
    general_update_data.update_animated_constants(simulation_params, cpu_constants, t)
    animated_force = update_data.update_animated_source_force_values(
        simulation_params,
        cpu_fields,
        t,
    )
    helper_functions._record_timing(timing_stats, "loop_animated_state", perf_counter() - section_start)

    if turbulence_count > 0:
        section_start = perf_counter()
        cpu_fields["turbulence_cos_coeffs"][:] = np.cos(turbulence_angular_frequencies * t)
        cpu_fields["turbulence_sin_coeffs"][:] = np.sin(turbulence_angular_frequencies * t)
        helper_functions._record_timing(timing_stats, "loop_turbulence_coefficients", perf_counter() - section_start)

    section_start = perf_counter()
    update_force_fields(
        cpu_fields["Fx_base"], cpu_fields["Fy_base"], cpu_fields["Fz_base"],
        cpu_fields["turbulence_Fx_a"], cpu_fields["turbulence_Fy_a"], cpu_fields["turbulence_Fz_a"],
        cpu_fields["turbulence_Fx_b"], cpu_fields["turbulence_Fy_b"], cpu_fields["turbulence_Fz_b"],
        cpu_fields["turbulence_cos_coeffs"], cpu_fields["turbulence_sin_coeffs"],
        turbulence_count,
        np.float32(animated_force["x"]),
        np.float32(animated_force["y"]),
        np.float32(animated_force["z"]),
        cpu_fields["Fx"], cpu_fields["Fy"], cpu_fields["Fz"],
    )
    helper_functions._record_timing(timing_stats, "loop_force_fields", perf_counter() - section_start)

    section_start = perf_counter()
    buoyancy_approximation(T, cpu_fields["Fz"], cpu_constants["BUOANCY_FACTOR"], cpu_constants["T_REFERENCE"])
    helper_functions._record_timing(timing_stats, "loop_buoyancy", perf_counter() - section_start)

    if cpu_constants["VORTICITY"] > 0.0:
        section_start = perf_counter()
        pressure_solve.pressure_equation_right_side(
            u, v, w, T, cpu_fields["obstacle_mask"], cpu_fields["pressure_rhs"],
            cpu_fields["vorticity_x"], cpu_fields["vorticity_y"], cpu_fields["vorticity_z"],
            cpu_fields["vorticity_magnitude"],
            dt, cpu_fields["point_divergence"], cpu_constants["DELTA"], cpu_constants["RHO"],
            cpu_constants["EXPANSION_RATE"], cpu_constants["T_REFERENCE"],
        )
        helper_functions._record_timing(timing_stats, "loop_vorticity_diagnostics", perf_counter() - section_start)

        section_start = perf_counter()
        apply_vorticity_confinement(
            cpu_fields["obstacle_mask"],
            cpu_fields["vorticity_x"],
            cpu_fields["vorticity_y"],
            cpu_fields["vorticity_z"],
            cpu_fields["vorticity_magnitude"],
            cpu_fields["Fx"],
            cpu_fields["Fy"],
            cpu_fields["Fz"],
            cpu_constants["DELTA"],
            cpu_constants["VORTICITY"],
        )
        helper_functions._record_timing(timing_stats, "loop_vorticity", perf_counter() - section_start)

    section_start = perf_counter()
    advection_schemes.advect_velocity_semi_lagrangian(
        u, v, w, u_tmp, v_tmp, w_tmp, dt, cpu_constants["DELTA"],
    )
    advection_schemes.update_velocity_maccormack(
        u, v, w, u_tmp, v_tmp, w_tmp,
        p, dt, cpu_fields["Fx"], cpu_fields["Fy"], cpu_fields["Fz"], u_work, v_work, w_work,
        cpu_constants["DELTA"], cpu_constants["RHO"], cpu_constants["NU"],
        simulation_params["MAX_VELOCITY_INCREMENT_FACTOR"],
        np.float32(0.0),
        np.float32(simulation_params["MACCORMACK_FACTOR"]),
    )
    helper_functions._record_timing(timing_stats, "loop_velocity", perf_counter() - section_start)

    u, u_work = u_work, u
    v, v_work = v_work, v
    w, w_work = w_work, w

    section_start = perf_counter()
    u, v, w, p, T, smoke, fuel, flame = apply_all_BC(
        u, v, w, p, T, smoke, fuel, flame,
        dt,
        simulation_params["BC_CONFIG"],
        cpu_constants["HAS_OBSTACLE"],
        cpu_fields["obstacle_mask"],
        cpu_fields["obstacle_velocity_x"],
        cpu_fields["obstacle_velocity_y"],
        cpu_fields["obstacle_velocity_z"],
        cpu_constants["HAS_SOURCE"],
        cpu_fields["source_mask"],
        cpu_fields["source_velocity_mask"],
        cpu_fields["source_temperature"],
        cpu_fields["source_smoke"],
        cpu_fields["source_fuel"],
        cpu_fields["source_velocity_x"],
        cpu_fields["source_velocity_y"],
        cpu_fields["source_velocity_z"],
        apply_source_velocity=True,
        apply_source_scalars=False,
    )
    helper_functions._record_timing(timing_stats, "loop_apply_boundaries_velocity", perf_counter() - section_start)

    section_start = perf_counter()
    p = pressure_solve.pressure_poisson(
        u, v, w, p, T, cpu_fields["obstacle_mask"], cpu_fields["pressure_rhs"],
        cpu_fields["vorticity_x"], cpu_fields["vorticity_y"], cpu_fields["vorticity_z"],
        cpu_fields["vorticity_magnitude"], dt, cpu_fields["point_divergence"],
        cpu_constants["DELTA"], cpu_constants["RHO"], cpu_constants["EXPANSION_RATE"],
        cpu_constants["T_REFERENCE"], simulation_params["MAX_ITER"],
    )
    helper_functions._record_timing(timing_stats, "loop_pressure", perf_counter() - section_start)

    section_start = perf_counter()
    u, v, w = pressure_solve.project_velocity(
        u, v, w, p, cpu_fields["obstacle_mask"], dt,
        cpu_constants["DELTA"], cpu_constants["RHO"],
    )
    helper_functions._record_timing(timing_stats, "loop_projection", perf_counter() - section_start)

    section_start = perf_counter()
    u, v, w, p, T, smoke, fuel, flame = apply_all_BC(
        u, v, w, p, T, smoke, fuel, flame,
        dt,
        simulation_params["BC_CONFIG"],
        cpu_constants["HAS_OBSTACLE"],
        cpu_fields["obstacle_mask"],
        cpu_fields["obstacle_velocity_x"],
        cpu_fields["obstacle_velocity_y"],
        cpu_fields["obstacle_velocity_z"],
        False,
        cpu_fields["source_mask"],
        cpu_fields["source_velocity_mask"],
        cpu_fields["source_temperature"],
        cpu_fields["source_smoke"],
        cpu_fields["source_fuel"],
        cpu_fields["source_velocity_x"],
        cpu_fields["source_velocity_y"],
        cpu_fields["source_velocity_z"],
    )
    helper_functions._record_timing(timing_stats, "loop_apply_boundaries_pressure", perf_counter() - section_start)

    section_start = perf_counter()
    scalar_update.predict_scalar_fields_maccormack(
        T, smoke, fuel, u, v, w, dt,
        temperature_tmp, smoke_tmp, fuel_tmp,
        cpu_constants["DELTA"],
    )
    scalar_update.update_scalar_fields_maccormack(
        T, smoke, fuel, temperature_tmp, smoke_tmp, fuel_tmp,
        u, v, w, dt,
        temperature_work, smoke_work, fuel_work, flame_work,
        cpu_constants["DELTA"],
        cpu_constants["TEMPERATURE_DISSIPATION_RATE"],
        cpu_constants["TEMPERATURE_PRODUCTION_RATE"],
        cpu_constants["SMOKE_DISSIPATION_RATE"],
        cpu_constants["SMOKE_PRODUCTION_RATE"],
        cpu_constants["FUEL_BURN_RATE"],
        cpu_constants["FUEL_IGNITION_TEMPERATURE"],
        cpu_constants["MINIMUM_OXYGEN_CONCENTRATION"],
        cpu_constants["T_REFERENCE"],
        np.float32(simulation_params["MACCORMACK_FACTOR"]),
    )
    helper_functions._record_timing(timing_stats, "loop_scalars", perf_counter() - section_start)

    T, temperature_work = temperature_work, T
    smoke, smoke_work = smoke_work, smoke
    fuel, fuel_work = fuel_work, fuel
    flame, flame_work = flame_work, flame

    section_start = perf_counter()
    u, v, w, p, T, smoke, fuel, flame = apply_all_BC(
        u, v, w, p, T, smoke, fuel, flame,
        dt,
        simulation_params["BC_CONFIG"],
        cpu_constants["HAS_OBSTACLE"],
        cpu_fields["obstacle_mask"],
        cpu_fields["obstacle_velocity_x"],
        cpu_fields["obstacle_velocity_y"],
        cpu_fields["obstacle_velocity_z"],
        cpu_constants["HAS_SOURCE"],
        cpu_fields["source_mask"],
        cpu_fields["source_velocity_mask"],
        cpu_fields["source_temperature"],
        cpu_fields["source_smoke"],
        cpu_fields["source_fuel"],
        cpu_fields["source_velocity_x"],
        cpu_fields["source_velocity_y"],
        cpu_fields["source_velocity_z"],
        apply_source_velocity=False,
        apply_source_scalars=True,
    )
    helper_functions._record_timing(timing_stats, "loop_apply_boundaries_scalars", perf_counter() - section_start)

    cpu_fields["u"] = u
    cpu_fields["v"] = v
    cpu_fields["w"] = w
    cpu_fields["u_work"] = u_work
    cpu_fields["v_work"] = v_work
    cpu_fields["w_work"] = w_work
    cpu_fields["u_tmp"] = u_tmp
    cpu_fields["v_tmp"] = v_tmp
    cpu_fields["w_tmp"] = w_tmp
    cpu_fields["p"] = p
    cpu_fields["T"] = T
    cpu_fields["temperature_work"] = temperature_work
    cpu_fields["temperature_tmp"] = temperature_tmp
    cpu_fields["smoke"] = smoke
    cpu_fields["smoke_work"] = smoke_work
    cpu_fields["smoke_tmp"] = smoke_tmp
    cpu_fields["fuel"] = fuel
    cpu_fields["fuel_work"] = fuel_work
    cpu_fields["fuel_tmp"] = fuel_tmp
    cpu_fields["flame"] = flame
    cpu_fields["flame_work"] = flame_work

    _sync_host_field_views(state["host_fields"], cpu_fields)
    state["animated_force"] = animated_force
    return state


def main(config=None):
    """Run the complete CPU fluid simulation from setup to shutdown."""
    state = _initialise_solver(config)

    if state["solver_diverged"]:
        print("ERROR: The solver diverged before output setup, stopping the simulation!")
    else:
        section_start = perf_counter()
        write_queue, buffer_pool, writer_threads, shared_memory_blocks = output_functions.setup_output(
            state["simulation_params"]["OUTPATH"],
            state["simulation_params"]["FRAME_START"],
            state["simulation_params"]["OUTPUT_BUFFER_VARIABLES"],
            helper_functions.select_fields(
                state["host_fields"],
                state["simulation_params"]["OUTPUT_BUFFER_VARIABLES"],
            ),
            state["simulation_params"]["WRITE_QUEUE_SIZE"],
            state["simulation_params"]["OUTPUT_FORWARDER_COUNT"],
            state["simulation_params"]["DELTA"],
            state["simulation_params"]["HOST_VDB_WRITER"],
            storage_dtype=state["simulation_params"]["OUTPUT_DTYPE"],
        )
        state["write_queue"] = write_queue
        state["writer_threads"] = writer_threads
        state["shared_memory_blocks"] = shared_memory_blocks
        helper_functions._record_timing(state["timing_stats"], "init_output_setup", perf_counter() - section_start)

        print("Start time iteration")
        helper_functions.emit_progress(0.0, state["t"])

        section_start = perf_counter()
        state["memory_tracker"].sample()
        helper_functions._record_timing(state["timing_stats"], "loop_memory_sample", perf_counter() - section_start)

        next_output_time = 0.0
        output_index = 0

        while state["t"] < state["simulation_params"]["T_MAX"]:
            step_start = perf_counter()
            if state["cancel_flag_path"] and Path(state["cancel_flag_path"]).exists():
                state["cancel_requested"] = True
                print("Bake cancellation requested. Stopping the simulation cleanly...")
                break

            state = _run_time_step(state)

            section_start = perf_counter()
            while state["t"] >= next_output_time:
                _enqueue_host_output(
                    write_queue,
                    buffer_pool,
                    state["simulation_params"]["OUTPUT_BUFFER_VARIABLES"],
                    state["host_fields"],
                    output_index,
                    state["t"],
                    state["simulation_params"]["OUTPUT_FIELD_CONFIG"],
                )
                output_index += 1
                state["output_frame_count"] += 1
                next_output_time += state["simulation_params"]["OUTPUT_TIME_STEP"]
                helper_functions.emit_progress(
                    state["t"] / state["simulation_params"]["T_MAX"] * 100.0,
                    state["t"],
                )
            helper_functions._record_timing(state["timing_stats"], "loop_output", perf_counter() - section_start)

            state["t"] += state["dt"]

            section_start = perf_counter()
            dt_new, state["solver_diverged"] = time_step.compute_new_timestep_cpu(
                state["cpu_fields"]["u"],
                state["cpu_fields"]["v"],
                state["cpu_fields"]["w"],
                state["cpu_fields"]["velocity_maxima"],
                state["fx_max"],
                state["fy_max"],
                state["fz_max"],
                state["cpu_constants"]["RHO"],
                state["cpu_constants"]["DELTA"],
                state["cpu_constants"]["NU"],
                state["simulation_params"]["CFL_MAX"],
                max_dt=1.0 / state["simulation_params"]["OUTPUT_FPS"],
            )
            helper_functions._record_timing(state["timing_stats"], "loop_timestep", perf_counter() - section_start)

            section_start = perf_counter()
            state["memory_tracker"].sample()
            helper_functions._record_timing(state["timing_stats"], "loop_memory_sample", perf_counter() - section_start)

            if state["solver_diverged"]:
                print("ERROR: The solver diverged, stopping the simulation!")
                break

            state["dt"] = dt_new
            state["step_count"] += 1
            helper_functions._record_timing(state["timing_stats"], "loop_total", perf_counter() - step_start)

        if state["write_queue"] is not None:
            section_start = perf_counter()
            output_functions.shutdown_output(
                state["write_queue"],
                state["writer_threads"],
                state["shared_memory_blocks"],
            )
            helper_functions._record_timing(state["timing_stats"], "shutdown_output", perf_counter() - section_start)

    if state["solver_diverged"]:
        print("Simulation stopped after solver divergence.")
    elif state["cancel_requested"]:
        print("Simulation cancelled after clean shutdown.")
    else:
        helper_functions.emit_progress(100.0, state["simulation_params"]["T_MAX"])
        print("Simulation finished!")

    section_start = perf_counter()
    state["memory_tracker"].sample()
    helper_functions._record_timing(state["timing_stats"], "final_memory_sample", perf_counter() - section_start)
    state["memory_tracker"].print_summary()

    total_runtime = perf_counter() - state["total_start_time"]
    helper_functions._print_timing_summary(
        state["timing_stats"],
        total_runtime,
        state["step_count"],
        state["output_frame_count"],
    )
    print(f"Solver runtime: {total_runtime:.3f} s")
    print("################################################################")
