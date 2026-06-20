from time import perf_counter
from pathlib import Path
import numpy as np
from numba import cuda

import Solver.Kernel_GPU.Boundary_Conditions.domain_bc as BC
import Solver.Kernel_GPU.advection_schemes as advection_schemes
import Solver.Kernel_GPU.forces as forces
import Solver.Kernel_GPU.scalar_update as scalar_update
import Solver.Kernel_GPU.pressure_solve as pressure_solve
import Solver.Kernel_GPU.vorticity as vorticity
import Solver.General.helper_functions as helper_functions
import Solver.General.output_functions as output_functions
import Solver.General.update_data as general_update_data
import Solver.Kernel_GPU.update_data as update_data
import Solver.Kernel_GPU.kernel_config as kernel_config
import Solver.Kernel_GPU.Boundary_Conditions.obstacle_bc as obstacle_bc
import Solver.Kernel_GPU.Boundary_Conditions.source_bc as source_bc
import Solver.Kernel_GPU.time_step as time_step

GPU_FIELD_DTYPE = np.float32




def _initialise_solver(config):
    """
    Initialise the GPU solver state before the first simulation step.

    This prepares the runtime configuration, uploads all simulation fields to
    the GPU, applies the initial boundary conditions, evaluates animated
    constants at time zero and computes the first stable timestep.

    """
    # ------------Runtime config-------------------
    total_start_time = perf_counter()
    step_count = 0
    output_frame_count = 0

    memory_tracker = helper_functions.MemoryUsageTracker(
        "VRAM", helper_functions._sample_gpu_memory_usage
    )
    simulation_params = helper_functions.apply_config(config)
    simulation_params["PRECISION"] = GPU_FIELD_DTYPE
    cancel_flag_path = (
        ((simulation_params.get("meta") or {}).get("cancel_flag_path")) or ""
    ).strip()

    print("################################################################")
    print("Initialise")
    print(
        "Cell count: ",
        int(
            simulation_params["NX"] * simulation_params["NY"] * simulation_params["NZ"]
        ),
    )

    # ------------Upload fields-------------------
    gpu_fields, gpu_constants = update_data.upload_simulation_state_to_gpu(
        simulation_params
    )
    cuda.synchronize()

    memory_tracker.sample()

    u = gpu_fields["u"]
    v = gpu_fields["v"]
    w = gpu_fields["w"]
    u_work = gpu_fields["u_work"]
    v_work = gpu_fields["v_work"]
    w_work = gpu_fields["w_work"]
    u_tmp = gpu_fields["u_tmp"]
    v_tmp = gpu_fields["v_tmp"]
    w_tmp = gpu_fields["w_tmp"]
    depart_x = gpu_fields["depart_x"]
    depart_y = gpu_fields["depart_y"]
    depart_z = gpu_fields["depart_z"]
    p = gpu_fields["p"]
    T = gpu_fields["T"]
    temperature_work = gpu_fields["temperature_work"]
    temperature_tmp = gpu_fields["temperature_tmp"]
    smoke = gpu_fields["smoke"]
    smoke_work = gpu_fields["smoke_work"]
    smoke_tmp = gpu_fields["smoke_tmp"]
    fuel = gpu_fields["fuel"]
    fuel_work = gpu_fields["fuel_work"]
    fuel_tmp = gpu_fields["fuel_tmp"]
    flame = gpu_fields["flame"]
    turbulence_angular_frequencies = np.asarray(
        simulation_params["force_field_data"]["turbulence"]["angular_frequencies"],
        dtype=GPU_FIELD_DTYPE,
    )
    turbulence_count = int(turbulence_angular_frequencies.size)
    velocity_maxima_host_zeros = np.zeros(3, dtype=np.float32)
    device_fields = {
        "u": u,
        "v": v,
        "w": w,
        "p": p,
        "T": T,
        "smoke": smoke,
        "fuel": fuel,
        "flame": flame,
    }

    # ------------Initial dynamic boundaries-------------------
    update_data.update_dynamic_boundary_data_on_gpu(
        simulation_params, gpu_fields, gpu_constants, 0.0
    )
    cuda.synchronize()

    memory_tracker.sample()

    # ------------Initial boundary conditions-------------------
    u, v, w, p, T, smoke, fuel, flame = apply_all_BC(
        u,
        v,
        w,
        p,
        T,
        smoke,
        fuel,
        flame,
        0.0,
        simulation_params["BC_CONFIG"],
        gpu_constants["HAS_OBSTACLE"],
        gpu_constants["HAS_OBSTACLE_VELOCITY"],
        gpu_fields["obstacle_mask"],
        gpu_fields["obstacle_velocity_x"],
        gpu_fields["obstacle_velocity_y"],
        gpu_fields["obstacle_velocity_z"],
        gpu_constants["HAS_SOURCE"],
        gpu_fields["source_mask"],
        gpu_fields["source_entry_masks"],
        gpu_fields["source_temperature_values"],
        gpu_fields["source_smoke_values"],
        gpu_fields["source_fuel_values"],
        gpu_fields["source_velocity_enabled"],
        gpu_fields["source_velocity_x_values"],
        gpu_fields["source_velocity_y_values"],
        gpu_fields["source_velocity_z_values"],
    )
    cuda.synchronize()

    # ------------Initial animated state-------------------
    t = 0.0
    general_update_data.update_animated_constants(simulation_params, gpu_constants, t)
    update_data.update_animated_source_force_values(
        simulation_params,
        gpu_fields,
        t,
    )
    fx_max, fy_max, fz_max = helper_functions.estimate_theoretical_force_maxima(
        gpu_constants,
        simulation_params["ANIMATION_STATE"],
    )

    write_queue = None
    writer_threads = None
    shared_memory_blocks = None
    device_output_staging = None
    solver_diverged = False
    cancel_requested = False

    # ------------Initial timestep-------------------
    time_step.reset_velocity_maxima(
        gpu_fields["velocity_maxima"], velocity_maxima_host_zeros
    )
    dt, solver_diverged = time_step.compute_new_timestep_gpu(
        u,
        v,
        w,
        gpu_fields["velocity_maxima"],
        fx_max,
        fy_max,
        fz_max,
        gpu_constants["RHO"],
        gpu_constants["DELTA"],
        gpu_constants["NU"],
        simulation_params["CFL_MAX"],
        max_dt=1.0 / simulation_params["OUTPUT_FPS"],
    )
    cuda.synchronize()

    return {
        "total_start_time": total_start_time,
        "step_count": step_count,
        "output_frame_count": output_frame_count,
        "memory_tracker": memory_tracker,
        "simulation_params": simulation_params,
        "cancel_flag_path": cancel_flag_path,
        "gpu_fields": gpu_fields,
        "gpu_constants": gpu_constants,
        "u": u,
        "v": v,
        "w": w,
        "u_work": u_work,
        "v_work": v_work,
        "w_work": w_work,
        "u_tmp": u_tmp,
        "v_tmp": v_tmp,
        "w_tmp": w_tmp,
        "depart_x": depart_x,
        "depart_y": depart_y,
        "depart_z": depart_z,
        "p": p,
        "T": T,
        "temperature_work": temperature_work,
        "temperature_tmp": temperature_tmp,
        "smoke": smoke,
        "smoke_work": smoke_work,
        "smoke_tmp": smoke_tmp,
        "fuel": fuel,
        "fuel_work": fuel_work,
        "fuel_tmp": fuel_tmp,
        "flame": flame,
        "turbulence_angular_frequencies": turbulence_angular_frequencies,
        "turbulence_count": turbulence_count,
        "velocity_maxima_host_zeros": velocity_maxima_host_zeros,
        "device_fields": device_fields,
        "t": t,
        "dt": dt,
        "fx_max": fx_max,
        "fy_max": fy_max,
        "fz_max": fz_max,
        "write_queue": write_queue,
        "writer_threads": writer_threads,
        "shared_memory_blocks": shared_memory_blocks,
        "device_output_staging": device_output_staging,
        "solver_diverged": solver_diverged,
        "cancel_requested": cancel_requested,
    }


def _run_time_step(state, blockspergrid_3d):
    """
    Advance the GPU solver by exactly one simulation timestep.

    The step updates animated inputs, recomputes force fields, solves pressure,
    advances velocity and scalar buffers, swaps working arrays, reapplies all
    boundary conditions and stores the current field references back into the
    shared solver state.

    """
    simulation_params = state["simulation_params"]
    gpu_fields = state["gpu_fields"]
    gpu_constants = state["gpu_constants"]
    device_fields = state["device_fields"]
    turbulence_angular_frequencies = state["turbulence_angular_frequencies"]
    turbulence_count = state["turbulence_count"]
    simulate_sparsely = state["simulation_params"].get("SIMULATE_SPARSELY", True)
    adaptive_domain_threshold = np.float32(
        state["simulation_params"].get("ADAPTIVE_DOMAIN_THRESHOLD", 0.001)
    )

    u = state["u"]
    v = state["v"]
    w = state["w"]
    u_work = state["u_work"]
    v_work = state["v_work"]
    w_work = state["w_work"]
    u_tmp = state["u_tmp"]
    v_tmp = state["v_tmp"]
    w_tmp = state["w_tmp"]
    depart_x = state["depart_x"]
    depart_y = state["depart_y"]
    depart_z = state["depart_z"]
    p = state["p"]
    T = state["T"]
    temperature_work = state["temperature_work"]
    temperature_tmp = state["temperature_tmp"]
    smoke = state["smoke"]
    smoke_work = state["smoke_work"]
    smoke_tmp = state["smoke_tmp"]
    fuel = state["fuel"]
    fuel_work = state["fuel_work"]
    fuel_tmp = state["fuel_tmp"]
    flame = state["flame"]
    t = state["t"]
    dt = state["dt"]
    scalar_tile_blocks = kernel_config.active_tile_shape(u.shape)
    scalar_tile_padding = kernel_config.active_tile_padding_tiles()
    active_tile_mask_blocks = kernel_config.volume_blocks_per_grid(
        gpu_fields["scalar_active_tiles"].shape,
        kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK,
    )

    # ------------Update dynamic inputs-------------------
    if simulation_params.get("HAS_DYNAMIC_BOUNDARIES", False):
        update_data.update_dynamic_boundary_data_on_gpu(
            simulation_params, gpu_fields, gpu_constants, t
        )
        cuda.synchronize()

    general_update_data.update_animated_constants(simulation_params, gpu_constants, t)
    animated_force = update_data.update_animated_source_force_values(
        simulation_params,
        gpu_fields,
        t,
    )

    # ------------Active tiles-------------------
    if simulate_sparsely:
        scalar_update.build_active_scalar_tile_mask[
            active_tile_mask_blocks, kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK
        ](
            T,
            smoke,
            fuel,
            flame,
            gpu_fields["scalar_active_tiles"],
            gpu_constants["T_REFERENCE"],
            adaptive_domain_threshold,
        )
        scalar_update.dilate_active_scalar_tile_mask[
            active_tile_mask_blocks, kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK
        ](
            gpu_fields["scalar_active_tiles"],
            gpu_fields["scalar_active_tiles_dilated"],
            scalar_tile_padding,
        )
    else:
        scalar_update.fill_active_scalar_tile_mask[
            active_tile_mask_blocks, kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK
        ](
            gpu_fields["scalar_active_tiles"],
            np.bool_(True),
        )
        scalar_update.fill_active_scalar_tile_mask[
            active_tile_mask_blocks, kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK
        ](
            gpu_fields["scalar_active_tiles_dilated"],
            np.bool_(True),
        )
    cuda.synchronize()

    # ------------Update force fields-------------------
    if turbulence_count > 0:
        gpu_fields["turbulence_signed_amplitudes"].copy_to_device(
            (
                np.asarray(
                    simulation_params["force_field_data"]["turbulence"]["amplitudes"],
                    dtype=GPU_FIELD_DTYPE,
                )
                * np.sin(turbulence_angular_frequencies * t)
            ).astype(
                GPU_FIELD_DTYPE, copy=False
            )
        )
        cuda.synchronize()

    forces.update_force_fields[
        scalar_tile_blocks, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
    ](
        gpu_fields["Fx_base"],
        gpu_fields["Fy_base"],
        gpu_fields["Fz_base"],
        gpu_fields["turbulence_Fx"],
        gpu_fields["turbulence_Fy"],
        gpu_fields["turbulence_Fz"],
        gpu_fields["turbulence_signed_amplitudes"],
        turbulence_count,
        np.float32(animated_force["x"]),
        np.float32(animated_force["y"]),
        np.float32(animated_force["z"]),
        gpu_fields["Fx"],
        gpu_fields["Fy"],
        gpu_fields["Fz"],
        gpu_fields["scalar_active_tiles_dilated"],
    )
    cuda.synchronize()

    forces.buoyancy_approximation[
        scalar_tile_blocks, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
    ](
        T,
        gpu_fields["Fz"],
        gpu_constants["BUOANCY_FACTOR"],
        gpu_constants["T_REFERENCE"],
        gpu_fields["scalar_active_tiles_dilated"],
    )
    cuda.synchronize()

    # ------------Vorticity-------------------
    if gpu_constants["VORTICITY"] > 0.0:
        vorticity.compute_vorticity[
            scalar_tile_blocks, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
        ](
            u,
            v,
            w,
            gpu_fields["obstacle_mask"],
            gpu_fields["vorticity_x"],
            gpu_fields["vorticity_y"],
            gpu_fields["vorticity_z"],
            gpu_fields["vorticity_magnitude"],
            gpu_constants["DELTA"],
            gpu_fields["scalar_active_tiles_dilated"],
        )
        vorticity.apply_vorticity_confinement[
            scalar_tile_blocks, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
        ](
            gpu_fields["obstacle_mask"],
            gpu_fields["vorticity_x"],
            gpu_fields["vorticity_y"],
            gpu_fields["vorticity_z"],
            gpu_fields["vorticity_magnitude"],
            gpu_fields["Fx"],
            gpu_fields["Fy"],
            gpu_fields["Fz"],
            gpu_constants["DELTA"],
            gpu_constants["VORTICITY"],
            gpu_fields["scalar_active_tiles_dilated"],
        )
        cuda.synchronize()

    # ------------Velocity update-------------------
    advection_schemes.preserve_inactive_velocity_tiles[
        scalar_tile_blocks, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
    ](
        u,
        v,
        w,
        u_work,
        v_work,
        w_work,
        gpu_fields["scalar_active_tiles_dilated"],
    )
    advection_schemes.advect_velocity_semi_lagrangian[
        scalar_tile_blocks, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
    ](
        u,
        v,
        w,
        u_tmp,
        v_tmp,
        w_tmp,
        depart_x,
        depart_y,
        depart_z,
        dt,
        gpu_constants["DELTA"],
        gpu_fields["scalar_active_tiles_dilated"],
    )
    advection_schemes.update_velocity_maccormack[
        scalar_tile_blocks, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
    ](
        u,
        v,
        w,
        u_tmp,
        v_tmp,
        w_tmp,
        depart_x,
        depart_y,
        depart_z,
        dt,
        gpu_fields["Fx"],
        gpu_fields["Fy"],
        gpu_fields["Fz"],
        u_work,
        v_work,
        w_work,
        gpu_constants["DELTA"],
        gpu_constants["RHO"],
        gpu_constants["NU"],
        np.float32(simulation_params["MACCORMACK_FACTOR"]),
        gpu_fields["scalar_active_tiles_dilated"],
    )
    cuda.synchronize()

    # ------------Velocity swap-------------------
    u, u_work = u_work, u
    v, v_work = v_work, v
    w, w_work = w_work, w

    # ------------Pressure solve-------------------
    p = pressure_solve.pressure_poisson(
        u,
        v,
        w,
        p,
        T,
        gpu_fields["obstacle_mask"],
        gpu_fields["pressure_rhs"],
        dt,
        gpu_fields["point_divergence"],
        gpu_fields["source_mask"],
        gpu_fields["source_entry_masks"],
        gpu_fields["source_extra_pressure_values"],
        gpu_constants["DELTA"],
        gpu_constants["RHO"],
        gpu_constants["EXPANSION_RATE"],
        gpu_constants["T_REFERENCE"],
        gpu_fields["scalar_active_tiles_dilated"],
        simulation_params["MAX_ITER"],
        rhs_partial_sums=gpu_fields["pressure_rhs_partial_sums"],
        rhs_sum_buffer=gpu_fields["pressure_rhs_sum"],
    )
    cuda.synchronize()

    # ------------Velocity projection-------------------
    pressure_solve.project_velocity_kernel[
        scalar_tile_blocks, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
    ](
        u,
        v,
        w,
        p,
        gpu_fields["obstacle_mask"],
        dt,
        gpu_constants["DELTA"],
        gpu_constants["RHO"],
        gpu_fields["scalar_active_tiles_dilated"],
    )
    cuda.synchronize()

    # ------------BC-------------------
    u, v, w, p, T, smoke, fuel, flame = apply_all_BC(
        u,
        v,
        w,
        p,
        T,
        smoke,
        fuel,
        flame,
        dt,
        simulation_params["BC_CONFIG"],
        gpu_constants["HAS_OBSTACLE"],
        gpu_constants["HAS_OBSTACLE_VELOCITY"],
        gpu_fields["obstacle_mask"],
        gpu_fields["obstacle_velocity_x"],
        gpu_fields["obstacle_velocity_y"],
        gpu_fields["obstacle_velocity_z"],
        gpu_constants["HAS_SOURCE"],
        gpu_fields["source_mask"],
        gpu_fields["source_entry_masks"],
        gpu_fields["source_temperature_values"],
        gpu_fields["source_smoke_values"],
        gpu_fields["source_fuel_values"],
        gpu_fields["source_velocity_enabled"],
        gpu_fields["source_velocity_x_values"],
        gpu_fields["source_velocity_y_values"],
        gpu_fields["source_velocity_z_values"],
        apply_source_velocity=True,
        apply_source_scalars=False,
    )
    cuda.synchronize()

    # ------------Scalar update-------------------
    scalar_update.preserve_inactive_scalar_tiles[
        scalar_tile_blocks, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
    ](
        T,
        smoke,
        fuel,
        flame,
        temperature_work,
        smoke_work,
        fuel_work,
        flame,
        gpu_fields["scalar_active_tiles_dilated"],
    )
    scalar_update.predict_scalar_fields_semi_lagrangian[
        scalar_tile_blocks, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
    ](
        T,
        smoke,
        fuel,
        u,
        v,
        w,
        dt,
        temperature_tmp,
        smoke_tmp,
        fuel_tmp,
        depart_x,
        depart_y,
        depart_z,
        gpu_constants["DELTA"],
        gpu_fields["scalar_active_tiles_dilated"],
    )
    scalar_update.update_scalar_fields_maccormack[
        scalar_tile_blocks, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
    ](
        T,
        smoke,
        fuel,
        temperature_tmp,
        smoke_tmp,
        fuel_tmp,
        depart_x,
        depart_y,
        depart_z,
        u,
        v,
        w,
        dt,
        temperature_work,
        smoke_work,
        fuel_work,
        flame,
        gpu_constants["DELTA"],
        gpu_constants["TEMPERATURE_DISSIPATION_RATE"],
        gpu_constants["TEMPERATURE_PRODUCTION_RATE"],
        gpu_constants["SMOKE_DISSIPATION_RATE"],
        gpu_constants["SMOKE_PRODUCTION_RATE"],
        gpu_constants["FUEL_DISSIPATION_RATE"],
        gpu_constants["FUEL_BURN_RATE"],
        gpu_constants["FUEL_IGNITION_TEMPERATURE"],
        gpu_constants["MINIMUM_OXYGEN_CONCENTRATION"],
        gpu_constants["T_REFERENCE"],
        np.float32(simulation_params["MACCORMACK_FACTOR"]),
        gpu_fields["scalar_active_tiles_dilated"],
    )
    cuda.synchronize()

    # ------------Swap-------------------
    T, temperature_work = temperature_work, T
    smoke, smoke_work = smoke_work, smoke
    fuel, fuel_work = fuel_work, fuel

    # ------------Boundary conditions-------------------
    u, v, w, p, T, smoke, fuel, flame = apply_all_BC(
        u,
        v,
        w,
        p,
        T,
        smoke,
        fuel,
        flame,
        dt,
        simulation_params["BC_CONFIG"],
        gpu_constants["HAS_OBSTACLE"],
        gpu_constants["HAS_OBSTACLE_VELOCITY"],
        gpu_fields["obstacle_mask"],
        gpu_fields["obstacle_velocity_x"],
        gpu_fields["obstacle_velocity_y"],
        gpu_fields["obstacle_velocity_z"],
        gpu_constants["HAS_SOURCE"],
        gpu_fields["source_mask"],
        gpu_fields["source_entry_masks"],
        gpu_fields["source_temperature_values"],
        gpu_fields["source_smoke_values"],
        gpu_fields["source_fuel_values"],
        gpu_fields["source_velocity_enabled"],
        gpu_fields["source_velocity_x_values"],
        gpu_fields["source_velocity_y_values"],
        gpu_fields["source_velocity_z_values"],
        apply_source_velocity=False,
        apply_source_scalars=True,
    )
    cuda.synchronize()

    # ------------Publish current fields-------------------
    device_fields["u"] = u
    device_fields["v"] = v
    device_fields["w"] = w
    device_fields["p"] = p
    device_fields["T"] = T
    device_fields["smoke"] = smoke
    device_fields["fuel"] = fuel
    device_fields["flame"] = flame

    state.update(
        {
            "u": u,
            "v": v,
            "w": w,
            "u_work": u_work,
            "v_work": v_work,
            "w_work": w_work,
            "u_tmp": u_tmp,
            "v_tmp": v_tmp,
            "w_tmp": w_tmp,
            "p": p,
            "T": T,
            "temperature_work": temperature_work,
            "temperature_tmp": temperature_tmp,
            "smoke": smoke,
            "smoke_work": smoke_work,
            "smoke_tmp": smoke_tmp,
            "fuel": fuel,
            "fuel_work": fuel_work,
            "fuel_tmp": fuel_tmp,
            "flame": flame,
        }
    )
    return state


# ===============================
# Main
# ===============================


def main(config=None):
    """
    Run the complete GPU fluid simulation from setup to shutdown.

    main() orchestrates solver initialisation, output preparation, repeated
    timestep execution, adaptive timestep updates, progress reporting, output
    shutdown and final timing and memory summaries.

    """
    # ------------Initialise-------------------
    state = _initialise_solver(config)

    # ------------Prepare output-------------------

    (
        write_queue,
        buffer_pool,
        writer_threads,
        shared_memory_blocks,
        device_output_staging,
    ) = output_functions.setup_output(
        state["simulation_params"]["OUTPATH"],
        state["simulation_params"]["FRAME_START"],
        state["simulation_params"]["OUTPUT_BUFFER_VARIABLES"],
        helper_functions.select_fields(
            state["device_fields"],
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
    state["device_output_staging"] = device_output_staging

    print("Start time iteration")
    helper_functions.emit_progress(0.0, state["t"])

    blockspergrid_3d = kernel_config.volume_blocks_per_grid(
        state["u"].shape,
        kernel_config.THREADS_PER_BLOCK_3D,
    )

    state["memory_tracker"].sample()

    # ------------Time iteration-------------------
    next_output_time = 0.0
    output_index = 0

    while state["t"] < state["simulation_params"]["T_MAX"]:
        if state["cancel_flag_path"] and Path(state["cancel_flag_path"]).exists():
            state["cancel_requested"] = True
            print("Bake cancellation requested. Stopping the simulation cleanly...")
            break

        # ------------One solver step-------------------
        state = _run_time_step(state, blockspergrid_3d)

        # ------------Output-------------------
        while state["t"] >= next_output_time:
            output_functions.enqueue_device_output(
                write_queue,
                buffer_pool,
                state["simulation_params"]["OUTPUT_BUFFER_VARIABLES"],
                helper_functions.select_fields(
                    state["device_fields"],
                    state["simulation_params"]["OUTPUT_BUFFER_VARIABLES"],
                ),
                state["device_output_staging"],
                output_index,
                state["t"],
                state["simulation_params"]["OUTPUT_FIELD_CONFIG"],
                state["simulation_params"]["OUTPUT_SPARSE_THRESHOLD"],
            )

            output_index += 1
            state["output_frame_count"] += 1
            next_output_time += state["simulation_params"]["OUTPUT_TIME_STEP"]
            helper_functions.emit_progress(
                state["t"] / state["simulation_params"]["T_MAX"] * 100.0,
                state["t"],
            )

        # ------------Adaptive timestep-------------------
        state["t"] += state["dt"]

        time_step.reset_velocity_maxima(
            state["gpu_fields"]["velocity_maxima"], state["velocity_maxima_host_zeros"]
        )

        dt_new, state["solver_diverged"] = time_step.compute_new_timestep_gpu(
            state["u"],
            state["v"],
            state["w"],
            state["gpu_fields"]["velocity_maxima"],
            state["fx_max"],
            state["fy_max"],
            state["fz_max"],
            state["gpu_constants"]["RHO"],
            state["gpu_constants"]["DELTA"],
            state["gpu_constants"]["NU"],
            state["simulation_params"]["CFL_MAX"],
            max_dt=1.0 / state["simulation_params"]["OUTPUT_FPS"],
        )

        cuda.synchronize()

        state["memory_tracker"].sample()

        if state["solver_diverged"]:
            print("ERROR: The solver diverged, stopping the simulation!")
            break

        state["dt"] = dt_new
        state["step_count"] += 1

    # ------------Shutdown output-------------------
    if state["write_queue"] is not None:
        output_functions.shutdown_output(
            state["write_queue"],
            state["writer_threads"],
            state["shared_memory_blocks"],
        )

    # ------------Conclusion-------------------
    if state["solver_diverged"]:
        print("Simulation stopped after solver divergence.")
    elif state["cancel_requested"]:
        print("Simulation cancelled after clean shutdown.")
    else:
        helper_functions.emit_progress(100.0, state["simulation_params"]["T_MAX"])
        print("Simulation finished!")

    state["memory_tracker"].sample()
    state["memory_tracker"].print_summary()

    total_runtime = perf_counter() - state["total_start_time"]
    print(f"Solver runtime: {total_runtime:.3f} s")
    print("################################################################")



















def get_source_values(simulations, var_name, t, index=None):
    source_entries = simulations[0].get("sources") or []
    values = np.zeros(len(source_entries), dtype=GPU_FIELD_DTYPE)

    for source_idx, source_entry in enumerate(source_entries):
        value = source_entry.get(var_name, 0.0)

        animation_entry = (source_entry.get("animations") or {}).get(var_name) or {}
        animation_times = animation_entry.get("times") or ()
        animation_values = animation_entry.get("values") or ()

        if animation_times and animation_values:
            nearest_time_idx = min(
                range(len(animation_times)),
                key=lambda idx: abs(float(animation_times[idx]) - float(t)),
            )
            value = animation_values[nearest_time_idx]

        if index is not None:
            if value is None:
                value = 0.0
            else:
                value = value[index]

        values[source_idx] = np.asarray(value, dtype=GPU_FIELD_DTYPE)

    return values





def apply_all_BC(
    simulations,
    t,
    u,
    v,
    w,
    p,
    T,
    smoke,
    fuel,
    flame,
    dt,
    obstacle_mask,
    source_masks
):
    """
    Apply domain, obstacle and source constraints in the fixed overwrite order.
    Domain BCs are applied always, source and obstalces are optional depending on
    user config.
    """
    bc_config = simulations[0].get("domain", {}).get("boundary_conditions", {})
    u, v, w, p, T, smoke, fuel = BC.domain_bc(u, v, w, p, T, smoke, fuel, bc_config)

    if bool(np.any(obstacle_mask)):
        blockspergrid = kernel_config.volume_blocks_per_grid(
            obstacle_mask.shape,
            kernel_config.THREADS_PER_BLOCK_3D,
        )
        obstacle_bc.obstacle_bc_kernel[
            blockspergrid, kernel_config.THREADS_PER_BLOCK_3D
        ](
            u,
            v,
            w,
            smoke,
            fuel,
            flame,
            obstacle_mask,
        )

    if bool(np.any(source_masks)):
        source_temperature_values = get_source_values(simulations,"temperature",t)
        source_smoke_values = get_source_values(simulations,"smoke",t)
        source_fuel_values = get_source_values(simulations,"fuel",t)
        source_velocity_x_values = get_source_values(simulations, "velocity", t, 0)
        source_velocity_y_values = get_source_values(simulations, "velocity", t, 1)
        source_velocity_z_values = get_source_values(simulations, "velocity", t, 2)

        source_count = len(source_masks)
        for source_idx in range(source_count):
            source_mask = source_masks[source_idx]

            blockspergrid = kernel_config.volume_blocks_per_grid(
                source_mask.shape,
                kernel_config.THREADS_PER_BLOCK_3D,
            )
            source_bc.source_bc_kernel[blockspergrid, kernel_config.THREADS_PER_BLOCK_3D](
                u,
                v,
                w,
                T,
                smoke,
                fuel,
                source_mask,
                source_temperature_values[source_idx],
                source_smoke_values[source_idx],
                source_fuel_values[source_idx],
                source_velocity_x_values[source_idx],
                source_velocity_y_values[source_idx],
                source_velocity_z_values[source_idx],
                dt
            )
    return u, v, w, p, T, smoke, fuel, flame






def compute_inital_velocity(simulation_cfg):
    total_u = 0.0
    total_v = 0.0
    total_w = 0.0
    inlet_count = 0

    for face_cfg in (simulation_cfg.get("domain") or {}).get("boundary_conditions", {}).values():
        bc_type = face_cfg.get("type", 0)
        if isinstance(bc_type, str):
            if bc_type.strip().upper() != "INFLOW":
                continue
        elif int(bc_type) != 1:
            continue

        velocity = face_cfg.get("velocity") or (0.0, 0.0, 0.0)
        total_u += float(velocity[0]) if len(velocity) > 0 else 0.0
        total_v += float(velocity[1]) if len(velocity) > 1 else 0.0
        total_w += float(velocity[2]) if len(velocity) > 2 else 0.0
        inlet_count += 1

    if inlet_count == 0:
        return 0.0, 0.0, 0.0

    inv_count = 1.0 / float(inlet_count)
    return total_u * inv_count, total_v * inv_count, total_w * inv_count






def solver(config,obstacle_mask,source_masks):
    simulations = config.get("simulations")

    #------------time-------------------
    t = 0
    dt = simulations[0].get("settings").get("dt")
    t_max = simulations[0].get("settings").get("simulation_length")

    #------------dimensions------------------
    delta = simulations[0].get("domain").get("resolution")
    nx = simulations[0]["domain"]["grid"]["nx"]
    ny = simulations[0]["domain"]["grid"]["nx"]
    nz = simulations[0]["domain"]["grid"]["nx"]
    shape = (nx,ny,nz)

    #------------tiles------------------
    active_tile_shape = kernel_config.active_tile_shape((shape))

    scalar_active_tiles = cuda.device_array(active_tile_shape, dtype=np.bool_)
    scalar_active_tiles_dilated = cuda.device_array(active_tile_shape, dtype=np.bool_)
    scalar_tile_padding = kernel_config.active_tile_padding_tiles()

    active_tile_mask_blocks = kernel_config.volume_blocks_per_grid(
        scalar_active_tiles.shape,
        kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK,
    )

    #------------fields------------------
    # velocity
    u = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    u_work = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    v = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    v_work = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    w = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    w_work = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)

    # pressure
    p = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    pressure_rhs_partial_sums = cuda.device_array(kernel_config.MAX_REDUCTION_BLOCKS,dtype=np.float32)
    pressure_rhs_sum = cuda.device_array(1, dtype=np.float32)

    # scalars
    temperature = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    temperature_work = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    smoke = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    smoke_work = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    fuel = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    fuel_work = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    flame = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)

    # depart
    depart_x = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    depart_y = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    depart_z = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)

    # forces
    fx = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    fy = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    fz = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)

    # scratch
    scratch_A_x = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    scratch_A_y = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    scratch_A_z = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)

    # vortictiy
    vorticity_magnitude = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)

    #------------intitialise------------------
    u_initial,v_initial,w_initial = compute_inital_velocity(simulations[0])

    u.copy_to_device(np.full((shape), u_initial, dtype=GPU_FIELD_DTYPE))
    v.copy_to_device(np.full((shape), v_initial, dtype=GPU_FIELD_DTYPE))
    w.copy_to_device(np.full((shape), w_initial, dtype=GPU_FIELD_DTYPE))

    p.copy_to_device(np.full((shape), 0, dtype=GPU_FIELD_DTYPE)) 

    temperature.copy_to_device(np.full((shape), simulations[0].get("physics").get("temperature").get("reference_temperature"), dtype=GPU_FIELD_DTYPE)) 
    smoke.copy_to_device(np.full((shape), 0, dtype=GPU_FIELD_DTYPE)) 
    fuel.copy_to_device(np.full((shape), 0, dtype=GPU_FIELD_DTYPE)) 
    flame.copy_to_device(np.full((shape), 0, dtype=GPU_FIELD_DTYPE)) 


    ############## PLACEHOLDER ##################### !!!!!!!!!!!!!!!!!!!!!!!
    point_divergence = cuda.to_device(np.zeros(shape, dtype=GPU_FIELD_DTYPE))

    #------------time loop------------------
    while t < t_max:
        


        #------------Start Active tiles-------------------
        if simulations[0].get("settings").get("simulate_sparsely"):
            scalar_update.build_active_scalar_tile_mask[
                active_tile_mask_blocks, kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK
            ](
                temperature,
                smoke,
                fuel,
                flame,
                scalar_active_tiles,
                simulations[0].get("physics").get("temperature").get("reference_temperature"),
                simulations[0].get("settings").get("adaptive_domain_threshold"),
            )
            scalar_update.dilate_active_scalar_tile_mask[
                active_tile_mask_blocks, kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK
            ](
                scalar_active_tiles,
                scalar_active_tiles_dilated,
                scalar_tile_padding,
            )
        else:
            scalar_update.fill_active_scalar_tile_mask[
                active_tile_mask_blocks, kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK
            ](
                scalar_active_tiles,
                np.bool_(True),
            )
            scalar_update.fill_active_scalar_tile_mask[
                active_tile_mask_blocks, kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK
            ](
                scalar_active_tiles_dilated,
                np.bool_(True),
            )
        cuda.synchronize()

        #------------Forces-------------------
        forces.buoyancy_approximation[
            active_tile_shape, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
        ](
            temperature,
            fz,
            simulations[0].get("physics", {}).get("temperature", {}).get("buoyancy"),
            simulations[0].get("physics").get("temperature").get("reference_temperature"),
            scalar_active_tiles_dilated,
        )
        cuda.synchronize()

        # ------------Vorticity-------------------
        if simulations[0].get("physics").get("extras").get("vorticity") > 0.0:
            vorticity.compute_vorticity[
                active_tile_shape, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
            ](
                u,
                v,
                w,
                obstacle_mask,
                scratch_A_x,
                scratch_A_y,
                scratch_A_z,
                vorticity_magnitude,
                delta,
                scalar_active_tiles_dilated,
            )
            vorticity.apply_vorticity_confinement[
                active_tile_shape, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
            ](
                obstacle_mask,
                scratch_A_x,
                scratch_A_y,
                scratch_A_z,
                vorticity_magnitude,
                fx,
                fy,
                fz,
                delta,
                simulations[0].get("physics").get("extras").get("vorticity"),
                scalar_active_tiles_dilated,
            )
            cuda.synchronize()

            # ------------Velocity update-------------------
            advection_schemes.preserve_inactive_velocity_tiles[
                active_tile_shape, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
            ](
                u,
                v,
                w,
                u_work,
                v_work,
                w_work,
                scalar_active_tiles_dilated,
            )
            advection_schemes.advect_velocity_semi_lagrangian[
                active_tile_shape, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
            ](
                u,
                v,
                w,
                scratch_A_x,
                scratch_A_y,
                scratch_A_z,
                depart_x,
                depart_y,
                depart_z,
                dt,
                delta,
                scalar_active_tiles_dilated,
            )
            advection_schemes.update_velocity_maccormack[
                active_tile_shape, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
            ](
                u,
                v,
                w,
                scratch_A_x,
                scratch_A_y,
                scratch_A_z,
                depart_x,
                depart_y,
                depart_z,
                dt,
                fx,
                fy,
                fz,
                u_work,
                v_work,
                w_work,
                delta,
                simulations[0].get("physics").get("fluid").get("density"),
                simulations[0].get("physics").get("fluid").get("viscosity"),
                scalar_active_tiles_dilated,
            )
            cuda.synchronize()


            # ------------Velocity swap-------------------
            u, u_work = u_work, u
            v, v_work = v_work, v
            w, w_work = w_work, w

            # ------------Pressure solve-------------------
            extra_pressure = get_source_values(simulations,"extra_pressure",t)

            p = pressure_solve.pressure_poisson(
                u,
                v,
                w,
                p,
                temperature,
                scratch_A_x,
                dt,
                point_divergence,
                source_masks,
                extra_pressure,
                delta,
                simulations[0].get("physics").get("fluid").get("density"),
                simulations[0].get("physics").get("temperature").get("expansion_rate"),
                simulations[0].get("physics").get("temperature").get("reference_temperature"),
                scalar_active_tiles_dilated,
                simulations[0].get("settings").get("iterations"),
                rhs_partial_sums=pressure_rhs_partial_sums,
                rhs_sum_buffer=pressure_rhs_sum,
            )
            cuda.synchronize()

            # ------------Velocity projection-------------------
            pressure_solve.project_velocity_kernel[
                active_tile_shape, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
            ](
                u,
                v,
                w,
                p,
                obstacle_mask,
                dt,
                delta,
                simulations[0].get("physics").get("fluid").get("density"),
                scalar_active_tiles_dilated,
            )
            cuda.synchronize()

            # ------------BC-------------------
            u, v, w, p, temperature, smoke, fuel, flame = apply_all_BC(
                simulations,
                t,
                u,
                v,
                w,
                p,
                temperature,
                smoke,
                fuel,
                flame,
                dt,
                obstacle_mask,
                source_masks,
            )
            cuda.synchronize()

            # ------------Scalar update-------------------
            scalar_update.preserve_inactive_scalar_tiles[
                active_tile_shape, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
            ](
                temperature,
                smoke,
                fuel,
                flame,
                temperature_work,
                smoke_work,
                fuel_work,
                flame,
                scalar_active_tiles_dilated,
            )
            scalar_update.predict_scalar_fields_semi_lagrangian[
                active_tile_shape, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
            ](
                temperature,
                smoke,
                fuel,
                u,
                v,
                w,
                dt,
                scratch_A_x,
                scratch_A_y,
                scratch_A_z,
                depart_x,
                depart_y,
                depart_z,
                delta,
                scalar_active_tiles_dilated,
            )
            scalar_update.update_scalar_fields_maccormack[
                active_tile_shape, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
            ](
                temperature,
                smoke,
                fuel,
                scratch_A_x,
                scratch_A_y,
                scratch_A_z,
                depart_x,
                depart_y,
                depart_z,
                u,
                v,
                w,
                dt,
                temperature_work,
                smoke_work,
                fuel_work,
                flame,
                delta,
                simulations[0].get("physics").get("temperature").get("dissipation"),
                simulations[0].get("physics").get("temperature").get("production_rate"),
                simulations[0].get("physics").get("smoke").get("dissipation"),
                simulations[0].get("physics").get("smoke").get("production_rate"),
                simulations[0].get("physics").get("fuel").get("dissipation"),
                simulations[0].get("physics").get("fuel").get("burn_rate"),
                simulations[0].get("physics").get("fuel").get("ignition_temperature"),
                simulations[0].get("physics").get("fuel").get("minimum_oxygen_concentration"),
                simulations[0].get("physics").get("temperature").get("reference_temperature"),
                scalar_active_tiles_dilated,
            )
            cuda.synchronize()

            # ------------Swap-------------------
            temperature, temperature_work = temperature_work, temperature
            smoke, smoke_work = smoke_work, smoke
            fuel, fuel_work = fuel_work, fuel

        t = t + dt
