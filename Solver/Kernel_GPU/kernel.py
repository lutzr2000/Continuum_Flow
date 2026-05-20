from time import perf_counter

from pathlib import Path
import warnings
import numpy as np
from numba import cuda
from numba.cuda.core.errors import NumbaPerformanceWarning as CudaNumbaPerformanceWarning

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

warnings.filterwarnings(
    "ignore",
    message=r"Grid size .* will likely result in GPU under-utilization due to low occupancy\.",
    category=CudaNumbaPerformanceWarning,
)

GPU_FIELD_DTYPE = np.float32


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
    """
    Apply domain, obstacle and source constraints in the fixed overwrite order.
    """
    u, v, w, p, T, smoke, fuel = BC.domain_bc(u, v, w, p, T, smoke, fuel, bc_config)

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


def _initialise_solver(config):
    """
    Initialise the GPU solver state before the first simulation step.

    This prepares the runtime configuration, uploads all simulation fields to
    the GPU, applies the initial boundary conditions, evaluates animated
    constants at time zero and computes the first stable timestep.

    Args:
        config (dict | None): optional simulation override configuration

    Returns:
        dict: mutable solver state used by main() and _run_time_step()
    """
    #------------Runtime config-------------------
    total_start_time = perf_counter()
    timing_stats = {}
    step_count = 0
    output_frame_count = 0

    section_start = perf_counter()
    memory_tracker = helper_functions.MemoryUsageTracker("VRAM", helper_functions._sample_gpu_memory_usage)
    simulation_params = helper_functions.apply_config(config)
    simulation_params["PRECISION"] = GPU_FIELD_DTYPE
    cancel_flag_path = (((simulation_params.get("meta") or {}).get("cancel_flag_path")) or "").strip()
    helper_functions._record_timing(timing_stats, "init_config", perf_counter() - section_start)
    helper_functions._record_timing(
        timing_stats,
        "init_forces",
        simulation_params.get("INIT_FORCE_BUILD_TIME", 0.0),
    )

    print("################################################################")
    print('Initialise')
    print('Cell count: ', int(simulation_params["NX"] * simulation_params["NY"] * simulation_params["NZ"]))

    #------------Upload fields-------------------
    section_start = perf_counter()
    gpu_fields, gpu_constants = update_data.upload_simulation_state_to_gpu(simulation_params)
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "init_upload_fields", perf_counter() - section_start)

    section_start = perf_counter()
    memory_tracker.sample()
    helper_functions._record_timing(timing_stats, "init_memory_sample", perf_counter() - section_start)

    u = gpu_fields["u"]
    v = gpu_fields["v"]
    w = gpu_fields["w"]
    u_work = gpu_fields["u_work"]
    v_work = gpu_fields["v_work"]
    w_work = gpu_fields["w_work"]
    u_tmp = gpu_fields["u_tmp"]
    v_tmp = gpu_fields["v_tmp"]
    w_tmp = gpu_fields["w_tmp"]
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

    #------------Initial dynamic boundaries-------------------
    section_start = perf_counter()
    update_data.update_dynamic_boundary_data_on_gpu(simulation_params, gpu_fields, gpu_constants, 0.0)
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "init_dynamic_boundaries", perf_counter() - section_start)

    section_start = perf_counter()
    memory_tracker.sample()
    helper_functions._record_timing(timing_stats, "init_memory_sample", perf_counter() - section_start)

    #------------Initial boundary conditions-------------------
    section_start = perf_counter()
    u, v, w, p, T, smoke, fuel, flame = apply_all_BC(
        u, v, w, p, T, smoke, fuel, flame,
        0.0,
        simulation_params["BC_CONFIG"],
        gpu_constants["HAS_OBSTACLE"],
        gpu_fields["obstacle_mask"],
        gpu_fields["obstacle_velocity_x"],
        gpu_fields["obstacle_velocity_y"],
        gpu_fields["obstacle_velocity_z"],
        gpu_constants["HAS_SOURCE"],
        gpu_fields["source_mask"],
        gpu_fields["source_velocity_mask"],
        gpu_fields["source_temperature"],
        gpu_fields["source_smoke"],
        gpu_fields["source_fuel"],
        gpu_fields["source_velocity_x"],
        gpu_fields["source_velocity_y"],
        gpu_fields["source_velocity_z"],
    )
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "init_apply_boundaries", perf_counter() - section_start)

    #------------Initial animated state-------------------
    t = 0.0
    section_start = perf_counter()
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
    helper_functions._record_timing(timing_stats, "init_animated_state", perf_counter() - section_start)

    write_queue = None
    writer_threads = None
    shared_memory_blocks = None
    device_output_staging = None
    solver_diverged = False
    cancel_requested = False

    #------------Initial timestep-------------------
    section_start = perf_counter()
    time_step.reset_velocity_maxima(gpu_fields["velocity_maxima"], velocity_maxima_host_zeros)
    dt, solver_diverged = time_step.compute_new_timestep_gpu(
        u, v, w, gpu_fields["velocity_maxima"], fx_max, fy_max, fz_max,
        gpu_constants["RHO"], gpu_constants["DELTA"], gpu_constants["NU"], simulation_params["CFL_MAX"],
        max_dt=1.0 / simulation_params["OUTPUT_FPS"],
    )
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "init_timestep", perf_counter() - section_start)

    return {
        "total_start_time": total_start_time,
        "timing_stats": timing_stats,
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

    Args:
        state (dict): mutable solver state assembled by _initialise_solver()
        blockspergrid_3d (tuple): CUDA launch grid for volume kernels

    Returns:
        dict: updated solver state after one timestep
    """
    simulation_params = state["simulation_params"]
    gpu_fields = state["gpu_fields"]
    gpu_constants = state["gpu_constants"]
    timing_stats = state["timing_stats"]
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

    #------------Update dynamic inputs-------------------
    if simulation_params.get("HAS_DYNAMIC_BOUNDARIES", False):
        section_start = perf_counter()
        update_data.update_dynamic_boundary_data_on_gpu(simulation_params, gpu_fields, gpu_constants, t)
        cuda.synchronize()
        helper_functions._record_timing(timing_stats, "loop_dynamic_boundaries", perf_counter() - section_start)

    section_start = perf_counter()
    general_update_data.update_animated_constants(simulation_params, gpu_constants, t)
    animated_force = update_data.update_animated_source_force_values(
        simulation_params,
        gpu_fields,
        t,
    )
    helper_functions._record_timing(timing_stats, "loop_animated_state", perf_counter() - section_start)

    #------------Update force fields-------------------
    if turbulence_count > 0:
        section_start = perf_counter()
        gpu_fields["turbulence_mix_factors"].copy_to_device(
            np.sin(turbulence_angular_frequencies * t).astype(GPU_FIELD_DTYPE, copy=False)
        )
        cuda.synchronize()
        helper_functions._record_timing(timing_stats, "loop_turbulence_mix_factors", perf_counter() - section_start)

    section_start = perf_counter()
    forces.update_force_fields[blockspergrid_3d, kernel_config.THREADS_PER_BLOCK_3D](
        gpu_fields["Fx_base"], gpu_fields["Fy_base"], gpu_fields["Fz_base"],
        gpu_fields["turbulence_Fx_a"], gpu_fields["turbulence_Fy_a"], gpu_fields["turbulence_Fz_a"],
        gpu_fields["turbulence_Fx_b"], gpu_fields["turbulence_Fy_b"], gpu_fields["turbulence_Fz_b"],
        gpu_fields["turbulence_mix_factors"],
        turbulence_count,
        np.float32(animated_force["x"]),
        np.float32(animated_force["y"]),
        np.float32(animated_force["z"]),
        gpu_fields["Fx"], gpu_fields["Fy"], gpu_fields["Fz"]
    )
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "loop_force_fields", perf_counter() - section_start)

    section_start = perf_counter()
    forces.buoyancy_approximation[blockspergrid_3d, kernel_config.THREADS_PER_BLOCK_3D](
        T, gpu_fields["Fz"], gpu_constants["BUOANCY_FACTOR"], gpu_constants["T_REFERENCE"]
    )
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "loop_buoyancy", perf_counter() - section_start)

    #------------Vorticity-------------------
    if gpu_constants["VORTICITY"] > 0.0:
        section_start = perf_counter()
        vorticity.compute_vorticity[blockspergrid_3d, kernel_config.THREADS_PER_BLOCK_3D](
            u,
            v,
            w,
            gpu_fields["obstacle_mask"],
            gpu_fields["vorticity_x"],
            gpu_fields["vorticity_y"],
            gpu_fields["vorticity_z"],
            gpu_fields["vorticity_magnitude"],
            gpu_constants["DELTA"],
        )
        vorticity.apply_vorticity_confinement[blockspergrid_3d, kernel_config.THREADS_PER_BLOCK_3D](
            gpu_fields["obstacle_mask"],
            gpu_fields["vorticity_x"],
            gpu_fields["vorticity_y"],
            gpu_fields["vorticity_z"],
            gpu_fields["vorticity_magnitude"],
            gpu_fields["Fx"],
            gpu_fields["Fy"],
            gpu_fields["Fz"],
            gpu_constants["DELTA"],
            gpu_constants["VORTICITY"]
        )
        cuda.synchronize()
        helper_functions._record_timing(timing_stats, "loop_vorticity", perf_counter() - section_start)

    #------------Velocity update-------------------
    section_start = perf_counter()
    active_tile_mask_blocks = kernel_config.volume_blocks_per_grid(
        gpu_fields["scalar_active_tiles"].shape,
        kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK,
    )
    if simulate_sparsely:
        scalar_update.build_active_scalar_tile_mask[
            active_tile_mask_blocks,
            kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK
        ](
            T, smoke, fuel, flame,
            gpu_fields["scalar_active_tiles"],
            gpu_constants["T_REFERENCE"],
            adaptive_domain_threshold,
        )
        scalar_update.dilate_active_scalar_tile_mask[
            active_tile_mask_blocks,
            kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK
        ](
            gpu_fields["scalar_active_tiles"],
            gpu_fields["scalar_active_tiles_dilated"],
            scalar_tile_padding,
        )
    else:
        scalar_update.fill_active_scalar_tile_mask[
            active_tile_mask_blocks,
            kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK
        ](
            gpu_fields["scalar_active_tiles"],
            np.bool_(True),
        )
        scalar_update.fill_active_scalar_tile_mask[
            active_tile_mask_blocks,
            kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK
        ](
            gpu_fields["scalar_active_tiles_dilated"],
            np.bool_(True),
        )
    advection_schemes.preserve_inactive_velocity_tiles[
        scalar_tile_blocks,
        kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
    ](
        u, v, w, u_work, v_work, w_work,
        gpu_fields["scalar_active_tiles_dilated"],
    )
    advection_schemes.advect_velocity_semi_lagrangian[
        scalar_tile_blocks, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
    ](
        u, v, w, u_tmp, v_tmp, w_tmp, dt, gpu_constants["DELTA"],
        gpu_fields["scalar_active_tiles_dilated"],
    )
    advection_schemes.update_velocity_maccormack[
        scalar_tile_blocks, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
    ](
        u, v, w, u_tmp, v_tmp, w_tmp,
        p, dt, gpu_fields["Fx"], gpu_fields["Fy"], gpu_fields["Fz"], u_work, v_work, w_work,
        gpu_constants["DELTA"], gpu_constants["RHO"], gpu_constants["NU"],
        simulation_params["MAX_VELOCITY_INCREMENT_FACTOR"],
        np.float32(0.0),
        np.float32(simulation_params["MACCORMACK_FACTOR"]),
        gpu_fields["scalar_active_tiles_dilated"],
    )
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "loop_velocity", perf_counter() - section_start)

    #------------Velocity swap-------------------
    u, u_work = u_work, u
    v, v_work = v_work, v
    w, w_work = w_work, w

    #------------Pressure solve-------------------
    section_start = perf_counter()
    p = pressure_solve.pressure_poisson(
        u, v, w, p, T, gpu_fields["obstacle_mask"], gpu_fields["pressure_rhs"],
        dt, gpu_fields["point_divergence"],
        gpu_constants["DELTA"], gpu_constants["RHO"], gpu_constants["EXPANSION_RATE"],
        gpu_constants["T_REFERENCE"], simulation_params["MAX_ITER"],
        rhs_partial_sums=gpu_fields["pressure_rhs_partial_sums"],
        rhs_sum_buffer=gpu_fields["pressure_rhs_sum"],
    )
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "loop_pressure", perf_counter() - section_start)

    #------------Velocity projection-------------------
    section_start = perf_counter()
    pressure_solve.project_velocity_kernel[blockspergrid_3d, kernel_config.THREADS_PER_BLOCK_3D](
        u,
        v,
        w,
        p,
        gpu_fields["obstacle_mask"],
        dt,
        gpu_constants["DELTA"],
        gpu_constants["RHO"],
    )
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "loop_projection", perf_counter() - section_start)

    #------------BC-------------------
    section_start = perf_counter()
    u, v, w, p, T, smoke, fuel, flame = apply_all_BC(
        u, v, w, p, T, smoke, fuel, flame,
        dt,
        simulation_params["BC_CONFIG"],
        gpu_constants["HAS_OBSTACLE"],
        gpu_fields["obstacle_mask"],
        gpu_fields["obstacle_velocity_x"],
        gpu_fields["obstacle_velocity_y"],
        gpu_fields["obstacle_velocity_z"],
        False,
        gpu_fields["source_mask"],
        gpu_fields["source_velocity_mask"],
        gpu_fields["source_temperature"],
        gpu_fields["source_smoke"],
        gpu_fields["source_fuel"],
        gpu_fields["source_velocity_x"],
        gpu_fields["source_velocity_y"],
        gpu_fields["source_velocity_z"],
    )
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "loop_apply_boundaries_pressure", perf_counter() - section_start)

    #------------Scalar update-------------------
    section_start = perf_counter()
    scalar_update.preserve_inactive_scalar_tiles[
        scalar_tile_blocks,
        kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
    ](
        T, smoke, fuel, flame,
        temperature_work, smoke_work, fuel_work, flame,
        gpu_fields["scalar_active_tiles_dilated"],
    )
    scalar_update.predict_scalar_fields_maccormack[
        scalar_tile_blocks,
        kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
    ](
        T, smoke, fuel, u, v, w, dt,
        temperature_tmp, smoke_tmp, fuel_tmp,
        gpu_constants["DELTA"],
        gpu_fields["scalar_active_tiles_dilated"],
    )
    scalar_update.update_scalar_fields_maccormack[
        scalar_tile_blocks,
        kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
    ](
        T, smoke, fuel, temperature_tmp, smoke_tmp, fuel_tmp,
        u, v, w, dt,
        temperature_work, smoke_work, fuel_work, flame,
        gpu_constants["DELTA"],
        gpu_constants["TEMPERATURE_DISSIPATION_RATE"],
        gpu_constants["TEMPERATURE_PRODUCTION_RATE"],
        gpu_constants["SMOKE_DISSIPATION_RATE"],
        gpu_constants["SMOKE_PRODUCTION_RATE"],
        gpu_constants["FUEL_BURN_RATE"],
        gpu_constants["FUEL_IGNITION_TEMPERATURE"],
        gpu_constants["MINIMUM_OXYGEN_CONCENTRATION"],
        gpu_constants["T_REFERENCE"],
        np.float32(simulation_params["MACCORMACK_FACTOR"]),
        gpu_fields["scalar_active_tiles_dilated"],
    )
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "loop_scalars", perf_counter() - section_start)

    #------------Swap-------------------
    T, temperature_work = temperature_work, T
    smoke, smoke_work = smoke_work, smoke
    fuel, fuel_work = fuel_work, fuel

    #------------Boundary conditions-------------------
    section_start = perf_counter()
    u, v, w, p, T, smoke, fuel, flame = apply_all_BC(
        u, v, w, p, T, smoke, fuel, flame,
        dt,
        simulation_params["BC_CONFIG"],
        gpu_constants["HAS_OBSTACLE"],
        gpu_fields["obstacle_mask"],
        gpu_fields["obstacle_velocity_x"],
        gpu_fields["obstacle_velocity_y"],
        gpu_fields["obstacle_velocity_z"],
        gpu_constants["HAS_SOURCE"],
        gpu_fields["source_mask"],
        gpu_fields["source_velocity_mask"],
        gpu_fields["source_temperature"],
        gpu_fields["source_smoke"],
        gpu_fields["source_fuel"],
        gpu_fields["source_velocity_x"],
        gpu_fields["source_velocity_y"],
        gpu_fields["source_velocity_z"],
        apply_source_velocity=False,
        apply_source_scalars=True,
    )
    cuda.synchronize()
    helper_functions._record_timing(timing_stats, "loop_apply_boundaries_scalars", perf_counter() - section_start)

    #------------Publish current fields-------------------
    device_fields["u"] = u
    device_fields["v"] = v
    device_fields["w"] = w
    device_fields["p"] = p
    device_fields["T"] = T
    device_fields["smoke"] = smoke
    device_fields["fuel"] = fuel
    device_fields["flame"] = flame

    state.update({
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
    })
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

    Args:
        config (dict | None): optional simulation override configuration
    """
    #------------Initialise-------------------
    state = _initialise_solver(config)

    #------------Prepare output-------------------
    section_start = perf_counter()

    write_queue, buffer_pool, writer_threads, shared_memory_blocks, device_output_staging = output_functions.setup_output(
        state["simulation_params"]["OUTPATH"],
        state["simulation_params"]["FRAME_START"],
        state["simulation_params"]["OUTPUT_BUFFER_VARIABLES"],
        helper_functions.select_fields(
            state["device_fields"],
            state["simulation_params"]["OUTPUT_BUFFER_VARIABLES"],),
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

    helper_functions._record_timing(state["timing_stats"], "init_output_setup", perf_counter() - section_start)

    print('Start time iteration')
    helper_functions.emit_progress(0.0, state["t"])

    blockspergrid_3d = kernel_config.volume_blocks_per_grid(state["u"].shape,kernel_config.THREADS_PER_BLOCK_3D,)

    section_start = perf_counter()
    state["memory_tracker"].sample()
    helper_functions._record_timing(state["timing_stats"], "loop_memory_sample", perf_counter() - section_start)

    #------------Time iteration-------------------
    next_output_time = 0.0
    output_index = 0

    while state["t"] < state["simulation_params"]["T_MAX"]:
        step_start = perf_counter()
        if state["cancel_flag_path"] and Path(state["cancel_flag_path"]).exists():
            state["cancel_requested"] = True
            print('Bake cancellation requested. Stopping the simulation cleanly...')
            break

        #------------One solver step-------------------
        state = _run_time_step(state, blockspergrid_3d)

        #------------Output-------------------
        section_start = perf_counter()
        while state["t"] >= next_output_time:
            output_timings = output_functions.enqueue_device_output(
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
            )
            helper_functions._record_timing(
                state["timing_stats"],
                "loop_output_wait_buffer",
                output_timings["wait_for_buffer"],
            )
            helper_functions._record_timing(
                state["timing_stats"],
                "loop_output_device_pack",
                output_timings["device_pack"],
            )
            helper_functions._record_timing(
                state["timing_stats"],
                "loop_output_device_copy",
                output_timings["device_copy"],
            )

            output_index += 1
            state["output_frame_count"] += 1
            next_output_time += state["simulation_params"]["OUTPUT_TIME_STEP"]
            helper_functions.emit_progress(
                state["t"] / state["simulation_params"]["T_MAX"] * 100.0,
                state["t"],
            )
            
        helper_functions._record_timing(state["timing_stats"], "loop_output", perf_counter() - section_start)

        #------------Adaptive timestep-------------------
        state["t"] += state["dt"]

        section_start = perf_counter()
        time_step.reset_velocity_maxima(
            state["gpu_fields"]["velocity_maxima"],
            state["velocity_maxima_host_zeros"])
        
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
        helper_functions._record_timing(state["timing_stats"], "loop_timestep", perf_counter() - section_start)

        section_start = perf_counter()
        state["memory_tracker"].sample()
        helper_functions._record_timing(state["timing_stats"], "loop_memory_sample", perf_counter() - section_start)

        if state["solver_diverged"]:
            print('ERROR: The solver diverged, stopping the simulation!')
            break

        state["dt"] = dt_new
        state["step_count"] += 1
        helper_functions._record_timing(state["timing_stats"], "loop_total", perf_counter() - step_start)

    #------------Shutdown output-------------------
    if state["write_queue"] is not None:
        section_start = perf_counter()
        output_functions.shutdown_output(
            state["write_queue"],
            state["writer_threads"],
            state["shared_memory_blocks"],
        )
        helper_functions._record_timing(state["timing_stats"], "shutdown_output", perf_counter() - section_start)

    #------------Conclusion-------------------
    if state["solver_diverged"]:
        print('Simulation stopped after solver divergence.')
    elif state["cancel_requested"]:
        print('Simulation cancelled after clean shutdown.')
    else:
        helper_functions.emit_progress(100.0, state["simulation_params"]["T_MAX"])
        print('Simulation finished!')

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
    print(f'Solver runtime: {total_runtime:.3f} s')
    print("################################################################")
