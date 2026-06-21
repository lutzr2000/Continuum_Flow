from time import perf_counter
from pathlib import Path
import numpy as np
from numba import cuda

import Solver.Kernel_GPU.Boundary_Conditions.domain_bc as BC
import Solver.Kernel_GPU.advection_schemes as advection_schemes
import Solver.Kernel_GPU.scalar_update as scalar_update
import Solver.Kernel_GPU.pressure_solve as pressure_solve
import Solver.Kernel_GPU.vorticity as vorticity
import Solver.General.helper_functions as helper_functions
import Solver.General.output_functions as output_functions
import Solver.Kernel_GPU.kernel_config as kernel_config
import Solver.Kernel_GPU.Boundary_Conditions.obstacle_bc as obstacle_bc
import Solver.Kernel_GPU.Boundary_Conditions.source_bc as source_bc
import Solver.General.update_masks as update_masks

GPU_FIELD_DTYPE = np.float32


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


def solver(config,obstacle_base_masks,obstacle_mask,source_masks):
    total_start_time = perf_counter()
    simulations = config.get("simulations")
    cancel_flag_path = ((config.get("meta") or {}).get("cancel_flag_path") or "").strip()
    cancel_requested = False
    output_frame_count = 0

    # ------------time-------------------
    t = 0
    dt = simulations[0].get("settings").get("dt")
    t_max = simulations[0].get("settings").get("simulation_length")

    # ------------dimensions------------------
    delta = simulations[0].get("domain").get("resolution")
    nx = simulations[0]["domain"]["grid"]["nx"]
    ny = simulations[0]["domain"]["grid"]["ny"]
    nz = simulations[0]["domain"]["grid"]["nz"]
    shape = (nx,ny,nz)

    origin_x = -0.5 * nx * delta
    origin_y = -0.5 * ny * delta
    origin_z = 0.0

    print("################################################################")
    print("Initialise")
    print("Cell count: ",int(nx * ny * nz))

    # ------------tiles------------------
    active_tile_shape = kernel_config.active_tile_shape((shape))

    scalar_active_tiles = cuda.device_array(active_tile_shape, dtype=np.bool_)
    scalar_active_tiles_dilated = cuda.device_array(active_tile_shape, dtype=np.bool_)
    scalar_tile_padding = kernel_config.active_tile_padding_tiles()

    active_tile_mask_blocks = kernel_config.volume_blocks_per_grid(
        scalar_active_tiles.shape,
        kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK,
    )

    # ------------fields------------------
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

    # scratch
    scratch_A_x = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    scratch_A_y = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    scratch_A_z = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)

    # vortictiy
    vorticity_magnitude = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)

    # ------------intitialise------------------
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
    ############## PLACEHOLDER ##################### !!!!!!!!!!!!!!!!!!!!!!!

    # ------------Prepare output-------------------
    output_cfg = ((simulations[0].get("outputs") or [None])[0]) or {}
    output_field_config = helper_functions.build_output_field_config(output_cfg)
    output_buffer_variables = helper_functions.collect_buffer_variables(output_field_config)
    output_performance = helper_functions.build_output_performance_config(output_cfg)
    output_dtype = helper_functions.resolve_output_dtype(output_cfg)
    output_time_step = 1.0 / int(output_cfg.get("fps", 24))
    output_sparse_threshold = float(simulations[0].get("settings", {}).get("adaptive_domain_threshold", 0.001))
    host_vdb_writer = config.get("_host_vdb_writer")

    device_fields = {
        "u": u,
        "v": v,
        "w": w,
        "p": p,
        "T": temperature,
        "smoke": smoke,
        "fuel": fuel,
        "flame": flame,
        "obstacle_mask": obstacle_mask,
        "source_mask": np.any(np.stack(source_masks, axis=0), axis=0) if source_masks else np.zeros(shape, dtype=np.bool_),
    }

    (
        write_queue,
        buffer_pool,
        writer_threads,
        shared_memory_blocks,
        device_output_staging,
    ) = output_functions.setup_output(
        output_cfg.get("output_path", ""),
        int(simulations[0].get("settings", {}).get("start_frame", 0)),
        output_buffer_variables,
        helper_functions.select_fields(device_fields, output_buffer_variables),
        output_performance["buffer_count"],
        output_performance["writer_processes"],
        delta,
        host_vdb_writer,
        storage_dtype=output_dtype,
    )

    # ------------time loop------------------
    print("Start time iteration")
    helper_functions.emit_progress(0.0, t)
    next_output_time = 0.0
    output_index = 0
    while t < t_max:
        if cancel_flag_path and Path(cancel_flag_path).exists():
            cancel_requested = True
            print("Bake cancellation requested. Stopping the simulation cleanly...")
            break

        # ------------Update masks-------------------
        update_masks.update_masks(
            obstacle_mask,
            obstacle_base_masks,
            t,
            delta,
            origin_x,
            origin_y,
            origin_z,
        )

        # ------------Start Active tiles-------------------
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

        # ------------Vorticity-------------------
        if simulations[0].get("physics").get("extras").get("vorticity") > 0.0:
            vorticity.compute_vorticity[
                active_tile_shape, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
            ](
                u,
                v,
                w,
                obstacle_mask,
                vorticity_magnitude,
                delta,
                scalar_active_tiles_dilated,
            )   

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
            obstacle_mask,
            scratch_A_x,
            scratch_A_y,
            scratch_A_z,
            depart_x,
            depart_y,
            depart_z,
            dt,
            u_work,
            v_work,
            w_work,
            delta,
            simulations[0].get("physics").get("fluid").get("density"),
            simulations[0].get("physics").get("fluid").get("viscosity"),
            vorticity_magnitude,
            simulations[0].get("physics").get("extras").get("vorticity"),
            temperature,
            simulations[0].get("physics", {}).get("temperature", {}).get("buoyancy"),
            simulations[0].get("physics").get("temperature").get("reference_temperature"),
            scalar_active_tiles_dilated,
        )

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

        # ------------Swap-------------------
        temperature, temperature_work = temperature_work, temperature
        smoke, smoke_work = smoke_work, smoke
        fuel, fuel_work = fuel_work, fuel

        # ------------Output-------------------
        while t >= next_output_time:
            device_fields["u"] = u
            device_fields["v"] = v
            device_fields["w"] = w
            device_fields["p"] = p
            device_fields["T"] = temperature
            device_fields["smoke"] = smoke
            device_fields["fuel"] = fuel
            device_fields["flame"] = flame
            output_functions.enqueue_device_output(
                write_queue,
                buffer_pool,
                output_buffer_variables,
                helper_functions.select_fields(
                    device_fields,
                    output_buffer_variables,
                ),
                device_output_staging,
                output_index,
                t,
                output_field_config,
                output_sparse_threshold,
            )

            output_index += 1
            output_frame_count += 1
            next_output_time += output_time_step
            helper_functions.emit_progress(
                t / t_max * 100.0,
                t,
            )

        # ------------Memory track-------------------
        if t % 10 == 0:
            ctx = cuda.current_context()
            free, total = ctx.get_memory_info()
            used = total - free
            print(f"VRAM used: {used / 1024**2:.1f} MB")  

        t = t + dt

    # ------------Shutdown-------------------
    if write_queue is not None:
        output_functions.shutdown_output(
            write_queue,
            writer_threads,
            shared_memory_blocks,
        )

    # ------------Conclusion-------------------
    if cancel_requested:
        print("Simulation cancelled after clean shutdown.")
    else:
        helper_functions.emit_progress(100.0, t_max)
        print("Simulation finished!")

    total_runtime = perf_counter() - total_start_time
    print(f"Solver runtime: {total_runtime:.3f} s")
    print("################################################################")
