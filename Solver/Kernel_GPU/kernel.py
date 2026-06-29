import json
import sys
from time import perf_counter
from pathlib import Path
import numpy as np
from numba import cuda
import warnings
warnings.filterwarnings("ignore")

import Solver.Kernel_GPU.Boundary_Conditions.domain_bc as BC
import Solver.Kernel_GPU.advection_schemes as advection_schemes
import Solver.Kernel_GPU.scalar_update as scalar_update
import Solver.Kernel_GPU.pressure_solve as pressure_solve
import Solver.Kernel_GPU.vorticity as vorticity
import Solver.Kernel_GPU.kernel_config as kernel_config
import Solver.Kernel_GPU.Boundary_Conditions.obstacle_bc as obstacle_bc
import Solver.Kernel_GPU.Boundary_Conditions.source_bc as source_bc
import Solver.Kernel_GPU.time_step as time_step
import Solver.Kernel_GPU.update_masks as update_masks
import Solver.Kernel_GPU.output as output
import Solver.General.forces as forces

GPU_FIELD_DTYPE = np.float32
PROGRESS_EVENT_PREFIX = "__CONTINUUM_FLOW_PROGRESS__ "


def _current_device_fields(u, v, w, p, temperature, smoke, fuel, flame):
    """
    Return the currently active device buffers for output export.
    """
    return {
        "u": u,
        "v": v,
        "w": w,
        "pressure": p,
        "temperature": temperature,
        "smoke": smoke,
        "fuel": fuel,
        "flame": flame,
    }


@cuda.jit(cache=True)
def expand_active_tiles_to_mask(active_tiles, output_mask, tile_size):
    i, j, k = cuda.grid(3)

    nx, ny, nz = output_mask.shape
    if i >= nx or j >= ny or k >= nz:
        return

    tile_i = i // tile_size
    tile_j = j // tile_size
    tile_k = k // tile_size

    output_mask[i, j, k] = active_tiles[tile_i, tile_j, tile_k]


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


def _prepare_force_config_device(config_array, row_width):
    host_array = np.ascontiguousarray(
        np.asarray(config_array, dtype=GPU_FIELD_DTYPE).reshape((-1, row_width))
    )
    return cuda.to_device(host_array)

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
    obstacle_velocity_x,
    obstacle_velocity_y,
    obstacle_velocity_z,
    source_masks
):
    """
    Apply domain, obstacle and source constraints in the fixed overwrite order.
    Domain BCs are applied always, source and obstalces are optional depending on
    user config.
    """
    bc_config = simulations[0].get("domain", {}).get("boundary_conditions", {})
    u, v, w, p, T, smoke, fuel = BC.domain_bc(u, v, w, p, T, smoke, fuel, bc_config)

    if obstacle_mask is not None:
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
            obstacle_velocity_x,
            obstacle_velocity_y,
            obstacle_velocity_z,
        )

    source_count = int(source_masks.shape[0])
    if source_count > 0:
        source_temperature_values = get_source_values(simulations,"temperature",t)
        source_smoke_values = get_source_values(simulations,"smoke",t)
        source_fuel_values = get_source_values(simulations,"fuel",t)
        source_velocity_x_values = get_source_values(simulations, "velocity", t, 0)
        source_velocity_y_values = get_source_values(simulations, "velocity", t, 1)
        source_velocity_z_values = get_source_values(simulations, "velocity", t, 2)

        for source_idx in range(source_count):
            source_mask_entry = source_masks[source_idx]
            blockspergrid = kernel_config.volume_blocks_per_grid(
                source_mask_entry.shape,
                kernel_config.THREADS_PER_BLOCK_3D,
            )
            source_bc.source_bc_kernel[blockspergrid, kernel_config.THREADS_PER_BLOCK_3D](
                u,
                v,
                w,
                T,
                smoke,
                fuel,
                source_mask_entry,
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


def create_multigrid_levels(shape, delta, min_size=8):
    p_levels = []
    b_levels = []
    delta_levels = []
    zero_levels = []

    nx, ny, nz = shape
    level = 0

    while nx >= min_size and ny >= min_size and nz >= min_size:
        level_shape = (nx, ny, nz)

        p_levels.append(cuda.device_array(level_shape, dtype=np.float32))
        b_levels.append(cuda.device_array(level_shape, dtype=np.float32))
        zero_levels.append(cuda.to_device(np.zeros(level_shape, dtype=np.float32)))
        delta_levels.append(delta * (2 ** level))

        nx = (nx + 1) // 2
        ny = (ny + 1) // 2
        nz = (nz + 1) // 2
        level += 1

    return p_levels, b_levels, delta_levels, zero_levels


def solver(config,obstacle_base_masks,obstacle_mask,source_base_masks,source_masks,animated_obstacles,animated_sources):
    total_start_time = perf_counter()
    simulations = config.get("simulations")
    cancel_flag_path = ((config.get("meta") or {}).get("cancel_flag_path") or "").strip()
    cancel_requested = False
    output_frame_count = 0

    # ------------time-------------------
    t = 0
    cfl = float(simulations[0].get("settings", {}).get("cfl", 10.0))
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

    # scratch
    scratch_A_x = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    scratch_A_y = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)
    scratch_A_z = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)

    # vortictiy
    vorticity_magnitude = cuda.device_array((shape), dtype=GPU_FIELD_DTYPE)

    # masks
    obstacle_mask = cuda.to_device(np.ascontiguousarray(obstacle_mask, dtype=np.bool_))
    source_mask_host = np.any(np.stack(source_masks, axis=0), axis=0) if source_masks else np.zeros(shape, dtype=np.bool_)
    source_mask = cuda.to_device(np.ascontiguousarray(source_mask_host, dtype=np.bool_))
    source_mask_stack = np.ascontiguousarray(np.asarray(source_masks, dtype=np.bool_)) if source_masks else np.zeros((0,) + shape, dtype=np.bool_)
    source_masks = cuda.to_device(source_mask_stack)

    # multigrid levels
    p_levels, b_levels, delta_levels, zero_levels = create_multigrid_levels(
        shape,
        delta,
        min_size=8,
    )

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

    velocity_maxima = cuda.to_device(np.zeros(3, dtype=np.float32))
    velocity_maxima_host_zeros = np.zeros(3, dtype=np.float32)

    # ------------output------------------
    output_cfg = ((simulations[0].get("outputs") or [None])[0]) or {}
    output_time_step = 1.0 / int(output_cfg.get("fps", 24))

    shared_memory_blocks, writer_slots = output.setup_output(
        simulations[0],
        simulations[0].get("outputs")[0].get("output_path"),
        shape,
    )

    device_fields = {
        "u": u,
        "v": v,
        "w": w,
        "pressure": p,
        "temperature": temperature,
        "smoke": smoke,
        "fuel": fuel,
        "flame": flame,
    }

    # ------------time loop------------------
    print("Start time iteration")
    next_output_time = 0.0
    output_index = 0
    while t < t_max:
        if cancel_flag_path and Path(cancel_flag_path).exists():
            cancel_requested = True
            print("Bake cancellation requested. Stopping the simulation cleanly...")
            break

        time_step.reset_velocity_maxima(velocity_maxima, velocity_maxima_host_zeros)
        dt = time_step.compute_new_timestep_gpu(
            u,
            v,
            w,
            velocity_maxima,
            delta,
            cfl,
            output_time_step,
        )

        # ------------Clear scratch-------------------
        scratch_A_x.copy_to_device(zero_levels[0])
        scratch_A_y.copy_to_device(zero_levels[0])
        scratch_A_z.copy_to_device(zero_levels[0])

        # ------------Update masks-------------------
        if animated_sources:
            update_masks.update_masks(
                source_masks,
                source_base_masks,
                t,
                delta,
                origin_x,
                origin_y,
                origin_z,
                aggregate_mask=source_mask,
            )

        if animated_obstacles:
            update_masks.update_masks(
                obstacle_mask,
                obstacle_base_masks,
                t,
                delta,
                origin_x,
                origin_y,
                origin_z,
                scratch_A_x,
                scratch_A_y,
                scratch_A_z,
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
            scratch_A_x,
            scratch_A_y,
            scratch_A_z,
            source_masks,
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

        # ------------force params-------------------   
        fx_const, fy_const, fz_const = forces.constant_force(simulations[0],t)     
        swirl_config, has_swirl_nodes = forces.swirl_force(simulations[0],t)
        turbulence_config, has_turbulence_nodes = forces.turbulence_force(simulations[0],t)
        swirl_config_device = _prepare_force_config_device(swirl_config, 8)
        turbulence_config_device = _prepare_force_config_device(turbulence_config, 4)

        # ------------Velocity update-------------------
        u_work.copy_to_device(u)
        v_work.copy_to_device(v)
        w_work.copy_to_device(w)
        advection_schemes.advect_velocity_semi_lagrangian[
            active_tile_shape, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
        ](
            u,
            v,
            w,
            scratch_A_x,
            scratch_A_y,
            scratch_A_z,
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
            fx_const,
            fy_const,
            fz_const,
            has_swirl_nodes,
            swirl_config_device,
            origin_x,
            origin_y,
            origin_z,
            has_turbulence_nodes,
            turbulence_config_device,
            t
        )

        # ------------Velocity swap-------------------
        u, u_work = u_work, u
        v, v_work = v_work, v
        w, w_work = w_work, w

        # ------------Pressure solve-------------------
        extra_pressure = get_source_values(simulations,"extra_pressure",t)

        p = pressure_solve.pressure_poisson_multigrid(
            u, v, w,
            p,
            temperature,
            scratch_A_x,
            dt,
            source_masks,
            extra_pressure,
            delta,
            simulations[0].get("physics").get("fluid").get("density"),
            simulations[0].get("physics").get("temperature").get("expansion_rate"),
            simulations[0].get("physics").get("temperature").get("reference_temperature"),
            scalar_active_tiles_dilated,
            p_levels,
            b_levels,
            delta_levels,
            simulations[0].get("settings").get("iterations"),
            pressure_rhs_partial_sums,
            pressure_rhs_sum,
            zero_levels,
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

        # ------------Scalar update-------------------
        temperature_work.copy_to_device(temperature)
        smoke_work.copy_to_device(smoke)
        fuel_work.copy_to_device(fuel)
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

        # ------------time updated-------------------
        t = t + dt

        # ------------Output-------------------
        device_fields = _current_device_fields(
            u, v, w, p, temperature, smoke, fuel, flame
        )
        while t >= next_output_time:
            output.enqueue_device_output(
                simulations[0],
                writer_slots,
                device_fields,
                output_index,
                t,
            )

            output_index += 1
            output_frame_count += 1
            next_output_time += output_time_step


        # ------------Memory track-------------------
        if output_index == 10:
            ctx = cuda.current_context()
            free, total = ctx.get_memory_info()
            used = total - free
            print(f"VRAM used: {used / 1024**2:.1f} MB")  

    # ------------Shutdown output-------------------
    output.shutdown_output(shared_memory_blocks, writer_slots)

    # ------------Conclusion-------------------
    if cancel_requested:
        print("Simulation cancelled after clean shutdown.")
    else:
        print("Simulation finished!")

    total_runtime = perf_counter() - total_start_time
    print(f"Solver runtime: {total_runtime:.3f} s")
    print("################################################################")