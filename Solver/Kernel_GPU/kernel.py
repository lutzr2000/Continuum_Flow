import json
import math
import sys
from time import perf_counter
from pathlib import Path
import numpy as np
from numba import cuda
import warnings

warnings.filterwarnings("ignore")

import Solver.Kernel_GPU.Boundary_Conditions.domain_bc as BC
import Solver.Kernel_GPU.velocity_update as velocity_update
import Solver.Kernel_GPU.scalar_update as scalar_update
import Solver.Kernel_GPU.pressure_solve as pressure_solve
import Solver.Kernel_GPU.vorticity as vorticity
import Solver.Kernel_GPU.kernel_config as kernel_config
import Solver.Kernel_GPU.Boundary_Conditions.obstacle_bc as obstacle_bc
import Solver.Kernel_GPU.Boundary_Conditions.source_bc as source_bc
import Solver.Kernel_GPU.time_step as time_step
import Solver.Kernel_GPU.update_masks as update_masks
import Solver.Kernel_GPU.voxelise_mesh as voxelise_mesh
import Solver.Kernel_GPU.output as output
import Solver.Kernel_GPU.multigrid as multigrid
import Solver.General.forces as forces
import Solver.Kernel_GPU.sparse_managment as sparse_managment

GPU_FIELD_DTYPE = kernel_config.GPU_FIELD_DTYPE


def get_source_values(simulation, var_name, t, index=None, dtype=GPU_FIELD_DTYPE):
    source_entries = simulation.get("sources") or []
    animation_times = (simulation.get("animation_timeline") or {}).get("times") or ()
    values = np.zeros(len(source_entries), dtype=dtype)

    for source_idx, source_entry in enumerate(source_entries):
        value = source_entry.get(var_name, 0.0)

        animation_entry = (source_entry.get("animations") or {}).get(var_name) or {}
        animation_values = animation_entry.get("values") or ()
        sample_count = min(len(animation_times), len(animation_values))

        if sample_count > 0:
            nearest_time_idx = min(
                range(sample_count),
                key=lambda idx: abs(float(animation_times[idx]) - float(t)),
            )
            value = animation_values[nearest_time_idx]

        if index is not None:
            if value is None:
                value = 0.0
            else:
                value = value[index]

        values[source_idx] = np.asarray(value, dtype=dtype)

    return values


def compute_inital_velocity(simulation_cfg):
    total_u = 0.0
    total_v = 0.0
    total_w = 0.0
    inlet_count = 0

    for face_cfg in (
        (simulation_cfg.get("domain") or {}).get("boundary_conditions", {}).values()
    ):
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


def is_animated(base_masks):
    for entry in base_masks:
        mesh_object = entry["mesh_object"]
        animation = mesh_object.get("transform_animation") or {}

        if len(animation.get("matrices_world") or ()) > 1:
            return True

    return False


def solver(
    config: dict,
) -> None:
    r"""
    Run one simulation.

    The solver initializes the tile pools, uploads static and animated
    masks to the GPU, advances the simulation in time, and writes output frames
    through the host-side VDB writer pipeline.

    The general time loop does the following steps:

    1. Update the active tile map from the current scalar fields and
       source activity.
    2. Grow pool capacity if the persistent tile map requires more tile
       slots.
    3. Compute the new timestep from the current velocity field using the user
       described CFL-number.
    4. Reset scratch pools for the upcoming operations.
    5. Refresh animated source masks, source noise fields, and obstacle masks
       if needed.
    6. Apply domain, obstacle, and source boundary conditions.
    7. Reset scratch pools again before physics updates.
    8. Compute vorticity magnitude when vorticity confinement is enabled.
    9. Build force parameters for constant, swirl, and turbulence forces.
    10. Copy the current velocity pools into work buffers.
    11. Advect and update velocity using MacCormack.
    12. Solve the pressure Poisson equation using multigrid.
    13. Project the velocity using the computed pressure.
    14. Advect and update scalar fields using MacCormack. Compute burn behavior.
    15. Advance simulation time and emit output frames whenever an output time
        step is reached.
    16. Periodically report active tile count and GPU memory usage.

    After the loop finishes, the writer pipeline is flushed and shut down
    cleanly.

    Parameters
    ----------
    config
        Full solver configuration containing simulation settings, domain
        dimensions, physics parameters, source definitions, output settings,
        and optional meta information such as the cancellation flag path.

    Returns
    -------
    None
        The simulation is executed in-place on the GPU and output frames are
        written to the configured VDB output directory.
    """
    total_start_time = perf_counter()
    simulation = config.get("simulation") or {}
    cancel_flag_path = (
        (config.get("meta") or {}).get("cancel_flag_path") or ""
    ).strip()
    cancel_requested = False

    # ------------time-------------------
    t = 0.0
    cfl = float(simulation.get("settings", {}).get("cfl", 10.0))
    t_max = simulation.get("settings").get("simulation_length")

    # ------------dimensions------------------
    delta = simulation.get("domain").get("resolution")
    nx = simulation["domain"]["grid"]["nx"]
    ny = simulation["domain"]["grid"]["ny"]
    nz = simulation["domain"]["grid"]["nz"]
    shape = (nx, ny, nz)

    origin_x = -0.5 * nx * delta
    origin_y = -0.5 * ny * delta
    origin_z = 0.0
    origin = (origin_x, origin_y, origin_z)

    simulate_sparsely = bool(simulation.get("settings").get("simulate_sparsely"))
    sparse_threshold = simulation.get("settings").get("adaptive_domain_threshold")

    # ------------physics------------------
    reference_temperature = (
        simulation.get("physics").get("temperature").get("reference_temperature")
    )

    # ------------tiles------------------
    tile_size_i = (nx + kernel_config.TILE_SIZE - 1) // kernel_config.TILE_SIZE
    tile_size_j = (ny + kernel_config.TILE_SIZE - 1) // kernel_config.TILE_SIZE
    tile_size_k = (nz + kernel_config.TILE_SIZE - 1) // kernel_config.TILE_SIZE
    tile_shape = (tile_size_i, tile_size_j, tile_size_k)
    total_tile_count = int(tile_size_i * tile_size_j * tile_size_k)

    if simulate_sparsely:
        tile_map_values = np.full(tile_shape, -1, dtype=np.int32)
        base_tile_map_values = np.full(tile_shape, -1, dtype=np.int32)
        initial_next_tile_index = 0
        initial_active_tile_count = 0
    else:
        tile_map_values = np.arange(total_tile_count, dtype=np.int32).reshape(
            tile_shape
        )
        base_tile_map_values = np.ones(tile_shape, dtype=np.int32)
        initial_next_tile_index = total_tile_count
        initial_active_tile_count = total_tile_count

    tile_map = cuda.to_device(tile_map_values)
    base_tile_map = cuda.to_device(base_tile_map_values)
    free_slot_stack = cuda.to_device(np.full(total_tile_count, -1, dtype=np.int32))
    free_slot_count = cuda.to_device(np.zeros(1, dtype=np.int32))
    reused_slot_stack = cuda.to_device(np.full(total_tile_count, -1, dtype=np.int32))
    reused_slot_count = cuda.to_device(np.zeros(1, dtype=np.int32))

    next_tile_index_counter = cuda.to_device(
        np.asarray([initial_next_tile_index], dtype=np.int32)
    )
    active_tile_counter = cuda.to_device(
        np.asarray([initial_active_tile_count], dtype=np.int32)
    )
    tile_growth_size = max(
        1,
        math.ceil(
            total_tile_count * (float(kernel_config.SPARSE_TILE_GROWTH_PERCENT) / 100.0)
        ),
    )

    print("################################################################")
    print("Initialise")
    print("Total tiles: ", total_tile_count)
    print("Maximum number of cells: ", total_tile_count * kernel_config.TILE_SIZE**3)

    # ------------fields------------------
    # --------------- sparse -------------------#
    sparse_tile_capacity = (
        total_tile_count if not simulate_sparsely else max(1, tile_growth_size)
    )
    sparse_pool_shape = (
        sparse_tile_capacity,
        kernel_config.TILE_SIZE,
        kernel_config.TILE_SIZE,
        kernel_config.TILE_SIZE,
    )

    zero_pool = cuda.to_device(np.zeros(sparse_pool_shape, dtype=GPU_FIELD_DTYPE))

    # velocity
    u_initial, v_initial, w_initial = compute_inital_velocity(simulation)

    u = cuda.to_device(np.full(sparse_pool_shape, u_initial, dtype=GPU_FIELD_DTYPE))
    v = cuda.to_device(np.full(sparse_pool_shape, v_initial, dtype=GPU_FIELD_DTYPE))
    w = cuda.to_device(np.full(sparse_pool_shape, w_initial, dtype=GPU_FIELD_DTYPE))

    u_work = cuda.to_device(
        np.full(sparse_pool_shape, u_initial, dtype=GPU_FIELD_DTYPE)
    )
    v_work = cuda.to_device(
        np.full(sparse_pool_shape, v_initial, dtype=GPU_FIELD_DTYPE)
    )
    w_work = cuda.to_device(
        np.full(sparse_pool_shape, w_initial, dtype=GPU_FIELD_DTYPE)
    )

    velocity_maxima = cuda.to_device(np.zeros(3, dtype=GPU_FIELD_DTYPE))

    # scalars
    temperature = cuda.to_device(
        np.full(sparse_pool_shape, reference_temperature, dtype=GPU_FIELD_DTYPE)
    )
    smoke = cuda.to_device(np.zeros(sparse_pool_shape, dtype=GPU_FIELD_DTYPE))
    fuel = cuda.to_device(np.zeros(sparse_pool_shape, dtype=GPU_FIELD_DTYPE))

    temperature_work = cuda.to_device(
        np.full(sparse_pool_shape, reference_temperature, dtype=GPU_FIELD_DTYPE)
    )
    smoke_work = cuda.to_device(np.zeros(sparse_pool_shape, dtype=GPU_FIELD_DTYPE))
    fuel_work = cuda.to_device(np.zeros(sparse_pool_shape, dtype=GPU_FIELD_DTYPE))

    # flame
    flame = cuda.to_device(np.zeros(sparse_pool_shape, dtype=GPU_FIELD_DTYPE))

    # vorticity
    vorticity_magnitude = cuda.to_device(
        np.zeros(sparse_pool_shape, dtype=GPU_FIELD_DTYPE)
    )

    # scratch
    scratch_A = cuda.to_device(
        np.full(sparse_pool_shape, reference_temperature, dtype=GPU_FIELD_DTYPE)
    )
    scratch_B = cuda.to_device(np.zeros(sparse_pool_shape, dtype=GPU_FIELD_DTYPE))
    scratch_C = cuda.to_device(np.zeros(sparse_pool_shape, dtype=GPU_FIELD_DTYPE))

    # pressure
    p = cuda.to_device(np.zeros(sparse_pool_shape, dtype=GPU_FIELD_DTYPE))
    pressure_rhs = cuda.to_device(np.zeros(sparse_pool_shape, dtype=GPU_FIELD_DTYPE))
    pressure_rhs_partial_sums = cuda.device_array(
        kernel_config.MAX_REDUCTION_BLOCKS,
        dtype=GPU_FIELD_DTYPE,
    )
    pressure_rhs_sum = cuda.device_array(1, dtype=GPU_FIELD_DTYPE)

    # masks
    bake_path = simulation["outputs"][0]["output_path"]

    sources = simulation.get("sources") or []
    obstacles = simulation.get("obstacles") or []
    # add times
    for entry in sources + obstacles:
        for obj in entry.get("geometry_inputs") or []:
            obj["animation_timeline"] = simulation["animation_timeline"]

    source_base_masks = [
        voxelise_mesh.voxelise_all_meshes(
            delta,
            source.get("geometry_inputs"),
            bake_path,
        )
        for source in sources
    ]

    obstacle_base_masks = [
        mask_entry
        for obstacle in obstacles
        for mask_entry in voxelise_mesh.voxelise_all_meshes(
            delta,
            obstacle.get("geometry_inputs"),
            bake_path,
        )
    ]

    source_masks = [
        cuda.to_device(np.zeros(sparse_pool_shape, dtype=np.bool_)) for _ in sources
    ]

    source_tile_mask = cuda.to_device(
        np.zeros(tile_shape, dtype=np.bool_)
    )  # this one is needed for detemining which tiles are active due to source activity

    obstacle_mask = cuda.to_device(np.zeros(sparse_pool_shape, dtype=np.bool_))

    # multigrid levels
    p_levels, b_levels, delta_levels, zero_levels = multigrid.create_multigrid_levels(
        shape,
        delta,
        min_size=8,
    )

    # ------------output------------------
    output_cfg = ((simulation.get("outputs") or [None])[0]) or {}
    output_time_step = 1.0 / int(output_cfg.get("fps", 24))

    shared_memory_blocks, writer_slots = output.setup_output(
        simulation,
        bake_path,
        shape,
    )

    # ------------time loop------------------
    print("Start time iteration")
    next_output_time = 0.0
    output_index = 0
    time_step_count = 0
    active_tile_counter_host = initial_active_tile_count
    next_tile_index_counter_host = initial_next_tile_index

    while t < t_max:
        if cancel_flag_path and Path(cancel_flag_path).exists():
            cancel_requested = True
            print("Bake cancellation requested. Stopping the simulation cleanly...")
            break

        # ------------Clear scratch-------------------
        sparse_managment.reset_pools(
            (scratch_A, scratch_B, scratch_C),
            zero_pool,
            next_tile_index_counter_host,
        )

        # ------------Update source tile mask-------------------
        update_masks.update_source_tile_mask(
            source_tile_mask,
            source_base_masks,
            t,
            delta,
            origin,
        )

        # ------------Start Active tiles-------------------
        if simulate_sparsely:
            sparse_managment.build_activity_mask[
                tile_shape, kernel_config.THREADS_PER_BLOCK_3D
            ](
                smoke,
                fuel,
                flame,
                tile_map,
                source_tile_mask,
                base_tile_map,
                sparse_threshold,
                nx,
                ny,
                nz,
            )

            active_tile_counter.copy_to_device(np.zeros(1, dtype=np.int32))
            free_slot_count.copy_to_device(np.zeros(1, dtype=np.int32))
            reused_slot_count.copy_to_device(np.zeros(1, dtype=np.int32))

            sparse_managment.release_inactive_tile_slots[
                tile_shape, kernel_config.THREADS_PER_BLOCK_3D
            ](
                base_tile_map,
                tile_map,
                kernel_config.TILE_DILATE,
                free_slot_stack,
                free_slot_count,
            )

            sparse_managment.activate_tiles_with_reuse[
                tile_shape, kernel_config.THREADS_PER_BLOCK_3D
            ](
                base_tile_map,
                tile_map,
                kernel_config.TILE_DILATE,
                free_slot_stack,
                free_slot_count,
                reused_slot_stack,
                reused_slot_count,
                next_tile_index_counter,
                active_tile_counter,
            )

            reused_slot_count_host = int(reused_slot_count.copy_to_host()[0])
            if reused_slot_count_host > 0:
                sparse_managment.reset_reused_pool_slots(
                    [
                        (u, u_initial),
                        (v, v_initial),
                        (w, w_initial),
                        (u_work, u_initial),
                        (v_work, v_initial),
                        (w_work, w_initial),
                        (scratch_A, reference_temperature),
                        (scratch_B, 0.0),
                        (scratch_C, 0.0),
                        (p, 0.0),
                        (pressure_rhs, 0.0),
                        (temperature, reference_temperature),
                        (smoke, 0.0),
                        (fuel, 0.0),
                        (temperature_work, reference_temperature),
                        (smoke_work, 0.0),
                        (fuel_work, 0.0),
                        (flame, 0.0),
                        (vorticity_magnitude, 0.0),
                        (obstacle_mask, False),
                        *[(mask, False) for mask in source_masks],
                    ],
                    reused_slot_stack,
                    reused_slot_count_host,
                )

            next_tile_index_counter_host = int(
                next_tile_index_counter.copy_to_host()[0]
            )

            if next_tile_index_counter_host > sparse_tile_capacity:
                next_sparse_tile_capacity = sparse_managment.required_pool_capacity(
                    sparse_tile_capacity,
                    next_tile_index_counter_host,
                    tile_growth_size,
                )

                (
                    u,
                    v,
                    w,
                    u_work,
                    v_work,
                    w_work,
                    scratch_A,
                    scratch_B,
                    scratch_C,
                    p,
                    pressure_rhs,
                    temperature,
                    smoke,
                    fuel,
                    temperature_work,
                    smoke_work,
                    fuel_work,
                    flame,
                    zero_pool,
                    vorticity_magnitude,
                    obstacle_mask,
                    *source_masks,
                ) = sparse_managment.ensure_pool_capacities(
                    [
                        (u, u_initial),
                        (v, v_initial),
                        (w, w_initial),
                        (u_work, u_initial),
                        (v_work, v_initial),
                        (w_work, w_initial),
                        (scratch_A, reference_temperature),
                        (scratch_B, 0.0),
                        (scratch_C, 0.0),
                        (p, 0.0),
                        (pressure_rhs, 0.0),
                        (temperature, reference_temperature),
                        (smoke, 0.0),
                        (fuel, 0.0),
                        (temperature_work, reference_temperature),
                        (smoke_work, 0.0),
                        (fuel_work, 0.0),
                        (flame, 0.0),
                        (zero_pool, 0.0),
                        (vorticity_magnitude, 0.0),
                        (obstacle_mask, False),
                        *[(mask, False) for mask in source_masks],
                    ],
                    sparse_tile_capacity,
                    next_sparse_tile_capacity,
                )

                sparse_tile_capacity = next_sparse_tile_capacity

                print(
                    "Tile buffer grown to:",
                    sparse_tile_capacity,
                    "tiles",
                )

            active_tile_counter_host = int(active_tile_counter.copy_to_host()[0])
        else:
            active_tile_counter_host = total_tile_count
            next_tile_index_counter_host = total_tile_count

        # ------------Update masks-------------------
        update_masks.update_source_masks(
            source_masks,
            source_base_masks,
            t,
            delta,
            origin_x,
            origin_y,
            origin_z,
            tile_map,
        )

        update_masks.update_obstacle_mask(
            obstacle_mask,
            obstacle_base_masks,
            t,
            delta,
            origin_x,
            origin_y,
            origin_z,
            tile_map,
            scratch_A,
            scratch_B,
            scratch_C,
        )

        # ------------time step-------------------
        velocity_maxima.copy_to_device(np.zeros(3, dtype=GPU_FIELD_DTYPE))

        dt = time_step.compute_new_timestep_gpu(
            u,
            v,
            w,
            tile_map,
            active_tile_counter_host,
            velocity_maxima,
            delta,
            cfl,
            output_time_step,
        )

        # ------------BCs-------------------
        # ------------Domain BC-------------------
        bc_config = simulation.get("domain", {}).get("boundary_conditions", {})

        u, v, w, p, temperature, smoke, fuel = BC.domain_bc(
            u,
            v,
            w,
            p,
            temperature,
            smoke,
            fuel,
            bc_config,
            tile_map,
            reference_temperature,
            u_initial,
            v_initial,
            w_initial,
            nx,
            ny,
            nz,
        )

        # ------------Obstacle BC-------------------
        obstacle_bc.obstacle_bc[
            tile_shape,
            kernel_config.THREADS_PER_BLOCK_3D,
        ](
            u,
            v,
            w,
            smoke,
            fuel,
            flame,
            obstacle_mask,
            scratch_A,
            scratch_B,
            scratch_C,
            tile_map,
        )

        # ------------Source BC-------------------
        source_temperature_values = get_source_values(simulation, "temperature", t)
        source_smoke_values = get_source_values(simulation, "smoke", t)
        source_fuel_values = get_source_values(simulation, "fuel", t)
        source_noise_scales = get_source_values(
            simulation,
            "noise_scale",
            t,
        )

        source_noise_amplitudes = (
            get_source_values(
                simulation,
                "noise_amplitude",
                t,
            )
            / 100.0
        )

        source_noise_seeds = get_source_values(
            simulation,
            "noise_seed",
            t,
            dtype=np.int32,
        )
        source_noise_enabled = get_source_values(
            simulation, "source_noise", t, dtype=np.bool_
        )
        source_noise_amplitudes[~source_noise_enabled] = 0.0

        source_velocity_x_values = get_source_values(
            simulation,
            "velocity",
            t,
            0,
        )
        source_velocity_y_values = get_source_values(
            simulation,
            "velocity",
            t,
            1,
        )
        source_velocity_z_values = get_source_values(
            simulation,
            "velocity",
            t,
            2,
        )

        for source_idx, source_mask in enumerate(source_masks):
            source_bc.source_bc[
                tile_shape,
                kernel_config.THREADS_PER_BLOCK_3D,
            ](
                u,
                v,
                w,
                temperature,
                smoke,
                fuel,
                tile_map,
                source_mask,
                source_temperature_values[source_idx],
                source_smoke_values[source_idx],
                source_fuel_values[source_idx],
                source_velocity_x_values[source_idx],
                source_velocity_y_values[source_idx],
                source_velocity_z_values[source_idx],
                source_noise_scales[source_idx],
                source_noise_amplitudes[source_idx],
                source_noise_seeds[source_idx],
                dt,
            )

        # ------------Clear scratch-------------------
        sparse_managment.reset_pools(
            (scratch_A, scratch_B, scratch_C),
            zero_pool,
            next_tile_index_counter_host,
        )

        # ------------Vorticity-------------------
        if simulation.get("physics").get("extras").get("vorticity") > 0.0:
            vorticity.compute_vorticity[tile_shape, kernel_config.THREADS_PER_BLOCK_3D](
                u,
                v,
                w,
                u_initial,
                v_initial,
                w_initial,
                obstacle_mask,
                vorticity_magnitude,
                delta,
                tile_map,
                nx,
                ny,
                nz,
            )

        # ------------force params-------------------
        fx_const, fy_const, fz_const = forces.constant_force(simulation, t)
        swirl_config, has_swirl_nodes = forces.swirl_force(simulation, t)
        turbulence_config, has_turbulence_nodes = forces.turbulence_force(simulation, t)
        swirl_config_device = cuda.to_device(
            np.ascontiguousarray(
                np.asarray(swirl_config, dtype=GPU_FIELD_DTYPE).reshape((-1, 8))
            )
        )
        turbulence_config_device = cuda.to_device(
            np.ascontiguousarray(
                np.asarray(turbulence_config, dtype=GPU_FIELD_DTYPE).reshape((-1, 4))
            )
        )

        # ------------Velocity update-------------------
        sparse_managment.copy_pools(
            (
                (u_work, u),
                (v_work, v),
                (w_work, w),
            ),
            next_tile_index_counter_host,
        )

        velocity_update.advect_velocity_semi_lagrangian[
            tile_shape, kernel_config.THREADS_PER_BLOCK_3D
        ](
            u,
            v,
            w,
            scratch_A,
            scratch_B,
            scratch_C,
            dt,
            delta,
            tile_map,
            u_initial,
            v_initial,
            w_initial,
            nx,
            ny,
            nz,
        )

        velocity_update.update_velocity_maccormack[
            tile_shape, kernel_config.THREADS_PER_BLOCK_3D
        ](
            u,
            v,
            w,
            obstacle_mask,
            scratch_A,
            scratch_B,
            scratch_C,
            dt,
            u_work,
            v_work,
            w_work,
            delta,
            simulation.get("physics").get("fluid").get("density"),
            simulation.get("physics").get("fluid").get("viscosity"),
            vorticity_magnitude,
            simulation.get("physics").get("extras").get("vorticity"),
            temperature,
            simulation.get("physics", {}).get("temperature", {}).get("buoyancy"),
            reference_temperature,
            tile_map,
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
            t,
            u_initial,
            v_initial,
            w_initial,
            nx,
            ny,
            nz,
        )

        # ------------Velocity swap-------------------
        u, u_work = u_work, u
        v, v_work = v_work, v
        w, w_work = w_work, w

        # ------------Pressure solve-------------------
        extra_pressure = get_source_values(simulation, "extra_pressure", t)

        p = pressure_solve.pressure_poisson_multigrid(
            u,
            v,
            w,
            p,
            temperature,
            pressure_rhs,
            dt,
            source_masks,
            source_noise_scales,
            source_noise_amplitudes,
            source_noise_seeds,
            extra_pressure,
            delta,
            simulation.get("physics").get("fluid").get("density"),
            simulation.get("physics").get("temperature").get("expansion_rate"),
            reference_temperature,
            tile_map,
            tile_shape,
            u_initial,
            v_initial,
            w_initial,
            p_levels,
            b_levels,
            delta_levels,
            simulation.get("settings").get("iterations"),
            pressure_rhs_partial_sums,
            pressure_rhs_sum,
            zero_levels,
            nx,
            ny,
            nz,
        )

        # ------------Velocity projection-------------------
        pressure_solve.project_velocity_kernel[
            tile_shape, kernel_config.THREADS_PER_BLOCK_3D
        ](
            u,
            v,
            w,
            p,
            obstacle_mask,
            dt,
            delta,
            simulation.get("physics").get("fluid").get("density"),
            tile_map,
            nx,
            ny,
            nz,
        )

        # ------------Scalar update-------------------
        sparse_managment.copy_pools(
            (
                (u_work, u),
                (v_work, v),
                (w_work, w),
            ),
            next_tile_index_counter_host,
        )

        scalar_update.predict_scalar_fields_semi_lagrangian[
            tile_shape, kernel_config.THREADS_PER_BLOCK_3D
        ](
            temperature,
            smoke,
            fuel,
            u,
            v,
            w,
            dt,
            scratch_A,
            scratch_B,
            scratch_C,
            delta,
            reference_temperature,
            tile_map,
            u_initial,
            v_initial,
            w_initial,
            nx,
            ny,
            nz,
        )

        scalar_update.update_scalar_fields_maccormack[
            tile_shape, kernel_config.THREADS_PER_BLOCK_3D
        ](
            temperature,
            smoke,
            fuel,
            scratch_A,
            scratch_B,
            scratch_C,
            u,
            v,
            w,
            dt,
            temperature_work,
            smoke_work,
            fuel_work,
            flame,
            delta,
            simulation.get("physics").get("temperature").get("dissipation"),
            simulation.get("physics").get("temperature").get("production_rate"),
            simulation.get("physics").get("smoke").get("dissipation"),
            simulation.get("physics").get("smoke").get("production_rate"),
            simulation.get("physics").get("fuel").get("dissipation"),
            simulation.get("physics").get("fuel").get("burn_rate"),
            simulation.get("physics").get("fuel").get("ignition_temperature"),
            simulation.get("physics").get("burning").get("scale"),
            simulation.get("physics").get("burning").get("amplitude"),
            reference_temperature,
            tile_map,
            u_initial,
            v_initial,
            w_initial,
            nx,
            ny,
            nz,
        )

        # ------------Swap-------------------
        temperature, temperature_work = temperature_work, temperature
        smoke, smoke_work = smoke_work, smoke
        fuel, fuel_work = fuel_work, fuel

        # ------------time updated-------------------
        t = t + dt
        time_step_count += 1

        # ------------Output-------------------
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
        while t >= next_output_time:
            output.enqueue_device_output(
                simulation,
                writer_slots,
                device_fields,
                tile_map,
                kernel_config.TILE_SIZE,
                active_tile_counter_host,
                next_tile_index_counter_host,
                output_index,
                t,
            )

            output_index += 1
            next_output_time += output_time_step

        # ------------Memory track-------------------
        if time_step_count % 30 == 0:
            print(
                f"Active cells: {active_tile_counter_host*kernel_config.TILE_SIZE**3} / ",
                total_tile_count * kernel_config.TILE_SIZE**3,
            )

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
