import json
import math
import sys
from time import perf_counter, sleep
from pathlib import Path
import numpy as np
from numba import cuda
import warnings

warnings.filterwarnings("ignore")

import Solver.Kernel_GPU_sparse.Boundary_Conditions.domain_bc as BC
import Solver.Kernel_GPU_sparse.advection_schemes as advection_schemes
import Solver.Kernel_GPU_sparse.scalar_update as scalar_update
import Solver.Kernel_GPU_sparse.pressure_solve as pressure_solve
import Solver.Kernel_GPU_sparse.vorticity as vorticity
import Solver.Kernel_GPU_sparse.kernel_config as kernel_config
import Solver.Kernel_GPU_sparse.Boundary_Conditions.obstacle_bc as obstacle_bc
import Solver.Kernel_GPU_sparse.Boundary_Conditions.source_bc as source_bc
import Solver.Kernel_GPU_sparse.time_step as time_step
import Solver.Kernel_GPU_sparse.update_masks as update_masks
import Solver.Kernel_GPU_sparse.output as output
import Solver.General.forces as forces
import Solver.Kernel_GPU_sparse.sparse_managment as sparse_managment

GPU_FIELD_DTYPE = kernel_config.GPU_FIELD_DTYPE
PROGRESS_EVENT_PREFIX = "__CONTINUUM_FLOW_PROGRESS__ "


def _current_device_fields(u, v, w, p, temperature, smoke, fuel, flame, tile_map):
    """
    Return the currently active device buffers for output export.
    """
    return {
        "u": {
            "data": u,
            "tile_map": tile_map,
            "tile_size": kernel_config.TILE_SIZE,
        },
        "v": {
            "data": v,
            "tile_map": tile_map,
            "tile_size": kernel_config.TILE_SIZE,
        },
        "w": {
            "data": w,
            "tile_map": tile_map,
            "tile_size": kernel_config.TILE_SIZE,
        },
        "pressure": p,
        "temperature": {
            "data": temperature,
            "tile_map": tile_map,
            "tile_size": kernel_config.TILE_SIZE,
        },
        "smoke": {
            "data": smoke,
            "tile_map": tile_map,
            "tile_size": kernel_config.TILE_SIZE,
        },
        "fuel": {
            "data": fuel,
            "tile_map": tile_map,
            "tile_size": kernel_config.TILE_SIZE,
        },
        "flame": {
            "data": flame,
            "tile_map": tile_map,
            "tile_size": kernel_config.TILE_SIZE,
        },
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


def get_source_values(simulation, var_name, t, index=None):
    source_entries = simulation.get("sources") or []
    animation_times = (simulation.get("animation_timeline") or {}).get("times") or ()
    values = np.zeros(len(source_entries), dtype=GPU_FIELD_DTYPE)

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

        values[source_idx] = np.asarray(value, dtype=GPU_FIELD_DTYPE)

    return values


def build_source_noise_fields(source_entries, source_base_masks):
    noise_fields = []

    for source_idx, source_entry in enumerate(source_entries or []):
        noise_amplitude = float(source_entry.get("noise_amplitude", 0.0)) / 100.0
        use_noise = (
            bool(source_entry.get("source_noise", False)) and noise_amplitude > 0.0
        )
        scale_voxels = max(float(source_entry.get("noise_scale", 1.0)), 1.0)
        seed_base = int(source_entry.get("noise_seed", 0))
        source_object_fields = []

        if use_noise and source_idx < len(source_base_masks):
            for object_idx, mask_entry in enumerate(source_base_masks[source_idx]):
                base_mask = np.ascontiguousarray(
                    mask_entry["voxels"]["mask"], dtype=np.bool_
                )
                base_shape = np.asarray(base_mask.shape, dtype=np.int32)
                coarse_shape = np.maximum(
                    1,
                    np.ceil(
                        base_shape.astype(np.float32) / np.float32(scale_voxels)
                    ).astype(np.int32),
                )

                rng = np.random.default_rng(seed_base + object_idx * 1009)
                coarse_noise = rng.uniform(
                    -1.0,
                    1.0,
                    size=tuple(int(v) for v in coarse_shape),
                ).astype(np.float32)

                repeat_x = int(
                    max(1, math.ceil(float(base_shape[0]) / float(coarse_shape[0])))
                )
                repeat_y = int(
                    max(1, math.ceil(float(base_shape[1]) / float(coarse_shape[1])))
                )
                repeat_z = int(
                    max(1, math.ceil(float(base_shape[2]) / float(coarse_shape[2])))
                )
                expanded_noise = np.repeat(
                    np.repeat(
                        np.repeat(coarse_noise, repeat_x, axis=0), repeat_y, axis=1
                    ),
                    repeat_z,
                    axis=2,
                )
                expanded_noise = np.ascontiguousarray(
                    expanded_noise[: base_shape[0], : base_shape[1], : base_shape[2]],
                    dtype=np.float32,
                )
                expanded_noise[~base_mask] = 0.0

                source_object_fields.append(
                    {
                        "mesh_object": mask_entry["mesh_object"],
                        "voxels": mask_entry["voxels"],
                        "values": expanded_noise,
                    }
                )

        noise_fields.append(source_object_fields)

    return noise_fields


def _prepare_force_config_device(config_array, row_width):
    host_array = np.ascontiguousarray(
        np.asarray(config_array, dtype=GPU_FIELD_DTYPE).reshape((-1, row_width))
    )
    return cuda.to_device(host_array)


def apply_all_BC(
    simulation,
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
    source_masks,
    source_noise,
    tile_map,
    u_initial,
    v_initial,
    w_initial,
):
    """
    Apply domain, obstacle and source constraints in the fixed overwrite order.
    Domain BCs are applied always, source and obstalces are optional depending on
    user config.
    """
    bc_config = simulation.get("domain", {}).get("boundary_conditions", {})
    u, v, w, p, T, smoke, fuel = BC.domain_bc(
        u,
        v,
        w,
        p,
        T,
        smoke,
        fuel,
        bc_config,
        tile_map,
        simulation.get("physics").get("temperature").get("reference_temperature"),
        u_initial,
        v_initial,
        w_initial,
    )

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
            tile_map,
        )

    source_count = int(source_masks.shape[0])
    if source_count > 0:
        source_temperature_values = get_source_values(simulation, "temperature", t)
        source_smoke_values = get_source_values(simulation, "smoke", t)
        source_fuel_values = get_source_values(simulation, "fuel", t)
        source_velocity_x_values = get_source_values(simulation, "velocity", t, 0)
        source_velocity_y_values = get_source_values(simulation, "velocity", t, 1)
        source_velocity_z_values = get_source_values(simulation, "velocity", t, 2)
        source_noise_amplitudes = get_source_values(
            simulation, "noise_amplitude", t
        ) / np.float32(100.0)

        for source_idx in range(source_count):
            source_mask_entry = source_masks[source_idx]
            source_noise_entry = source_noise[source_idx]
            blockspergrid = kernel_config.volume_blocks_per_grid(
                source_mask_entry.shape,
                kernel_config.THREADS_PER_BLOCK_3D,
            )
            source_bc.source_bc_kernel[
                blockspergrid, kernel_config.THREADS_PER_BLOCK_3D
            ](
                u,
                v,
                w,
                T,
                smoke,
                fuel,
                tile_map,
                source_mask_entry,
                source_noise_entry,
                source_temperature_values[source_idx],
                source_smoke_values[source_idx],
                source_fuel_values[source_idx],
                source_velocity_x_values[source_idx],
                source_velocity_y_values[source_idx],
                source_velocity_z_values[source_idx],
                source_noise_amplitudes[source_idx],
                dt,
            )
    return u, v, w, p, T, smoke, fuel, flame


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
        delta_levels.append(delta * (2**level))

        nx = (nx + 1) // 2
        ny = (ny + 1) // 2
        nz = (nz + 1) // 2
        level += 1

    return p_levels, b_levels, delta_levels, zero_levels


def solver(
    config,
    obstacle_base_masks,
    obstacle_mask,
    source_base_masks,
    source_masks,
    animated_obstacles,
    animated_sources,
):
    total_start_time = perf_counter()
    simulation = config.get("simulation") or {}
    if not simulation:
        raise ValueError("Solver config must contain a non-empty 'simulation' object.")
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

    # ------------tiles------------------
    tile_i = (nx + kernel_config.TILE_SIZE - 1) // kernel_config.TILE_SIZE
    tile_j = (ny + kernel_config.TILE_SIZE - 1) // kernel_config.TILE_SIZE
    tile_k = (nz + kernel_config.TILE_SIZE - 1) // kernel_config.TILE_SIZE
    tile_shape = (tile_i, tile_j, tile_k)
    total_tile_count = int(np.prod(tile_shape))
    tile_map_values = np.full(tile_shape, -1, dtype=np.int32)
    tile_map = cuda.to_device(tile_map_values)
    base_tile_map = cuda.to_device(np.full(tile_shape, -1, dtype=np.int32))

    next_tile_index_counter = cuda.to_device(np.asarray([0], dtype=np.int32))
    active_tile_counter = cuda.to_device(np.zeros(1, dtype=np.int32))
    spare_tile_growth_size_percent = float(kernel_config.SPARSE_TILE_GROWTH_PERCENT)
    tile_growth_size = max(
        1,
        math.ceil(total_tile_count * (spare_tile_growth_size_percent / 100.0)),
    )

    print("################################################################")
    print("Initialise")
    print("Cell count: ", int(nx * ny * nz))
    print("Tile shape: ", tile_shape)
    print("Total tiles: ", total_tile_count)

    # ------------fields------------------
    # scalars + sparse velocity
    ref_temp = simulation.get("physics").get("temperature").get("reference_temperature")

    sparse_tile_capacity = max(1, tile_growth_size)
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
    u_work = cuda.to_device(
        np.full(sparse_pool_shape, u_initial, dtype=GPU_FIELD_DTYPE)
    )
    v = cuda.to_device(np.full(sparse_pool_shape, v_initial, dtype=GPU_FIELD_DTYPE))
    v_work = cuda.to_device(
        np.full(sparse_pool_shape, v_initial, dtype=GPU_FIELD_DTYPE)
    )
    w = cuda.to_device(np.full(sparse_pool_shape, w_initial, dtype=GPU_FIELD_DTYPE))
    w_work = cuda.to_device(
        np.full(sparse_pool_shape, w_initial, dtype=GPU_FIELD_DTYPE)
    )

    # scalars
    temperature = cuda.to_device(
        np.full(sparse_pool_shape, ref_temp, dtype=GPU_FIELD_DTYPE)
    )
    temperature_work = cuda.to_device(
        np.full(sparse_pool_shape, ref_temp, dtype=GPU_FIELD_DTYPE)
    )
    smoke = cuda.to_device(np.zeros(sparse_pool_shape, dtype=GPU_FIELD_DTYPE))
    smoke_work = cuda.to_device(np.zeros(sparse_pool_shape, dtype=GPU_FIELD_DTYPE))
    fuel = cuda.to_device(np.zeros(sparse_pool_shape, dtype=GPU_FIELD_DTYPE))
    fuel_work = cuda.to_device(np.zeros(sparse_pool_shape, dtype=GPU_FIELD_DTYPE))
    flame = cuda.to_device(np.zeros(sparse_pool_shape, dtype=GPU_FIELD_DTYPE))

    # vorticity
    vorticity_magnitude = cuda.to_device(
        np.zeros(sparse_pool_shape, dtype=GPU_FIELD_DTYPE)
    )

    # scratch
    scratch_A = cuda.to_device(
        np.full(sparse_pool_shape, ref_temp, dtype=GPU_FIELD_DTYPE)
    )
    scratch_B = cuda.to_device(np.zeros(sparse_pool_shape, dtype=GPU_FIELD_DTYPE))
    scratch_C = cuda.to_device(np.zeros(sparse_pool_shape, dtype=GPU_FIELD_DTYPE))

    # --------------- dense -------------------#
    # pressure
    p = cuda.device_array(shape, dtype=GPU_FIELD_DTYPE)
    pressure_rhs = cuda.device_array(shape, dtype=GPU_FIELD_DTYPE)
    pressure_rhs_partial_sums = cuda.device_array(
        kernel_config.MAX_REDUCTION_BLOCKS,
        dtype=np.float32,
    )
    pressure_rhs_sum = cuda.device_array(1, dtype=np.float32)

    # multigrid levels
    p_levels, b_levels, delta_levels, zero_levels = create_multigrid_levels(
        shape,
        delta,
        min_size=8,
    )

    # masks
    obstacle_mask = cuda.to_device(np.ascontiguousarray(obstacle_mask, dtype=np.bool_))
    source_mask_host = (
        np.any(np.stack(source_masks, axis=0), axis=0)
        if source_masks
        else np.zeros(shape, dtype=np.bool_)
    )
    source_mask = cuda.to_device(np.ascontiguousarray(source_mask_host, dtype=np.bool_))
    source_mask_stack = (
        np.ascontiguousarray(np.asarray(source_masks, dtype=np.bool_))
        if source_masks
        else np.zeros((0,) + shape, dtype=np.bool_)
    )
    source_masks = cuda.to_device(source_mask_stack)
    source_noise_base_fields = build_source_noise_fields(
        simulation.get("sources") or [],
        source_base_masks,
    )
    source_noise_host = np.zeros(
        (len(source_noise_base_fields),) + shape, dtype=np.float32
    )
    source_noise = cuda.to_device(
        np.ascontiguousarray(source_noise_host, dtype=np.float32)
    )

    # ------------intitialise------------------
    p.copy_to_device(np.full(shape, 0, dtype=GPU_FIELD_DTYPE))

    velocity_maxima = cuda.to_device(np.zeros(3, dtype=np.float32))
    velocity_maxima_host_zeros = np.zeros(3, dtype=np.float32)

    if source_noise_base_fields:
        update_masks.update_source_values(
            source_noise,
            source_noise_base_fields,
            t,
            delta,
            origin_x,
            origin_y,
            origin_z,
        )

    # ------------output------------------
    output_cfg = ((simulation.get("outputs") or [None])[0]) or {}
    viewer_cfg = ((simulation.get("viewers") or [None])[0]) or {}
    output_time_step = 1.0 / int(output_cfg.get("fps", 24))
    target_realtime_preview = bool(viewer_cfg.get("target_realtime_preview", False))

    shared_memory_blocks, writer_slots = output.setup_output(
        simulation,
        simulation.get("outputs")[0].get("output_path"),
        shape,
    )

    device_fields = _current_device_fields(
        u,
        v,
        w,
        p,
        temperature,
        smoke,
        fuel,
        flame,
        tile_map,
    )

    # ------------time loop------------------
    print("Start time iteration")
    next_output_time = 0.0
    output_index = 0
    time_step_count = 0
    last_output_wall_time = None
    while t < t_max:
        if cancel_flag_path and Path(cancel_flag_path).exists():
            cancel_requested = True
            print("Bake cancellation requested. Stopping the simulation cleanly...")
            break

        # ------------Start Active tiles-------------------
        if simulation.get("settings").get("simulate_sparsely"):
            sparse_managment.build_activity_mask[
                tile_shape, kernel_config.THREADS_PER_BLOCK_3D
            ](
                temperature,
                smoke,
                fuel,
                flame,
                tile_map,
                source_mask,
                base_tile_map,
                simulation.get("settings").get("adaptive_domain_threshold"),
                ref_temp,
            )

            active_tile_counter.copy_to_device(np.zeros(1, dtype=np.int32))

            sparse_managment.dilate_tile_map_persistent[
                tile_shape, kernel_config.THREADS_PER_BLOCK_3D
            ](
                base_tile_map,
                tile_map,
                kernel_config.TILE_DILATE,
                next_tile_index_counter,
                active_tile_counter,
            )

            required_tile_capacity = int(next_tile_index_counter.copy_to_host()[0])

            if required_tile_capacity > sparse_tile_capacity:
                next_sparse_tile_capacity = sparse_managment.required_pool_capacity(
                    sparse_tile_capacity,
                    required_tile_capacity,
                    tile_growth_size,
                )

                temperature = sparse_managment.ensure_pool_capacity(
                    temperature,
                    sparse_tile_capacity,
                    next_sparse_tile_capacity,
                    ref_temp,
                )
                temperature_work = sparse_managment.ensure_pool_capacity(
                    temperature_work,
                    sparse_tile_capacity,
                    next_sparse_tile_capacity,
                    ref_temp,
                )
                smoke = sparse_managment.ensure_pool_capacity(
                    smoke, sparse_tile_capacity, next_sparse_tile_capacity, 0.0
                )
                smoke_work = sparse_managment.ensure_pool_capacity(
                    smoke_work, sparse_tile_capacity, next_sparse_tile_capacity, 0.0
                )
                fuel = sparse_managment.ensure_pool_capacity(
                    fuel, sparse_tile_capacity, next_sparse_tile_capacity, 0.0
                )
                fuel_work = sparse_managment.ensure_pool_capacity(
                    fuel_work, sparse_tile_capacity, next_sparse_tile_capacity, 0.0
                )
                flame = sparse_managment.ensure_pool_capacity(
                    flame, sparse_tile_capacity, next_sparse_tile_capacity, 0.0
                )

                scratch_A = sparse_managment.ensure_pool_capacity(
                    scratch_A,
                    sparse_tile_capacity,
                    next_sparse_tile_capacity,
                    ref_temp,
                )
                scratch_B = sparse_managment.ensure_pool_capacity(
                    scratch_B,
                    sparse_tile_capacity,
                    next_sparse_tile_capacity,
                    0.0,
                )
                scratch_C = sparse_managment.ensure_pool_capacity(
                    scratch_C,
                    sparse_tile_capacity,
                    next_sparse_tile_capacity,
                    0.0,
                )

                u = sparse_managment.ensure_pool_capacity(
                    u,
                    sparse_tile_capacity,
                    next_sparse_tile_capacity,
                    u_initial,
                )
                u_work = sparse_managment.ensure_pool_capacity(
                    u_work,
                    sparse_tile_capacity,
                    next_sparse_tile_capacity,
                    u_initial,
                )
                v = sparse_managment.ensure_pool_capacity(
                    v,
                    sparse_tile_capacity,
                    next_sparse_tile_capacity,
                    v_initial,
                )
                v_work = sparse_managment.ensure_pool_capacity(
                    v_work,
                    sparse_tile_capacity,
                    next_sparse_tile_capacity,
                    v_initial,
                )
                w = sparse_managment.ensure_pool_capacity(
                    w,
                    sparse_tile_capacity,
                    next_sparse_tile_capacity,
                    w_initial,
                )
                w_work = sparse_managment.ensure_pool_capacity(
                    w_work,
                    sparse_tile_capacity,
                    next_sparse_tile_capacity,
                    w_initial,
                )

                zero_pool = sparse_managment.ensure_pool_capacity(
                    zero_pool,
                    sparse_tile_capacity,
                    next_sparse_tile_capacity,
                    0.0,
                )

                vorticity_magnitude = sparse_managment.ensure_pool_capacity(
                    vorticity_magnitude,
                    sparse_tile_capacity,
                    next_sparse_tile_capacity,
                    0.0,
                )

                sparse_tile_capacity = next_sparse_tile_capacity

                print(
                    "Tile buffer grown to:",
                    sparse_tile_capacity,
                    "tiles",
                )

        # ------------time step-------------------
        active_sparse_tile_count = int(active_tile_counter.copy_to_host()[0])

        velocity_maxima.copy_to_device(velocity_maxima_host_zeros)

        dt = time_step.compute_new_timestep_gpu(
            u,
            v,
            w,
            tile_map,
            active_sparse_tile_count,
            velocity_maxima,
            delta,
            cfl,
            output_time_step,
        )

        # ------------Clear scratch-------------------
        sparse_managment.reset_pool(
            scratch_A,
            zero_pool,
            active_sparse_tile_count,
        )
        sparse_managment.reset_pool(
            scratch_B,
            zero_pool,
            active_sparse_tile_count,
        )
        sparse_managment.reset_pool(
            scratch_C,
            zero_pool,
            active_sparse_tile_count,
        )

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
            if source_noise_base_fields:
                update_masks.update_source_values(
                    source_noise,
                    source_noise_base_fields,
                    t,
                    delta,
                    origin_x,
                    origin_y,
                    origin_z,
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
                scratch_A,
                scratch_B,
                scratch_C,
                tile_map,
            )

        # ------------BC-------------------
        u, v, w, p, temperature, smoke, fuel, flame = apply_all_BC(
            simulation,
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
            scratch_A,
            scratch_B,
            scratch_C,
            source_masks,
            source_noise,
            tile_map,
            u_initial,
            v_initial,
            w_initial,
        )

        # ------------Clear scratch-------------------
        sparse_managment.reset_pool(
            scratch_A,
            zero_pool,
            active_sparse_tile_count,
        )
        sparse_managment.reset_pool(
            scratch_B,
            zero_pool,
            active_sparse_tile_count,
        )
        sparse_managment.reset_pool(
            scratch_C,
            zero_pool,
            active_sparse_tile_count,
        )

        # ------------Vorticity-------------------
        if simulation.get("physics").get("extras").get("vorticity") > 0.0:
            vorticity.compute_vorticity[tile_shape, kernel_config.THREADS_PER_BLOCK_3D](
                u,
                v,
                w,
                obstacle_mask,
                vorticity_magnitude,
                delta,
                tile_map,
                u_initial,
                v_initial,
                w_initial,
                nx,
                ny,
                nz,
            )

        # ------------force params-------------------
        fx_const, fy_const, fz_const = forces.constant_force(simulation, t)
        swirl_config, has_swirl_nodes = forces.swirl_force(simulation, t)
        turbulence_config, has_turbulence_nodes = forces.turbulence_force(simulation, t)
        swirl_config_device = _prepare_force_config_device(swirl_config, 8)
        turbulence_config_device = _prepare_force_config_device(turbulence_config, 4)

        # ------------Velocity update-------------------
        sparse_managment.copy_pool(
            u_work,
            u,
            active_sparse_tile_count,
        )
        sparse_managment.copy_pool(
            v_work,
            v,
            active_sparse_tile_count,
        )
        sparse_managment.copy_pool(
            w_work,
            w,
            active_sparse_tile_count,
        )

        advection_schemes.advect_velocity_semi_lagrangian[
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

        advection_schemes.update_velocity_maccormack[
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
            ref_temp,
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
        noise_amplitudes = get_source_values(
            simulation, "noise_amplitude", t
        ) / np.float32(100.0)

        p = pressure_solve.pressure_poisson_multigrid(
            u,
            v,
            w,
            p,
            temperature,
            pressure_rhs,
            dt,
            source_masks,
            extra_pressure,
            source_noise,
            noise_amplitudes,
            delta,
            simulation.get("physics").get("fluid").get("density"),
            simulation.get("physics").get("temperature").get("expansion_rate"),
            ref_temp,
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
        )

        # ------------Scalar update-------------------
        sparse_managment.copy_pool(
            temperature_work,
            temperature,
            active_sparse_tile_count,
        )
        sparse_managment.copy_pool(
            smoke_work,
            smoke,
            active_sparse_tile_count,
        )
        sparse_managment.copy_pool(
            fuel_work,
            fuel,
            active_sparse_tile_count,
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
            ref_temp,
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
            ref_temp,
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
        device_fields = _current_device_fields(
            u, v, w, p, temperature, smoke, fuel, flame, tile_map
        )
        while t >= next_output_time:
            if target_realtime_preview and last_output_wall_time is not None:
                elapsed_since_last_output = perf_counter() - last_output_wall_time
                remaining_time = output_time_step - elapsed_since_last_output
                if remaining_time > 0.0:
                    sleep(remaining_time)

            output.enqueue_device_output(
                simulation,
                writer_slots,
                device_fields,
                output_index,
                t,
            )

            last_output_wall_time = perf_counter()

            output_index += 1
            next_output_time += output_time_step

        # ------------Memory track-------------------
        if time_step_count % 30 == 0:
            active_tile_count = int(active_tile_counter.copy_to_host()[0])
            print(f"Active tiles: {active_tile_count} / ", total_tile_count)

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
