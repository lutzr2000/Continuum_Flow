import json
import math
import sys
from time import perf_counter, sleep
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


def get_source_values(simulation, var_name, t, index=None):
    source_entries = simulation.get("sources") or []
    animation_times = (
        (simulation.get("animation_timeline") or {}).get("times") or ()
    )
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
        use_noise = bool(source_entry.get("source_noise", False)) and noise_amplitude > 0.0
        scale_voxels = max(float(source_entry.get("noise_scale", 1.0)), 1.0)
        seed_base = int(source_entry.get("noise_seed", 0))
        source_object_fields = []

        if use_noise and source_idx < len(source_base_masks):
            for object_idx, mask_entry in enumerate(source_base_masks[source_idx]):
                base_mask = np.ascontiguousarray(mask_entry["voxels"]["mask"], dtype=np.bool_)
                base_shape = np.asarray(base_mask.shape, dtype=np.int32)
                coarse_shape = np.maximum(
                    1,
                    np.ceil(base_shape.astype(np.float32) / np.float32(scale_voxels)).astype(np.int32),
                )

                rng = np.random.default_rng(seed_base + object_idx * 1009)
                coarse_noise = rng.uniform(
                    -1.0,
                    1.0,
                    size=tuple(int(v) for v in coarse_shape),
                ).astype(np.float32)

                repeat_x = int(max(1, math.ceil(float(base_shape[0]) / float(coarse_shape[0]))))
                repeat_y = int(max(1, math.ceil(float(base_shape[1]) / float(coarse_shape[1]))))
                repeat_z = int(max(1, math.ceil(float(base_shape[2]) / float(coarse_shape[2]))))
                expanded_noise = np.repeat(
                    np.repeat(np.repeat(coarse_noise, repeat_x, axis=0), repeat_y, axis=1),
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
    source_noise
):
    """
    Apply domain, obstacle and source constraints in the fixed overwrite order.
    Domain BCs are applied always, source and obstalces are optional depending on
    user config.
    """
    bc_config = simulation.get("domain", {}).get("boundary_conditions", {})
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
        source_temperature_values = get_source_values(simulation,"temperature",t)
        source_smoke_values = get_source_values(simulation,"smoke",t)
        source_fuel_values = get_source_values(simulation,"fuel",t)
        source_velocity_x_values = get_source_values(simulation, "velocity", t, 0)
        source_velocity_y_values = get_source_values(simulation, "velocity", t, 1)
        source_velocity_z_values = get_source_values(simulation, "velocity", t, 2)
        source_noise_amplitudes = get_source_values(simulation, "noise_amplitude", t) / np.float32(100.0)

        for source_idx in range(source_count):
            source_mask_entry = source_masks[source_idx]
            source_noise_entry = source_noise[source_idx]
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
                source_noise_entry,
                source_temperature_values[source_idx],
                source_smoke_values[source_idx],
                source_fuel_values[source_idx],
                source_velocity_x_values[source_idx],
                source_velocity_y_values[source_idx],
                source_velocity_z_values[source_idx],
                source_noise_amplitudes[source_idx],
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
    simulation = config.get("simulation") or {}
    if not simulation:
        raise ValueError("Solver config must contain a non-empty 'simulation' object.")
    cancel_flag_path = ((config.get("meta") or {}).get("cancel_flag_path") or "").strip()
    cancel_requested = False
    profiling_stats = {}

    def _record_timing(name, elapsed):
        stat = profiling_stats.setdefault(name, {"count": 0, "total": 0.0})
        stat["count"] += 1
        stat["total"] += elapsed

    def _profile_call(name, func, *args, cuda_sync=False, **kwargs):
        if cuda_sync:
            cuda.synchronize()
        start_time = perf_counter()
        result = func(*args, **kwargs)
        if cuda_sync:
            cuda.synchronize()
        _record_timing(name, perf_counter() - start_time)
        return result

    init_start_time = perf_counter()

    # ------------time-------------------
    t = 0
    cfl = float(simulation.get("settings", {}).get("cfl", 10.0))
    t_max = simulation.get("settings").get("simulation_length")

    # ------------dimensions------------------
    delta = simulation.get("domain").get("resolution")
    nx = simulation["domain"]["grid"]["nx"]
    ny = simulation["domain"]["grid"]["ny"]
    nz = simulation["domain"]["grid"]["nz"]
    shape = (nx,ny,nz)

    origin_x = -0.5 * nx * delta
    origin_y = -0.5 * ny * delta
    origin_z = 0.0

    print("################################################################")
    print("Initialise")
    print("Cell count: ",int(nx * ny * nz))

    # ------------tiles------------------
    active_tile_shape = _profile_call(
        "kernel_config.active_tile_shape",
        kernel_config.active_tile_shape,
        shape,
    )

    scalar_active_tiles = _profile_call(
        "cuda.device_array[scalar_active_tiles]",
        cuda.device_array,
        active_tile_shape,
        dtype=np.bool_,
        cuda_sync=True,
    )
    scalar_active_tiles_dilated = _profile_call(
        "cuda.device_array[scalar_active_tiles_dilated]",
        cuda.device_array,
        active_tile_shape,
        dtype=np.bool_,
        cuda_sync=True,
    )
    scalar_tile_padding = _profile_call(
        "kernel_config.active_tile_padding_tiles",
        kernel_config.active_tile_padding_tiles,
    )

    active_tile_mask_blocks = _profile_call(
        "kernel_config.volume_blocks_per_grid[active_tile_mask]",
        kernel_config.volume_blocks_per_grid,
        scalar_active_tiles.shape,
        kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK,
    )

    # ------------fields------------------
    # velocity
    u = _profile_call("cuda.device_array[u]", cuda.device_array, shape, dtype=GPU_FIELD_DTYPE, cuda_sync=True)
    u_work = _profile_call("cuda.device_array[u_work]", cuda.device_array, shape, dtype=GPU_FIELD_DTYPE, cuda_sync=True)
    v = _profile_call("cuda.device_array[v]", cuda.device_array, shape, dtype=GPU_FIELD_DTYPE, cuda_sync=True)
    v_work = _profile_call("cuda.device_array[v_work]", cuda.device_array, shape, dtype=GPU_FIELD_DTYPE, cuda_sync=True)
    w = _profile_call("cuda.device_array[w]", cuda.device_array, shape, dtype=GPU_FIELD_DTYPE, cuda_sync=True)
    w_work = _profile_call("cuda.device_array[w_work]", cuda.device_array, shape, dtype=GPU_FIELD_DTYPE, cuda_sync=True)

    # pressure
    p = _profile_call("cuda.device_array[p]", cuda.device_array, shape, dtype=GPU_FIELD_DTYPE, cuda_sync=True)
    pressure_rhs_partial_sums = _profile_call(
        "cuda.device_array[pressure_rhs_partial_sums]",
        cuda.device_array,
        kernel_config.MAX_REDUCTION_BLOCKS,
        dtype=np.float32,
        cuda_sync=True,
    )
    pressure_rhs_sum = _profile_call(
        "cuda.device_array[pressure_rhs_sum]",
        cuda.device_array,
        1,
        dtype=np.float32,
        cuda_sync=True,
    )

    # scalars
    temperature = _profile_call("cuda.device_array[temperature]", cuda.device_array, shape, dtype=GPU_FIELD_DTYPE, cuda_sync=True)
    temperature_work = _profile_call("cuda.device_array[temperature_work]", cuda.device_array, shape, dtype=GPU_FIELD_DTYPE, cuda_sync=True)
    smoke = _profile_call("cuda.device_array[smoke]", cuda.device_array, shape, dtype=GPU_FIELD_DTYPE, cuda_sync=True)
    smoke_work = _profile_call("cuda.device_array[smoke_work]", cuda.device_array, shape, dtype=GPU_FIELD_DTYPE, cuda_sync=True)
    fuel = _profile_call("cuda.device_array[fuel]", cuda.device_array, shape, dtype=GPU_FIELD_DTYPE, cuda_sync=True)
    fuel_work = _profile_call("cuda.device_array[fuel_work]", cuda.device_array, shape, dtype=GPU_FIELD_DTYPE, cuda_sync=True)
    flame = _profile_call("cuda.device_array[flame]", cuda.device_array, shape, dtype=GPU_FIELD_DTYPE, cuda_sync=True)

    # scratch
    scratch_A_x = _profile_call("cuda.device_array[scratch_A_x]", cuda.device_array, shape, dtype=GPU_FIELD_DTYPE, cuda_sync=True)
    scratch_A_y = _profile_call("cuda.device_array[scratch_A_y]", cuda.device_array, shape, dtype=GPU_FIELD_DTYPE, cuda_sync=True)
    scratch_A_z = _profile_call("cuda.device_array[scratch_A_z]", cuda.device_array, shape, dtype=GPU_FIELD_DTYPE, cuda_sync=True)

    # vortictiy
    vorticity_magnitude = _profile_call(
        "cuda.device_array[vorticity_magnitude]",
        cuda.device_array,
        shape,
        dtype=GPU_FIELD_DTYPE,
        cuda_sync=True,
    )

    # masks
    obstacle_mask = _profile_call(
        "cuda.to_device[obstacle_mask]",
        cuda.to_device,
        np.ascontiguousarray(obstacle_mask, dtype=np.bool_),
        cuda_sync=True,
    )
    source_mask_host = np.any(np.stack(source_masks, axis=0), axis=0) if source_masks else np.zeros(shape, dtype=np.bool_)
    source_mask = _profile_call(
        "cuda.to_device[source_mask]",
        cuda.to_device,
        np.ascontiguousarray(source_mask_host, dtype=np.bool_),
        cuda_sync=True,
    )
    source_mask_stack = np.ascontiguousarray(np.asarray(source_masks, dtype=np.bool_)) if source_masks else np.zeros((0,) + shape, dtype=np.bool_)
    source_masks = _profile_call(
        "cuda.to_device[source_masks]",
        cuda.to_device,
        source_mask_stack,
        cuda_sync=True,
    )
    source_noise_base_fields = _profile_call(
        "build_source_noise_fields",
        build_source_noise_fields,
        simulation.get("sources") or [],
        source_base_masks,
    )
    source_noise_host = np.zeros((len(source_noise_base_fields),) + shape, dtype=np.float32)
    source_noise = _profile_call(
        "cuda.to_device[source_noise]",
        cuda.to_device,
        np.ascontiguousarray(source_noise_host, dtype=np.float32),
        cuda_sync=True,
    )

    # multigrid levels
    p_levels, b_levels, delta_levels, zero_levels = _profile_call(
        "create_multigrid_levels",
        create_multigrid_levels,
        shape,
        delta,
        min_size=8,
        cuda_sync=True,
    )

    # ------------intitialise------------------
    u_initial,v_initial,w_initial = _profile_call(
        "compute_inital_velocity",
        compute_inital_velocity,
        simulation,
    )

    _profile_call("u.copy_to_device[init]", u.copy_to_device, np.full(shape, u_initial, dtype=GPU_FIELD_DTYPE), cuda_sync=True)
    _profile_call("v.copy_to_device[init]", v.copy_to_device, np.full(shape, v_initial, dtype=GPU_FIELD_DTYPE), cuda_sync=True)
    _profile_call("w.copy_to_device[init]", w.copy_to_device, np.full(shape, w_initial, dtype=GPU_FIELD_DTYPE), cuda_sync=True)

    _profile_call("p.copy_to_device[init]", p.copy_to_device, np.full(shape, 0, dtype=GPU_FIELD_DTYPE), cuda_sync=True)

    _profile_call(
        "temperature.copy_to_device[init]",
        temperature.copy_to_device,
        np.full(shape, simulation.get("physics").get("temperature").get("reference_temperature"), dtype=GPU_FIELD_DTYPE),
        cuda_sync=True,
    )
    _profile_call("smoke.copy_to_device[init]", smoke.copy_to_device, np.full(shape, 0, dtype=GPU_FIELD_DTYPE), cuda_sync=True)
    _profile_call("fuel.copy_to_device[init]", fuel.copy_to_device, np.full(shape, 0, dtype=GPU_FIELD_DTYPE), cuda_sync=True)
    _profile_call("flame.copy_to_device[init]", flame.copy_to_device, np.full(shape, 0, dtype=GPU_FIELD_DTYPE), cuda_sync=True)

    velocity_maxima = _profile_call(
        "cuda.to_device[velocity_maxima]",
        cuda.to_device,
        np.zeros(3, dtype=np.float32),
        cuda_sync=True,
    )
    velocity_maxima_host_zeros = np.zeros(3, dtype=np.float32)

    if source_noise_base_fields:
        _profile_call(
            "update_masks.update_source_values",
            update_masks.update_source_values,
            source_noise,
            source_noise_base_fields,
            t,
            delta,
            origin_x,
            origin_y,
            origin_z,
            cuda_sync=True,
        )

    # ------------output------------------
    output_cfg = ((simulation.get("outputs") or [None])[0]) or {}
    viewer_cfg = ((simulation.get("viewers") or [None])[0]) or {}
    output_time_step = 1.0 / int(output_cfg.get("fps", 24))
    target_realtime_preview = bool(viewer_cfg.get("target_realtime_preview", False))

    shared_memory_blocks, writer_slots = _profile_call(
        "output.setup_output",
        output.setup_output,
        simulation,
        simulation.get("outputs")[0].get("output_path"),
        shape,
    )

    device_fields = _profile_call(
        "_current_device_fields",
        _current_device_fields,
        u,
        v,
        w,
        p,
        temperature,
        smoke,
        fuel,
        flame,
    )
    _record_timing("initialization", perf_counter() - init_start_time)

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

        _profile_call(
            "time_step.reset_velocity_maxima",
            time_step.reset_velocity_maxima,
            velocity_maxima,
            velocity_maxima_host_zeros,
            cuda_sync=True,
        )
        dt = _profile_call(
            "time_step.compute_new_timestep_gpu",
            time_step.compute_new_timestep_gpu,
            u,
            v,
            w,
            velocity_maxima,
            delta,
            cfl,
            output_time_step,
            cuda_sync=True,
        )

        # ------------Clear scratch-------------------
        _profile_call("scratch_A_x.copy_to_device[clear]", scratch_A_x.copy_to_device, zero_levels[0], cuda_sync=True)
        _profile_call("scratch_A_y.copy_to_device[clear]", scratch_A_y.copy_to_device, zero_levels[0], cuda_sync=True)
        _profile_call("scratch_A_z.copy_to_device[clear]", scratch_A_z.copy_to_device, zero_levels[0], cuda_sync=True)

        # ------------Update masks-------------------
        if animated_sources:
            _profile_call(
                "update_masks.update_masks[source]",
                update_masks.update_masks,
                source_masks,
                source_base_masks,
                t,
                delta,
                origin_x,
                origin_y,
                origin_z,
                cuda_sync=True,
                aggregate_mask=source_mask,
            )
            if source_noise_base_fields:
                _profile_call(
                    "update_masks.update_source_values",
                    update_masks.update_source_values,
                    source_noise,
                    source_noise_base_fields,
                    t,
                    delta,
                    origin_x,
                    origin_y,
                    origin_z,
                    cuda_sync=True,
                )

        if animated_obstacles:
            _profile_call(
                "update_masks.update_masks[obstacle]",
                update_masks.update_masks,
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
                cuda_sync=True,
            )

        # ------------BC-------------------
        u, v, w, p, temperature, smoke, fuel, flame = _profile_call(
            "apply_all_BC",
            apply_all_BC,
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
            scratch_A_x,
            scratch_A_y,
            scratch_A_z,
            source_masks,
            source_noise,
            cuda_sync=True,
        )

        # ------------Start Active tiles-------------------
        if simulation.get("settings").get("simulate_sparsely"):
            _profile_call(
                "scalar_update.build_active_scalar_tile_mask",
                lambda: scalar_update.build_active_scalar_tile_mask[
                    active_tile_mask_blocks, kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK
                ](
                    temperature,
                    smoke,
                    fuel,
                    flame,
                    scalar_active_tiles,
                    simulation.get("physics").get("temperature").get("reference_temperature"),
                    simulation.get("settings").get("adaptive_domain_threshold"),
                ),
                cuda_sync=True,
            )
            _profile_call(
                "scalar_update.dilate_active_scalar_tile_mask",
                lambda: scalar_update.dilate_active_scalar_tile_mask[
                    active_tile_mask_blocks, kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK
                ](
                    scalar_active_tiles,
                    scalar_active_tiles_dilated,
                    scalar_tile_padding,
                ),
                cuda_sync=True,
            )
        else:
            _profile_call(
                "scalar_update.fill_active_scalar_tile_mask[scalar_active_tiles]",
                lambda: scalar_update.fill_active_scalar_tile_mask[
                    active_tile_mask_blocks, kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK
                ](
                    scalar_active_tiles,
                    np.bool_(True),
                ),
                cuda_sync=True,
            )
            _profile_call(
                "scalar_update.fill_active_scalar_tile_mask[scalar_active_tiles_dilated]",
                lambda: scalar_update.fill_active_scalar_tile_mask[
                    active_tile_mask_blocks, kernel_config.ACTIVE_TILE_MASK_THREADS_PER_BLOCK
                ](
                    scalar_active_tiles_dilated,
                    np.bool_(True),
                ),
                cuda_sync=True,
            )

        # ------------Vorticity-------------------
        if simulation.get("physics").get("extras").get("vorticity") > 0.0:
            _profile_call(
                "vorticity.compute_vorticity",
                lambda: vorticity.compute_vorticity[
                    active_tile_shape, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
                ](
                    u,
                    v,
                    w,
                    obstacle_mask,
                    vorticity_magnitude,
                    delta,
                    scalar_active_tiles_dilated,
                ),
                cuda_sync=True,
            )

        # ------------force params-------------------
        fx_const, fy_const, fz_const = _profile_call("forces.constant_force", forces.constant_force, simulation, t)
        swirl_config, has_swirl_nodes = _profile_call("forces.swirl_force", forces.swirl_force, simulation, t)
        turbulence_config, has_turbulence_nodes = _profile_call("forces.turbulence_force", forces.turbulence_force, simulation, t)
        swirl_config_device = _profile_call(
            "_prepare_force_config_device[swirl]",
            _prepare_force_config_device,
            swirl_config,
            8,
            cuda_sync=True,
        )
        turbulence_config_device = _profile_call(
            "_prepare_force_config_device[turbulence]",
            _prepare_force_config_device,
            turbulence_config,
            4,
            cuda_sync=True,
        )

        # ------------Velocity update-------------------
        _profile_call("u_work.copy_to_device", u_work.copy_to_device, u, cuda_sync=True)
        _profile_call("v_work.copy_to_device", v_work.copy_to_device, v, cuda_sync=True)
        _profile_call("w_work.copy_to_device", w_work.copy_to_device, w, cuda_sync=True)
        _profile_call(
            "advection_schemes.advect_velocity_semi_lagrangian",
            lambda: advection_schemes.advect_velocity_semi_lagrangian[
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
            ),
            cuda_sync=True,
        )
        _profile_call(
            "advection_schemes.update_velocity_maccormack",
            lambda: advection_schemes.update_velocity_maccormack[
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
                simulation.get("physics").get("fluid").get("density"),
                simulation.get("physics").get("fluid").get("viscosity"),
                vorticity_magnitude,
                simulation.get("physics").get("extras").get("vorticity"),
                temperature,
                simulation.get("physics", {}).get("temperature", {}).get("buoyancy"),
                simulation.get("physics").get("temperature").get("reference_temperature"),
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
            ),
            cuda_sync=True,
        )

        # ------------Velocity swap-------------------
        u, u_work = u_work, u
        v, v_work = v_work, v
        w, w_work = w_work, w

        # ------------Pressure solve-------------------
        extra_pressure = _profile_call("get_source_values[extra_pressure]", get_source_values, simulation, "extra_pressure", t)
        noise_amplitudes = _profile_call("get_source_values[noise_amplitude]", get_source_values, simulation, "noise_amplitude", t) / np.float32(100.0)

        p = _profile_call(
            "pressure_solve.pressure_poisson_multigrid",
            pressure_solve.pressure_poisson_multigrid,
            u, v, w,
            p,
            temperature,
            scratch_A_x,
            dt,
            source_masks,
            extra_pressure,
            source_noise,
            noise_amplitudes,
            delta,
            simulation.get("physics").get("fluid").get("density"),
            simulation.get("physics").get("temperature").get("expansion_rate"),
            simulation.get("physics").get("temperature").get("reference_temperature"),
            scalar_active_tiles_dilated,
            p_levels,
            b_levels,
            delta_levels,
            simulation.get("settings").get("iterations"),
            pressure_rhs_partial_sums,
            pressure_rhs_sum,
            zero_levels,
            cuda_sync=True,
        )

        # ------------Velocity projection-------------------
        _profile_call(
            "pressure_solve.project_velocity_kernel",
            lambda: pressure_solve.project_velocity_kernel[
                active_tile_shape, kernel_config.ACTIVE_TILE_THREADS_PER_BLOCK
            ](
                u,
                v,
                w,
                p,
                obstacle_mask,
                dt,
                delta,
                simulation.get("physics").get("fluid").get("density"),
                scalar_active_tiles_dilated,
            ),
            cuda_sync=True,
        )

        # ------------Scalar update-------------------
        _profile_call("temperature_work.copy_to_device", temperature_work.copy_to_device, temperature, cuda_sync=True)
        _profile_call("smoke_work.copy_to_device", smoke_work.copy_to_device, smoke, cuda_sync=True)
        _profile_call("fuel_work.copy_to_device", fuel_work.copy_to_device, fuel, cuda_sync=True)
        _profile_call(
            "scalar_update.predict_scalar_fields_semi_lagrangian",
            lambda: scalar_update.predict_scalar_fields_semi_lagrangian[
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
            ),
            cuda_sync=True,
        )
        _profile_call(
            "scalar_update.update_scalar_fields_maccormack",
            lambda: scalar_update.update_scalar_fields_maccormack[
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
                simulation.get("physics").get("temperature").get("dissipation"),
                simulation.get("physics").get("temperature").get("production_rate"),
                simulation.get("physics").get("smoke").get("dissipation"),
                simulation.get("physics").get("smoke").get("production_rate"),
                simulation.get("physics").get("fuel").get("dissipation"),
                simulation.get("physics").get("fuel").get("burn_rate"),
                simulation.get("physics").get("fuel").get("ignition_temperature"),
                simulation.get("physics").get("burning").get("scale"),
                simulation.get("physics").get("burning").get("amplitude"),
                simulation.get("physics").get("temperature").get("reference_temperature"),
                scalar_active_tiles_dilated,
            ),
            cuda_sync=True,
        )

        # ------------Swap-------------------
        temperature, temperature_work = temperature_work, temperature
        smoke, smoke_work = smoke_work, smoke
        fuel, fuel_work = fuel_work, fuel

        # ------------time updated-------------------
        t = t + dt
        time_step_count += 1

        # ------------Output-------------------
        device_fields = _profile_call(
            "_current_device_fields",
            _current_device_fields,
            u, v, w, p, temperature, smoke, fuel, flame
        )
        while t >= next_output_time:
            if target_realtime_preview and last_output_wall_time is not None:
                elapsed_since_last_output = perf_counter() - last_output_wall_time
                remaining_time = output_time_step - elapsed_since_last_output
                if remaining_time > 0.0:
                    sleep(remaining_time)

            _profile_call(
                "output.enqueue_device_output",
                output.enqueue_device_output,
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
        if time_step_count == 10:
            ctx = cuda.current_context()
            free, total = ctx.get_memory_info()
            used = total - free
            print(f"VRAM used: {used / 1024**2:.1f} MB")

    # ------------Shutdown output-------------------
    _profile_call(
        "output.shutdown_output",
        output.shutdown_output,
        shared_memory_blocks,
        writer_slots,
        cuda_sync=True,
    )

    # ------------Conclusion-------------------
    if cancel_requested:
        print("Simulation cancelled after clean shutdown.")
    else:
        print("Simulation finished!")

    total_runtime = perf_counter() - total_start_time
    print(f"Solver runtime: {total_runtime:.3f} s")
    print("Timing summary:")
    sorted_stats = sorted(
        profiling_stats.items(),
        key=lambda item: item[1]["total"],
        reverse=True,
    )
    name_width = max(len("Method"), *(len(name) for name, _ in sorted_stats))
    total_width = len("Total [s]")
    count_width = len("Calls")
    avg_width = len("Avg [s]")
    separator = (
        f"+-{'-' * name_width}-+-{'-' * total_width}-+-{'-' * count_width}-+-{'-' * avg_width}-+"
    )

    print(separator)
    print(
        f"| {'Method'.ljust(name_width)} | {'Total [s]'.rjust(total_width)} | "
        f"{'Calls'.rjust(count_width)} | {'Avg [s]'.rjust(avg_width)} |"
    )
    print(separator)
    for name, stat in sorted_stats:
        average = stat["total"] / stat["count"] if stat["count"] else 0.0
        print(
            f"| {name.ljust(name_width)} | {stat['total']:>{total_width}.6f} | "
            f"{stat['count']:>{count_width}} | {average:>{avg_width}.6f} |"
        )
    print(separator)
