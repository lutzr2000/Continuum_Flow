import ctypes
import math
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import Solver.Kernel_CPU.Boundary_Conditions.domain_bc as BC
import Solver.Kernel_CPU.advection_schemes as advection_schemes
import Solver.Kernel_CPU.scalar_update as scalar_update
import Solver.Kernel_CPU.pressure_solve as pressure_solve
import Solver.Kernel_CPU.vorticity as vorticity
import Solver.Kernel_CPU.kernel_config as kernel_config
import Solver.Kernel_CPU.Boundary_Conditions.obstacle_bc as obstacle_bc
import Solver.Kernel_CPU.Boundary_Conditions.source_bc as source_bc
import Solver.Kernel_CPU.time_step as time_step
import Solver.Kernel_CPU.update_masks as update_masks
import Solver.Kernel_CPU.output as output
import Solver.General.forces as forces

CPU_FIELD_DTYPE = np.float32
PROGRESS_EVENT_PREFIX = "__CONTINUUM_FLOW_PROGRESS__ "


if sys.platform == "win32":
    class _ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]


    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
    _kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    _psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_ulong,
    ]
    _psapi.GetProcessMemoryInfo.restype = ctypes.c_long
elif sys.platform == "darwin":
    _mach = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)

    class _TimeValue(ctypes.Structure):
        _fields_ = [("seconds", ctypes.c_int), ("microseconds", ctypes.c_int)]


    class _TaskBasicInfo64(ctypes.Structure):
        _fields_ = [
            ("virtual_size", ctypes.c_uint64),
            ("resident_size", ctypes.c_uint64),
            ("resident_size_max", ctypes.c_uint64),
            ("user_time", _TimeValue),
            ("system_time", _TimeValue),
            ("policy", ctypes.c_int),
            ("suspend_count", ctypes.c_int),
        ]


    _TASK_BASIC_INFO_64 = 5
    _TASK_BASIC_INFO_64_COUNT = ctypes.c_uint32(
        ctypes.sizeof(_TaskBasicInfo64) // ctypes.sizeof(ctypes.c_int)
    )
    _mach.mach_task_self.restype = ctypes.c_uint32
    _mach.task_info.argtypes = [
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    _mach.task_info.restype = ctypes.c_int


def _process_ram_usage_bytes():
    if sys.platform == "win32":
        counters = _ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(_ProcessMemoryCountersEx)
        process_handle = _kernel32.GetCurrentProcess()
        ok = _psapi.GetProcessMemoryInfo(
            process_handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize)

    if sys.platform.startswith("linux"):
        with open("/proc/self/statm", "r", encoding="ascii") as statm_file:
            statm_fields = statm_file.read().split()
        if len(statm_fields) < 2:
            raise RuntimeError("Unexpected /proc/self/statm format")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(statm_fields[1]) * int(page_size)

    if sys.platform == "darwin":
        task_info = _TaskBasicInfo64()
        task_info_count = ctypes.c_uint32(_TASK_BASIC_INFO_64_COUNT.value)
        result = _mach.task_info(
            _mach.mach_task_self(),
            _TASK_BASIC_INFO_64,
            ctypes.byref(task_info),
            ctypes.byref(task_info_count),
        )
        if result != 0:
            raise OSError(ctypes.get_errno(), "task_info failed")
        return int(task_info.resident_size)

    try:
        import resource

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError, OSError, ValueError) as exc:
        raise OSError(f"Memory tracking is not supported on platform '{sys.platform}'") from exc

    if "bsd" in sys.platform or sys.platform.startswith("dragonfly"):
        return rss
    return rss * 1024


def _current_output_fields(u, v, w, p, temperature, smoke, fuel, flame):
    """
    Return the active field buffers for output export.
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


def expand_active_tiles_to_mask(active_tiles, output_mask, tile_size):
    nx, ny, nz = output_mask.shape
    for i in range(nx):
        tile_i = i // tile_size
        for j in range(ny):
            tile_j = j // tile_size
            for k in range(nz):
                tile_k = k // tile_size
                output_mask[i, j, k] = active_tiles[tile_i, tile_j, tile_k]


def get_source_values(simulations, var_name, t, index=None):
    source_entries = simulations[0].get("sources") or []
    values = np.zeros(len(source_entries), dtype=CPU_FIELD_DTYPE)

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

        values[source_idx] = np.asarray(value, dtype=CPU_FIELD_DTYPE)

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


def _prepare_force_config(config_array, row_width):
    return np.ascontiguousarray(
        np.asarray(config_array, dtype=CPU_FIELD_DTYPE).reshape((-1, row_width))
    )


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
    source_masks,
    source_noise,
):
    """
    Apply domain, obstacle, and source boundary conditions.
    """
    bc_config = simulations[0].get("domain", {}).get("boundary_conditions", {})
    u, v, w, p, T, smoke, fuel = BC.domain_bc(u, v, w, p, T, smoke, fuel, bc_config)

    if obstacle_mask is not None:
        obstacle_bc.obstacle_bc_kernel(
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
        source_temperature_values = get_source_values(simulations, "temperature", t)
        source_smoke_values = get_source_values(simulations, "smoke", t)
        source_fuel_values = get_source_values(simulations, "fuel", t)
        source_velocity_x_values = get_source_values(simulations, "velocity", t, 0)
        source_velocity_y_values = get_source_values(simulations, "velocity", t, 1)
        source_velocity_z_values = get_source_values(simulations, "velocity", t, 2)
        source_noise_amplitudes = get_source_values(simulations, "noise_amplitude", t) / np.float32(100.0)

        for source_idx in range(source_count):
            source_bc.source_bc_kernel(
                u,
                v,
                w,
                T,
                smoke,
                fuel,
                source_masks[source_idx],
                source_noise[source_idx],
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

        p_levels.append(np.zeros(level_shape, dtype=np.float32))
        b_levels.append(np.zeros(level_shape, dtype=np.float32))
        zero_levels.append(np.zeros(level_shape, dtype=np.float32))
        delta_levels.append(delta * (2 ** level))

        nx = (nx + 1) // 2
        ny = (ny + 1) // 2
        nz = (nz + 1) // 2
        level += 1

    return p_levels, b_levels, delta_levels, zero_levels


def solver(config, obstacle_base_masks, obstacle_mask, source_base_masks, source_masks, animated_obstacles, animated_sources):
    total_start_time = perf_counter()
    simulations = config.get("simulations")
    cancel_flag_path = ((config.get("meta") or {}).get("cancel_flag_path") or "").strip()
    cancel_requested = False

    # ------------time-------------------
    t = 0.0
    cfl = float(simulations[0].get("settings", {}).get("cfl", 10.0))
    t_max = simulations[0].get("settings").get("simulation_length")

    # ------------dimensions------------------
    delta = simulations[0].get("domain").get("resolution")
    nx = simulations[0]["domain"]["grid"]["nx"]
    ny = simulations[0]["domain"]["grid"]["ny"]
    nz = simulations[0]["domain"]["grid"]["nz"]
    shape = (nx, ny, nz)

    origin_x = -0.5 * nx * delta
    origin_y = -0.5 * ny * delta
    origin_z = 0.0

    print("################################################################")
    print("Initialise")
    print("Cell count: ", int(nx * ny * nz))

    # ------------tiles------------------
    active_tile_shape = kernel_config.active_tile_shape(shape)

    scalar_active_tiles = np.zeros(active_tile_shape, dtype=np.bool_)
    scalar_active_tiles_dilated = np.zeros(active_tile_shape, dtype=np.bool_)
    scalar_tile_padding = kernel_config.active_tile_padding_tiles()

    # ------------fields------------------
    # velocity
    u = np.empty(shape, dtype=CPU_FIELD_DTYPE)
    u_work = np.empty(shape, dtype=CPU_FIELD_DTYPE)
    v = np.empty(shape, dtype=CPU_FIELD_DTYPE)
    v_work = np.empty(shape, dtype=CPU_FIELD_DTYPE)
    w = np.empty(shape, dtype=CPU_FIELD_DTYPE)
    w_work = np.empty(shape, dtype=CPU_FIELD_DTYPE)

    # pressure
    p = np.empty(shape, dtype=CPU_FIELD_DTYPE)

    # scalars
    temperature = np.empty(shape, dtype=CPU_FIELD_DTYPE)
    temperature_work = np.empty(shape, dtype=CPU_FIELD_DTYPE)
    smoke = np.empty(shape, dtype=CPU_FIELD_DTYPE)
    smoke_work = np.empty(shape, dtype=CPU_FIELD_DTYPE)
    fuel = np.empty(shape, dtype=CPU_FIELD_DTYPE)
    fuel_work = np.empty(shape, dtype=CPU_FIELD_DTYPE)
    flame = np.empty(shape, dtype=CPU_FIELD_DTYPE)

    # scratch
    scratch_A_x = np.empty(shape, dtype=CPU_FIELD_DTYPE)
    scratch_A_y = np.empty(shape, dtype=CPU_FIELD_DTYPE)
    scratch_A_z = np.empty(shape, dtype=CPU_FIELD_DTYPE)

    # vortictiy
    vorticity_magnitude = np.empty(shape, dtype=CPU_FIELD_DTYPE)

    # masks
    obstacle_mask = np.ascontiguousarray(obstacle_mask, dtype=np.bool_)
    source_mask_host = np.any(np.stack(source_masks, axis=0), axis=0) if source_masks else np.zeros(shape, dtype=np.bool_)
    source_mask = np.ascontiguousarray(source_mask_host, dtype=np.bool_)
    source_mask_stack = np.ascontiguousarray(np.asarray(source_masks, dtype=np.bool_)) if source_masks else np.zeros((0,) + shape, dtype=np.bool_)
    source_masks = source_mask_stack
    source_noise_base_fields = build_source_noise_fields(simulations[0].get("sources") or [], source_base_masks)
    source_noise = np.zeros((len(source_noise_base_fields),) + shape, dtype=np.float32)

    # multigrid levels
    p_levels, b_levels, delta_levels, zero_levels = create_multigrid_levels(
        shape,
        delta,
        min_size=8,
    )

    # ------------intitialise------------------
    u_initial, v_initial, w_initial = compute_inital_velocity(simulations[0])

    u[...] = np.full(shape, u_initial, dtype=CPU_FIELD_DTYPE)
    v[...] = np.full(shape, v_initial, dtype=CPU_FIELD_DTYPE)
    w[...] = np.full(shape, w_initial, dtype=CPU_FIELD_DTYPE)

    p[...] = np.full(shape, 0, dtype=CPU_FIELD_DTYPE)

    ref_temp = simulations[0].get("physics").get("temperature").get("reference_temperature")
    temperature[...] = np.full(shape, ref_temp, dtype=CPU_FIELD_DTYPE)
    smoke[...] = np.full(shape, 0, dtype=CPU_FIELD_DTYPE)
    fuel[...] = np.full(shape, 0, dtype=CPU_FIELD_DTYPE)
    flame[...] = np.full(shape, 0, dtype=CPU_FIELD_DTYPE)

    velocity_maxima = np.zeros(3, dtype=np.float32)

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
    output_cfg = ((simulations[0].get("outputs") or [None])[0]) or {}
    output_time_step = 1.0 / int(output_cfg.get("fps", 24))

    shared_memory_blocks, writer_slots = output.setup_output(
        simulations[0],
        simulations[0].get("outputs")[0].get("output_path"),
        shape,
    )


    # ------------time loop------------------
    print("Start time iteration")
    next_output_time = 0.0
    output_index = 0
    time_step_count = 0
    while t < t_max:
        if cancel_flag_path and Path(cancel_flag_path).exists():
            cancel_requested = True
            print("Bake cancellation requested. Stopping the simulation cleanly...")
            break

        dt = time_step.compute_new_timestep(
            u,
            v,
            w,
            velocity_maxima,
            delta,
            cfl,
            output_time_step,
        )

        # ------------Clear scratch-------------------
        scratch_A_x[...] = zero_levels[0]
        scratch_A_y[...] = zero_levels[0]
        scratch_A_z[...] = zero_levels[0]
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
            source_noise,
        )
        # ------------Start Active tiles-------------------
        if simulations[0].get("settings").get("simulate_sparsely"):
            scalar_update.build_active_scalar_tile_mask(
                temperature,
                smoke,
                fuel,
                flame,
                scalar_active_tiles,
                ref_temp,
                simulations[0].get("settings").get("adaptive_domain_threshold"),
            )
            scalar_update.dilate_active_scalar_tile_mask(
                scalar_active_tiles,
                scalar_active_tiles_dilated,
                scalar_tile_padding,
            )
        else:
            scalar_update.fill_active_scalar_tile_mask(
                scalar_active_tiles,
                np.bool_(True),
            )
            scalar_update.fill_active_scalar_tile_mask(
                scalar_active_tiles_dilated,
                np.bool_(True),
            )
        # ------------Vorticity-------------------
        if simulations[0].get("physics").get("extras").get("vorticity") > 0.0:
            vorticity.compute_vorticity(
                u,
                v,
                w,
                obstacle_mask,
                vorticity_magnitude,
                delta,
                scalar_active_tiles_dilated,
            )
        # ------------force params-------------------
        fx_const, fy_const, fz_const = forces.constant_force(simulations[0], t)
        swirl_config, has_swirl_nodes = forces.swirl_force(simulations[0], t)
        turbulence_config, has_turbulence_nodes = forces.turbulence_force(simulations[0], t)
        swirl_config_buffer = _prepare_force_config(swirl_config, 8)
        turbulence_config_buffer = _prepare_force_config(turbulence_config, 4)
        # ------------Velocity update-------------------
        u_work[...] = u
        v_work[...] = v
        w_work[...] = w
        advection_schemes.advect_velocity_semi_lagrangian(
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
        advection_schemes.update_velocity_maccormack(
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
            ref_temp,
            scalar_active_tiles_dilated,
            fx_const,
            fy_const,
            fz_const,
            has_swirl_nodes,
            swirl_config_buffer,
            origin_x,
            origin_y,
            origin_z,
            has_turbulence_nodes,
            turbulence_config_buffer,
            t,
        )
        # ------------Velocity swap-------------------
        u, u_work = u_work, u
        v, v_work = v_work, v
        w, w_work = w_work, w

        # ------------Pressure solve-------------------
        extra_pressure = get_source_values(simulations, "extra_pressure", t)
        noise_amplitudes = get_source_values(simulations, "noise_amplitude", t) / np.float32(100.0)

        p = pressure_solve.pressure_poisson_multigrid(
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
            simulations[0].get("physics").get("fluid").get("density"),
            simulations[0].get("physics").get("temperature").get("expansion_rate"),
            ref_temp,
            scalar_active_tiles_dilated,
            p_levels,
            b_levels,
            delta_levels,
            simulations[0].get("settings").get("iterations"),
            zero_levels,
        )
        # ------------Velocity projection-------------------
        pressure_solve.project_velocity_kernel(
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
        temperature_work[...] = temperature
        smoke_work[...] = smoke
        fuel_work[...] = fuel
        scalar_update.predict_scalar_fields_semi_lagrangian(
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
        scalar_update.update_scalar_fields_maccormack(
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
            simulations[0].get("physics").get("burning").get("scale"),
            simulations[0].get("physics").get("burning").get("amplitude"),
            ref_temp,
            scalar_active_tiles_dilated,
        )
        # ------------Swap-------------------
        temperature, temperature_work = temperature_work, temperature
        smoke, smoke_work = smoke_work, smoke
        fuel, fuel_work = fuel_work, fuel

        # ------------time updated-------------------
        t = t + dt
        time_step_count += 1

        # ------------Output-------------------
        output_fields = _current_output_fields(
            u, v, w, p, temperature, smoke, fuel, flame
        )
        while t >= next_output_time:
            output.enqueue_output(
                simulations[0],
                writer_slots,
                output_fields,
                output_index,
                t,
            )

            output_index += 1
            next_output_time += output_time_step
        # ------------Memory track-------------------
        if time_step_count == 10:
            try:
                ram_usage = _process_ram_usage_bytes()
            except OSError as exc:
                print(f"RAM usage tracking unavailable: {exc}")
            else:
                print(f"RAM used: {ram_usage / 1024**2:.1f} MB")

    # ------------Shutdown output-------------------
    output.shutdown_output(shared_memory_blocks, writer_slots)

    # ------------Conclusion-------------------
    if cancel_requested:
        print("Simulation cancelled after clean shutdown.")
    else:
        print("Simulation finished!")

    total_runtime = perf_counter() - total_start_time
    print(f"Solver runtime: {total_runtime:.3f} s")

