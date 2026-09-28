import pyopencl as cl
import math
import numpy as np
from typing import Any
from pathlib import Path
from Solver.General.main import emit_message

import Solver.Kernel.kernel_config as kernel_config
import Solver.Kernel.update_masks as update_masks

FIELD_DTYPE = kernel_config.FIELD_DTYPE

mf = cl.mem_flags


def to_device(context, array):
    """NumPy-Array copy to OpenCL-Buffer."""
    return cl.Buffer(
        context,
        mf.READ_WRITE | mf.COPY_HOST_PTR,
        hostbuf=array,
    )


def device_array(context, shape, dtype):
    """Allocate uninitialized OpenCL buffer."""
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize

    return cl.Buffer(
        context,
        mf.READ_WRITE,
        size=size,
    )


def zeros_device(context, shape, dtype=FIELD_DTYPE):
    return to_device(context, np.zeros(shape, dtype=dtype))


def full_device(context, shape, value, dtype=FIELD_DTYPE):
    return to_device(context, np.full(shape, value, dtype=dtype))


def is_animated(base_masks: list[dict[str, Any]]) -> bool:
    """
    Determine whether any mesh represented by the base masks is animated.

    A mesh is considered animated when its transform animation contains
    multiple world matrices and at least one matrix differs from the first
    sample. Repeated samples of an identical transform are therefore treated
    as static.
    """
    for entry in base_masks:
        matrices = np.asarray(entry.get("matrix_matrices", ()))
        if matrices.size == 0:
            mesh_object = entry["mesh_object"]
            animation = mesh_object.get("transform_animation") or {}
            matrices = np.asarray(animation.get("matrices_world", ()))
        if matrices.size == 0:
            continue
        matrices = matrices.reshape(-1, 4, 4)

        # Repeated samples of the same transform are static.
        if len(matrices) > 1 and np.any(matrices[1:] != matrices[0]):
            return True

    return False


def compute_inital_velocity(
    simulation_cfg: dict[str, Any],
) -> tuple[float, float, float]:
    """
    Compute the initial velocity from the configured domain inflows.

    The velocity vectors of all domain faces configured as inflows are
    averaged component-wise. Both the string value ``"INFLOW"`` and the
    numeric boundary-condition value ``1`` are recognized as inflows.
    """
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


def solver(config: dict):

    # ------------device-------------------
    device = None

    for platform in cl.get_platforms():
        devices = platform.get_devices(device_type=cl.device_type.GPU)

        if devices:
            device = devices[0]
            break

    # ------------context-------------------
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    print("################################################################")
    print(f"Running on: {device.name}")

    # ------------config-------------------
    simulation = config.get("simulation") or {}
    cancel_flag_path = (
        (config.get("meta") or {}).get("cancel_flag_path") or ""
    ).strip()
    cancel_requested = False

    bake_path = simulation["outputs"][0]["output_path"]

    # ------------time-------------------
    t = 0.0
    cfl = float(simulation.get("settings", {}).get("cfl", 10.0))
    tile_dilate = math.ceil(cfl / kernel_config.TILE_SIZE)
    advection_substeps = max(
        1, int(simulation.get("settings", {}).get("advection_substeps", 1))
    )
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

    # ------------sparse layout------------------
    if simulate_sparsely:
        tile_map_values = np.full(
            tile_shape,
            -1,
            dtype=np.int32,
        )

        base_tile_map_values = np.full(
            tile_shape,
            -1,
            dtype=np.int32,
        )

        initial_next_tile_index = 0
        initial_active_tile_count = 0

    else:
        tile_map_values = np.arange(
            total_tile_count,
            dtype=np.int32,
        ).reshape(tile_shape)

        base_tile_map_values = np.ones(
            tile_shape,
            dtype=np.int32,
        )

        initial_next_tile_index = total_tile_count
        initial_active_tile_count = total_tile_count

    # controls the indexing in which slot a tile is
    tile_map = to_device(context, tile_map_values)
    # controls which tile is active
    base_tile_map = to_device(context, base_tile_map_values)

    free_slot_stack = to_device(context, np.full(total_tile_count, -1, dtype=np.int32))
    free_slot_count = to_device(context, np.zeros(1, dtype=np.int32))

    reused_slot_stack = to_device(
        context, np.full(total_tile_count, -1, dtype=np.int32)
    )
    reused_slot_count = to_device(context, np.zeros(1, dtype=np.int32))

    next_tile_index_counter = to_device(
        context, np.asarray([initial_next_tile_index], dtype=np.int32)
    )
    active_tile_counter = to_device(
        context, np.asarray([initial_active_tile_count], dtype=np.int32)
    )

    tile_growth_size = max(
        1,
        math.ceil(
            total_tile_count * (float(kernel_config.SPARSE_TILE_GROWTH_PERCENT) / 100.0)
        ),
    )

    print("Initialise")
    print("Total tiles: ", total_tile_count)
    print("Maximum number of cells: ", total_tile_count * kernel_config.TILE_SIZE**3)

    # ------------fields------------------
    sparse_tile_capacity = (
        total_tile_count if not simulate_sparsely else max(1, tile_growth_size)
    )
    sparse_pool_shape = (
        sparse_tile_capacity,
        kernel_config.TILE_SIZE,
        kernel_config.TILE_SIZE,
        kernel_config.TILE_SIZE,
    )

    zero_pool = zeros_device(context, sparse_pool_shape)

    # velocity
    u_initial, v_initial, w_initial = compute_inital_velocity(simulation)

    u = full_device(context, sparse_pool_shape, u_initial)
    v = full_device(context, sparse_pool_shape, v_initial)
    w = full_device(context, sparse_pool_shape, w_initial)

    u_work = full_device(context, sparse_pool_shape, u_initial)
    v_work = full_device(context, sparse_pool_shape, v_initial)
    w_work = full_device(context, sparse_pool_shape, w_initial)

    velocity_maxima = zeros_device(context, 3)

    # scalars
    temperature = full_device(context, sparse_pool_shape, reference_temperature)
    smoke = zeros_device(context, sparse_pool_shape)
    fuel = zeros_device(context, sparse_pool_shape)

    temperature_work = full_device(context, sparse_pool_shape, reference_temperature)
    smoke_work = zeros_device(context, sparse_pool_shape)
    fuel_work = zeros_device(context, sparse_pool_shape)

    # flame
    flame = zeros_device(context, sparse_pool_shape)

    # vorticity
    vorticity_magnitude = zeros_device(context, sparse_pool_shape)

    # scratch
    scratch_A = full_device(context, sparse_pool_shape, reference_temperature)
    scratch_B = zeros_device(context, sparse_pool_shape)
    scratch_C = zeros_device(context, sparse_pool_shape)

    # pressure
    p = zeros_device(context, sparse_pool_shape)
    pressure_rhs = zeros_device(context, sparse_pool_shape)

    rhs_partial_sums = zeros_device(context, kernel_config.MAX_REDUCTION_BLOCKS)
    rhs_partial_counts = zeros_device(context, kernel_config.MAX_REDUCTION_BLOCKS)
    rhs_mean_buffer = zeros_device(context, 1)

    # ------------reference frame------------------
    reference_frame = simulation.get("reference_frame") or {}

    has_reference_frame = bool(reference_frame.get("object_name"))

    reference_frame["animation_timeline"] = simulation["animation_timeline"]
    reference_matrix_data = update_masks.prepare_matrix_data(reference_frame)
    reference_velocity_transfer = float(reference_frame.get("velocity_transfer", 0.0))

    reference_frame_animated = bool(
        has_reference_frame
        and len(reference_matrix_data[1]) > 1
        and np.any(reference_matrix_data[1][1:] != reference_matrix_data[1][0])
    )
    previous_reference_velocity_transform = np.zeros((3, 4), dtype=FIELD_DTYPE)

    # ------------source meshes------------------
    sources = simulation.get("sources") or []

    source_base_masks = []
    source_masks = []
    geometry_source_masks = []

    for source in sources:
        # voxelise source meshes
        base_masks = voxelise_mesh.voxelise_all_meshes(
            delta,
            source.get("geometry_inputs"),
            bake_path,
        )

        # prepare source transforms
        for entry in base_masks:
            times, matrices, rates = update_masks.prepare_matrix_data(
                entry["mesh_object"]
            )

            if has_reference_frame:
                times, matrices, rates = update_masks.make_matrix_data_relative(
                    times,
                    matrices,
                    *reference_matrix_data,
                )

            entry["matrix_times"] = times
            entry["matrix_matrices"] = matrices
            entry["matrix_rates"] = rates

        source_base_masks.append(base_masks)

        # source masks
        source_mask = zeros_device(context, sparse_pool_shape, dtype=np.bool_)
        source_masks.append(source_mask)

        geometry_source_masks.append(
            zeros_device(context, sparse_pool_shape, dtype=np.bool_)
            if has_particle_sources
            else source_mask
        )

    has_animated_sources = any(
        is_animated(base_masks) for base_masks in source_base_masks
    )

    source_tile_mask = zeros_device(
        context, tile_shape, dtype=np.bool_
    )  # determines which tiles are active due to source activity

    # ------------source particles------------------
    has_particle_sources = any(
        source.get("particle_system_inputs") for source in sources
    )

    particle_sources = (
        particles.load_particle_sources(
            sources,
            bake_path,
            reference_matrix_data if has_reference_frame else None,
        )
        if has_particle_sources
        else [[] for _ in sources]
    )

    particle_source_flags = [bool(entries) for entries in particle_sources]

    # ------------obstacle meshes------------------
    obstacles = simulation.get("obstacles") or []

    obstacle_base_masks = []

    for obstacle in obstacles:
        base_masks = voxelise_mesh.voxelise_all_meshes(
            delta,
            obstacle.get("geometry_inputs"),
            bake_path,
        )

        for entry in base_masks:
            times, matrices, rates = update_masks.prepare_matrix_data(
                entry["mesh_object"]
            )

            if has_reference_frame:
                times, matrices, rates = update_masks.make_matrix_data_relative(
                    times,
                    matrices,
                    *reference_matrix_data,
                )

            entry["matrix_times"] = times
            entry["matrix_matrices"] = matrices
            entry["matrix_rates"] = rates

        obstacle_base_masks.extend(base_masks)

    obstacle_mask = zeros_device(context, sparse_pool_shape, dtype=np.bool_)

    # add times
    for entry in sources + obstacles:
        for obj in entry.get("geometry_inputs") or []:
            obj["animation_timeline"] = simulation["animation_timeline"]

    # ------------output------------------
    output_cfg = ((simulation.get("outputs") or [None])[0]) or {}
    output_time_step = 1.0 / int(output_cfg.get("fps", 24))

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

        dt = output_time_step  # !!!!!!!!!!!!!!!!!!!!

        # ------------time updated-------------------
        t = t + dt
        time_step_count += 1

        # ------------Output-------------------
        while t >= next_output_time:

            buffers = {
                id(obj): obj for obj in locals().values() if isinstance(obj, cl.Buffer)
            }

            allocated_vram = sum(buffer.size for buffer in buffers.values())

            total_vram = device.global_mem_size
            free_vram = total_vram - allocated_vram

            emit_message(
                {
                    "type": "stats",
                    "frame": output_index,
                    "active_tiles": active_tile_counter_host,
                    "total_tiles": total_tile_count,
                    "active_cells": active_tile_counter_host
                    * kernel_config.TILE_SIZE**3,
                    "total_cells": total_tile_count * kernel_config.TILE_SIZE**3,
                    "vram_used_mb": (total_vram - free_vram) / 1024**2,
                    "vram_total_mb": total_vram / 1024**2,
                }
            )

            output_index += 1
            next_output_time += output_time_step

    # ------------Conclusion-------------------
    if cancel_requested:
        print("Simulation cancelled after clean shutdown.")
    else:
        print("Simulation finished!")
