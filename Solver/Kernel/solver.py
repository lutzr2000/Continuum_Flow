import pyopencl as cl
import math
import numpy as np
from numpy.typing import NDArray
from typing import Any
from pathlib import Path
from Solver.General.main import emit_message

import Solver.Kernel.kernel_config as kernel_config
import Solver.Kernel.voxelise_mesh as voxelise_mesh
import Solver.Kernel.multigrid as multigrid
import Solver.Kernel.helper as helper
import Solver.Kernel.sparse_managment as sparse_managment

FIELD_DTYPE = kernel_config.FIELD_DTYPE


def get_source_values(
    simulation: dict[str, Any],
    t: float,
) -> dict[str, NDArray]:
    """Resolve all source properties at the animation sample nearest to ``t``."""
    source_entries = simulation.get("sources") or []
    animation_times = (simulation.get("animation_timeline") or {}).get("times") or ()
    property_map = {
        "temperature": ("temperature", None, FIELD_DTYPE, 1.0),
        "smoke": ("smoke", None, FIELD_DTYPE, 1.0),
        "fuel": ("fuel", None, FIELD_DTYPE, 1.0),
        "noise_scale": ("noise_scale", None, FIELD_DTYPE, 1.0),
        "noise_amplitude": ("noise_amplitude", None, FIELD_DTYPE, 0.01),
        "noise_seed": ("noise_seed", None, np.int32, 1.0),
        "noise_enabled": ("source_noise", None, np.bool_, 1.0),
        "velocity_x": ("velocity", 0, FIELD_DTYPE, 1.0),
        "velocity_y": ("velocity", 1, FIELD_DTYPE, 1.0),
        "velocity_z": ("velocity", 2, FIELD_DTYPE, 1.0),
        "extra_pressure": ("extra_pressure", None, FIELD_DTYPE, 1.0),
    }
    values = {
        name: np.zeros(len(source_entries), dtype=dtype)
        for name, (_, _, dtype, _) in property_map.items()
    }

    for value_name, (property_name, component, dtype, scale) in property_map.items():
        for source_idx, source_entry in enumerate(source_entries):
            value = source_entry.get(property_name, 0.0)
            animation_values = (
                (source_entry.get("animations") or {}).get(property_name) or {}
            ).get("values") or ()
            sample_count = min(len(animation_times), len(animation_values))
            if sample_count > 0:
                nearest_time_idx = min(
                    range(sample_count),
                    key=lambda idx: abs(float(animation_times[idx]) - float(t)),
                )
                value = animation_values[nearest_time_idx]
            if component is not None:
                value = value[component] if value is not None else 0.0
            values[value_name][source_idx] = np.asarray(value, dtype=dtype) * scale

    values["noise_amplitude"][~values["noise_enabled"]] = 0.0
    values["velocity_local"] = np.asarray(
        [
            str(source.get("velocity_space", "WORLD")).upper() == "LOCAL"
            for source in source_entries
        ],
        dtype=np.bool_,
    )
    return values


def get_simulation_values(simulation: dict[str, Any], t: float) -> dict[str, Any]:
    """Resolve all physics values at the animation sample nearest to ``t``."""
    physics = simulation.get("physics") or {}
    animation_times = (simulation.get("animation_timeline") or {}).get("times") or ()
    animations = physics.get("animations") or {}
    property_map = {
        "fluid": {"density": "fluid_density", "viscosity": "fluid_viscosity"},
        "temperature": {
            "dissipation": "temperature_dissipation",
            "production_rate": "temperature_production_rate",
            "reference_temperature": "reference_temperature",
            "buoyancy": "buoyancy",
            "expansion_rate": "expansion_rate",
        },
        "smoke": {
            "dissipation": "smoke_dissipation",
            "production_rate": "smoke_production_rate",
        },
        "fuel": {
            "dissipation": "fuel_dissipation",
            "burn_rate": "fuel_burn_rate",
            "ignition_temperature": "fuel_ignition_temperature",
        },
        "burning": {"scale": "burn_noise_scale", "amplitude": "burn_noise_amplitude"},
        "extras": {"vorticity": "vorticity"},
    }

    values = {}
    for section_name, section_properties in property_map.items():
        static_section = physics.get(section_name) or {}
        values[section_name] = {}
        for value_name, animation_name in section_properties.items():
            value = static_section.get(value_name, 0.0)
            animation_values = (animations.get(animation_name) or {}).get(
                "values"
            ) or ()
            sample_count = min(len(animation_times), len(animation_values))
            if sample_count > 0:
                nearest_time_idx = min(
                    range(sample_count),
                    key=lambda idx: abs(float(animation_times[idx]) - float(t)),
                )
                value = animation_values[nearest_time_idx]
            values[section_name][value_name] = float(value)

    return values


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

    kernel_path = Path(__file__).parent / "OpenCL" / "voxelise_mesh.cl"

    with kernel_path.open("r", encoding="utf-8") as f:
        voxelise_mesh_program = cl.Program(context, f.read()).build()

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
    tile_map = helper.to_device(context, tile_map_values)
    # controls which tile is active
    base_tile_map = helper.to_device(context, base_tile_map_values)

    free_slot_stack = helper.to_device(
        context, np.full(total_tile_count, -1, dtype=np.int32)
    )
    free_slot_count = helper.to_device(context, np.zeros(1, dtype=np.int32))

    reused_slot_stack = helper.to_device(
        context, np.full(total_tile_count, -1, dtype=np.int32)
    )
    reused_slot_count = helper.to_device(context, np.zeros(1, dtype=np.int32))

    next_tile_index_counter = helper.to_device(
        context, np.asarray([initial_next_tile_index], dtype=np.int32)
    )
    active_tile_counter = helper.to_device(
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

    zero_pool = helper.zeros_device(context, sparse_pool_shape)

    # velocity
    u_initial, v_initial, w_initial = compute_inital_velocity(simulation)

    u = helper.full_device(context, sparse_pool_shape, u_initial)
    v = helper.full_device(context, sparse_pool_shape, v_initial)
    w = helper.full_device(context, sparse_pool_shape, w_initial)

    u_work = helper.full_device(context, sparse_pool_shape, u_initial)
    v_work = helper.full_device(context, sparse_pool_shape, v_initial)
    w_work = helper.full_device(context, sparse_pool_shape, w_initial)

    velocity_maxima = helper.zeros_device(context, 3)

    # scalars
    temperature = helper.full_device(context, sparse_pool_shape, reference_temperature)
    smoke = helper.zeros_device(context, sparse_pool_shape)
    fuel = helper.zeros_device(context, sparse_pool_shape)

    temperature_work = helper.full_device(
        context, sparse_pool_shape, reference_temperature
    )
    smoke_work = helper.zeros_device(context, sparse_pool_shape)
    fuel_work = helper.zeros_device(context, sparse_pool_shape)

    # flame
    flame = helper.zeros_device(context, sparse_pool_shape)

    # vorticity
    vorticity_magnitude = helper.zeros_device(context, sparse_pool_shape)

    # scratch
    scratch_A = helper.full_device(context, sparse_pool_shape, reference_temperature)
    scratch_B = helper.zeros_device(context, sparse_pool_shape)
    scratch_C = helper.zeros_device(context, sparse_pool_shape)

    # pressure
    p = helper.zeros_device(context, sparse_pool_shape)
    pressure_rhs = helper.zeros_device(context, sparse_pool_shape)

    rhs_partial_sums = helper.zeros_device(context, kernel_config.MAX_REDUCTION_BLOCKS)
    rhs_partial_counts = helper.zeros_device(
        context, kernel_config.MAX_REDUCTION_BLOCKS
    )
    rhs_mean_buffer = helper.zeros_device(context, 1)

    # ------------reference frame------------------
    reference_frame = simulation.get("reference_frame") or {}

    has_reference_frame = bool(reference_frame.get("object_name"))

    reference_frame["animation_timeline"] = simulation["animation_timeline"]
    reference_matrix_data = helper.prepare_matrix_data(reference_frame)
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

    for source in sources:
        # voxelise source meshes
        base_masks = voxelise_mesh.voxelise_all_meshes(
            context,
            queue,
            voxelise_mesh_program,
            delta,
            source.get("geometry_inputs"),
            bake_path,
        )

        # prepare source transforms
        for entry in base_masks:
            times, matrices, rates = helper.prepare_matrix_data(entry["mesh_object"])

            if has_reference_frame:
                times, matrices, rates = helper.make_matrix_data_relative(
                    times,
                    matrices,
                    *reference_matrix_data,
                )

            entry["matrix_times"] = times
            entry["matrix_matrices"] = matrices
            entry["matrix_rates"] = rates

        source_base_masks.append(base_masks)

        # geometry source mask
        source_masks.append(
            helper.zeros_device(context, sparse_pool_shape, dtype=np.bool_)
        )

    has_animated_sources = any(
        is_animated(base_masks) for base_masks in source_base_masks
    )

    source_tile_mask = helper.zeros_device(
        context, tile_shape, dtype=np.bool_
    )  # determines which tiles are active due to source activity

    # ------------source particles------------------
    # particle_source_masks = [
    #     helper.zeros_device(context, sparse_pool_shape, dtype=np.bool_)
    #     for _ in sources
    # ]

    # has_particle_sources = any(
    #     source.get("particle_system_inputs") for source in sources
    # )

    # particle_sources = (
    #     particles.load_particle_sources(
    #         sources,
    #         bake_path,
    #         reference_matrix_data if has_reference_frame else None,
    #     )
    #     if has_particle_sources
    #     else [[] for _ in sources]
    # )

    # particle_source_flags = [bool(entries) for entries in particle_sources]

    # ------------obstacle meshes------------------
    obstacles = simulation.get("obstacles") or []

    obstacle_base_masks = []

    for obstacle in obstacles:
        base_masks = voxelise_mesh.voxelise_all_meshes(
            context,
            queue,
            voxelise_mesh_program,
            delta,
            obstacle.get("geometry_inputs"),
            bake_path,
        )

        for entry in base_masks:
            times, matrices, rates = helper.prepare_matrix_data(entry["mesh_object"])

            if has_reference_frame:
                times, matrices, rates = helper.make_matrix_data_relative(
                    times,
                    matrices,
                    *reference_matrix_data,
                )

            entry["matrix_times"] = times
            entry["matrix_matrices"] = matrices
            entry["matrix_rates"] = rates

        obstacle_base_masks.extend(base_masks)

    obstacle_mask = helper.zeros_device(context, sparse_pool_shape, dtype=np.bool_)

    # add times
    for entry in sources + obstacles:
        for obj in entry.get("geometry_inputs") or []:
            obj["animation_timeline"] = simulation["animation_timeline"]

    # ------------multigrid------------------
    (
        p_levels,
        b_levels,
        delta_levels,
        multigrid_tile_maps,
        multigrid_active_tiles,
        multigrid_active_tile_counts,
        multigrid_level_shapes,
    ) = multigrid.create_multigrid_levels(
        context,
        shape,
        delta,
        sparse_tile_capacity,
        min_size=8,
    )

    # ------------output------------------
    output_cfg = ((simulation.get("outputs") or [None])[0]) or {}
    output_time_step = 1.0 / int(output_cfg.get("fps", 24))
    # Stuff missing!!!!!!!!!!!!!!!!

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

        physics_values = get_simulation_values(simulation, t)
        source_values = get_source_values(simulation, t)

        # ------------reference frame-------------------
        if has_reference_frame:
            reference_matrix, _ = helper.get_matrix_data(
                *reference_matrix_data,
                t,
            )

            inverse_reference_linear = np.linalg.inv(reference_matrix)[:3, :3].astype(
                FIELD_DTYPE
            )

            helper.transform_source_velocities(
                source_values,
                inverse_reference_linear,
            )
        else:
            inverse_reference_linear = np.eye(3, dtype=FIELD_DTYPE)

        gravity = inverse_reference_linear @ np.asarray(
            (0.0, 0.0, 9.81), dtype=FIELD_DTYPE
        )

        # ------------Clear scratch-------------------
        sparse_managment.reset_pools(
            queue,
            (scratch_A, scratch_B, scratch_C),
            zero_pool,
        )

        # ------------time step-------------------
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
