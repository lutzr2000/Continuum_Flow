import pyopencl as cl
import math
import numpy as np
from numpy.typing import NDArray
from typing import Any
from pathlib import Path
from Solver.General.main import emit_message
import Solver.General.forces as forces

import Solver.Kernel.kernel_config as kernel_config
import Solver.Kernel.voxelise_mesh as voxelise_mesh
import Solver.Kernel.multigrid as multigrid
import Solver.Kernel.particles as particles
import Solver.Kernel.update_masks as update_masks
import Solver.Kernel.time_step as time_step
import Solver.Kernel.helper as helper
import Solver.Kernel.sparse_managment as sparse_managment
import Solver.Kernel.domain_bc as domain_bc
import Solver.Kernel.pressure_solve as pressure_solve

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

    kernel_path = Path(__file__).parent / "OpenCL"

    voxelise_mesh_kernels = helper.load_program(
        context,
        kernel_path / "voxelise_mesh.cl",
    )

    update_masks_kernels = helper.load_program(
        context,
        kernel_path / "update_masks.cl",
    )

    particles_kernels = helper.load_program(
        context,
        kernel_path / "particles.cl",
    )

    sparse_managment_kernels = helper.load_program(
        context,
        kernel_path / "sparse_managment.cl",
    )

    time_step_kernels = helper.load_program(
        context,
        kernel_path / "time_step.cl",
    )

    reference_frame_kernels = helper.load_program(
        context,
        kernel_path / "reference_frame.cl",
    )

    domain_bc_kernels = helper.load_program(
        context,
        kernel_path / "domain_bc.cl",
    )

    source_bc_kernels = helper.load_program(
        context,
        kernel_path / "source_bc.cl",
    )

    obstacle_bc_kernels = helper.load_program(
        context,
        kernel_path / "obstacle_bc.cl",
    )

    vorticity_kernels = helper.load_program(
        context,
        kernel_path / "vorticity.cl",
    )

    velocity_update_kernels = helper.load_program(
        context,
        kernel_path / "velocity_update.cl",
    )

    scalar_update_kernels = helper.load_program(
        context,
        kernel_path / "scalar_update.cl",
    )

    pressure_solve_kernels = helper.load_program(
        context,
        kernel_path / "pressure_solve.cl",
    )

    multigrid_kernels = helper.load_program(
        context,
        kernel_path / "multigrid.cl",
    )

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
    dilated_tile_map_a = helper.device_array(context, tile_shape, np.int32)
    dilated_tile_map_b = helper.device_array(context, tile_shape, np.int32)

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
    sparse_counters_device = helper.zeros_device(context, 3, dtype=np.int32)
    sparse_counters_host = np.empty(3, dtype=np.int32)

    tile_growth_size = max(
        1,
        math.ceil(
            total_tile_count * (float(kernel_config.SPARSE_TILE_GROWTH_PERCENT) / 100.0)
        ),
    )

    local_work_size = kernel_config.THREADS_PER_BLOCK_3D

    global_work_size = (
        tile_shape[0] * local_work_size[0],
        tile_shape[1] * local_work_size[1],
        tile_shape[2] * local_work_size[2],
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

    partial_velocity_maxima = helper.device_array(
        context, (kernel_config.REDUCTION_THREADS_PER_BLOCK, 3), FIELD_DTYPE
    )

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
    obstacles = simulation.get("obstacles") or []

    for entry in sources + obstacles:
        for obj in entry.get("geometry_inputs") or []:
            obj["animation_timeline"] = simulation["animation_timeline"]

    source_base_masks = []
    geometry_source_masks = []

    for source in sources:
        # voxelise source meshes
        base_masks = voxelise_mesh.voxelise_all_meshes(
            context,
            queue,
            voxelise_mesh_kernels["surface"],
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
        geometry_source_masks.append(
            helper.zeros_device(context, sparse_pool_shape, dtype=np.bool_)
        )

    animated_sources = [is_animated(base_masks) for base_masks in source_base_masks]
    has_animated_sources = any(animated_sources)

    source_tile_mask = helper.zeros_device(
        context, tile_shape, dtype=np.bool_
    )  # determines which tiles are active due to source activity

    # ------------source particles------------------
    particle_source_masks = [
        helper.zeros_device(context, sparse_pool_shape, dtype=np.bool_) for _ in sources
    ]

    has_particle_sources = any(
        source.get("particle_system_inputs") for source in sources
    )

    particle_sources = (
        particles.load_particle_sources(
            context,
            sources,
            bake_path,
            reference_matrix_data if has_reference_frame else None,
        )
        if has_particle_sources
        else [[] for _ in sources]
    )

    particle_source_flags = [bool(entries) for entries in particle_sources]

    # ------------obstacle meshes------------------
    obstacle_base_masks = []

    for obstacle in obstacles:
        base_masks = voxelise_mesh.voxelise_all_meshes(
            context,
            queue,
            voxelise_mesh_kernels["surface"],
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

        # ------------Update source tile mask-------------------
        update_masks.update_source_tile_mask(
            queue,
            update_masks_kernels["mark_source_tiles"],
            source_tile_mask,
            source_base_masks,
            tile_shape,
            t,
            delta,
            origin,
        )

        if has_particle_sources:
            particles.update_source_tile_mask(
                queue,
                particles_kernels["mark_particle_tiles"],
                particles_kernels["sample_interpolated_vectors"],
                source_tile_mask,
                particle_sources,
                tile_shape,
                t,
                delta,
                origin,
            )

        # ------------Start Active tiles-------------------
        if simulate_sparsely:
            sparse_managment_kernels["build_activity_mask"](
                queue,
                tile_shape,
                None,
                smoke,
                fuel,
                flame,
                tile_map,
                source_tile_mask,
                base_tile_map,
                np.float32(sparse_threshold),
                np.int32(nx),
                np.int32(ny),
                np.int32(nz),
                np.int32(tile_shape[0]),
                np.int32(tile_shape[1]),
                np.int32(tile_shape[2]),
            )

            for kernel_name, src_map, dst_map in (
                ("dilate_activity_x", base_tile_map, dilated_tile_map_a),
                ("dilate_activity_y", dilated_tile_map_a, dilated_tile_map_b),
                ("dilate_activity_z", dilated_tile_map_b, dilated_tile_map_a),
            ):
                sparse_managment_kernels[kernel_name](
                    queue,
                    tile_shape,
                    None,
                    src_map,
                    dst_map,
                    np.int32(tile_dilate),
                    np.int32(tile_shape[0]),
                    np.int32(tile_shape[1]),
                    np.int32(tile_shape[2]),
                )

            helper.fill_device(queue, active_tile_counter, 0, dtype=np.int32)
            helper.fill_device(queue, free_slot_count, 0, dtype=np.int32)
            helper.fill_device(queue, reused_slot_count, 0, dtype=np.int32)

            sparse_managment_kernels["release_inactive_tile_slots"](
                queue,
                tile_shape,
                None,
                dilated_tile_map_a,
                tile_map,
                free_slot_stack,
                free_slot_count,
                np.int32(tile_shape[0]),
                np.int32(tile_shape[1]),
                np.int32(tile_shape[2]),
            )

            sparse_managment_kernels["activate_tiles_with_reuse"](
                queue,
                tile_shape,
                None,
                dilated_tile_map_a,
                tile_map,
                free_slot_stack,
                free_slot_count,
                reused_slot_stack,
                reused_slot_count,
                next_tile_index_counter,
                active_tile_counter,
                np.int32(tile_shape[0]),
                np.int32(tile_shape[1]),
                np.int32(tile_shape[2]),
            )

            sparse_managment_kernels["pack_sparse_counters"](
                queue,
                (1,),
                None,
                reused_slot_count,
                next_tile_index_counter,
                active_tile_counter,
                sparse_counters_device,
            )
            cl.enqueue_copy(
                queue,
                sparse_counters_host,
                sparse_counters_device,
                is_blocking=True,
            )
            reused_slot_count_host = int(sparse_counters_host[0])
            next_tile_index_counter_host = int(sparse_counters_host[1])
            active_tile_counter_host = int(sparse_counters_host[2])

            if reused_slot_count_host > 0:
                sparse_managment.reset_reused_pool_slots(
                    queue,
                    sparse_managment_kernels,
                    [
                        (u, u_initial, FIELD_DTYPE),
                        (v, v_initial, FIELD_DTYPE),
                        (w, w_initial, FIELD_DTYPE),
                        (u_work, u_initial, FIELD_DTYPE),
                        (v_work, v_initial, FIELD_DTYPE),
                        (w_work, w_initial, FIELD_DTYPE),
                        (scratch_A, reference_temperature, FIELD_DTYPE),
                        (scratch_B, 0.0, FIELD_DTYPE),
                        (scratch_C, 0.0, FIELD_DTYPE),
                        (p, 0.0, FIELD_DTYPE),
                        (pressure_rhs, 0.0, FIELD_DTYPE),
                        (temperature, reference_temperature, FIELD_DTYPE),
                        (smoke, 0.0, FIELD_DTYPE),
                        (fuel, 0.0, FIELD_DTYPE),
                        (temperature_work, reference_temperature, FIELD_DTYPE),
                        (smoke_work, 0.0, FIELD_DTYPE),
                        (fuel_work, 0.0, FIELD_DTYPE),
                        (flame, 0.0, FIELD_DTYPE),
                        (vorticity_magnitude, 0.0, FIELD_DTYPE),
                        (obstacle_mask, False, np.bool_),
                        *[(mask, False, np.bool_) for mask in geometry_source_masks],
                        *[(mask, False, np.bool_) for mask in particle_source_masks],
                    ],
                    reused_slot_stack,
                    reused_slot_count_host,
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
                    *resized_source_masks,
                ) = sparse_managment.ensure_pool_capacities(
                    context,
                    queue,
                    [
                        (u, u_initial, FIELD_DTYPE),
                        (v, v_initial, FIELD_DTYPE),
                        (w, w_initial, FIELD_DTYPE),
                        (u_work, u_initial, FIELD_DTYPE),
                        (v_work, v_initial, FIELD_DTYPE),
                        (w_work, w_initial, FIELD_DTYPE),
                        (scratch_A, reference_temperature, FIELD_DTYPE),
                        (scratch_B, 0.0, FIELD_DTYPE),
                        (scratch_C, 0.0, FIELD_DTYPE),
                        (p, 0.0, FIELD_DTYPE),
                        (pressure_rhs, 0.0, FIELD_DTYPE),
                        (temperature, reference_temperature, FIELD_DTYPE),
                        (smoke, 0.0, FIELD_DTYPE),
                        (fuel, 0.0, FIELD_DTYPE),
                        (temperature_work, reference_temperature, FIELD_DTYPE),
                        (smoke_work, 0.0, FIELD_DTYPE),
                        (fuel_work, 0.0, FIELD_DTYPE),
                        (flame, 0.0, FIELD_DTYPE),
                        (zero_pool, 0.0, FIELD_DTYPE),
                        (vorticity_magnitude, 0.0, FIELD_DTYPE),
                        (obstacle_mask, False, np.bool_),
                        *[(mask, False, np.bool_) for mask in geometry_source_masks],
                        *[(mask, False, np.bool_) for mask in particle_source_masks],
                    ],
                    sparse_tile_capacity,
                    next_sparse_tile_capacity,
                )

                source_count = len(sources)
                geometry_source_masks = resized_source_masks[:source_count]
                particle_source_masks = resized_source_masks[source_count:]

                sparse_tile_capacity = next_sparse_tile_capacity

                # needed for growning the coarse capacities simply by recreating them
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

        else:
            active_tile_counter_host = total_tile_count
            next_tile_index_counter_host = total_tile_count

        # ------------Update geometry source masks-------------------
        update_geometry_sources = time_step_count == 0 or has_animated_sources

        if update_geometry_sources:
            update_masks.update_source_masks(
                queue,
                update_masks_kernels,
                geometry_source_masks,
                source_base_masks,
                animated_sources,
                time_step_count == 0,
                t,
                delta,
                origin_x,
                origin_y,
                origin_z,
                tile_map,
                tile_shape,
            )

        # ------------Update particle source masks-------------------
        if has_particle_sources:
            particles.update_particle_source_masks(
                queue,
                particles_kernels["rasterize_particle_spheres"],
                particles_kernels["sample_interpolated_vectors"],
                particle_source_masks,
                particle_sources,
                t,
                delta,
                origin,
                tile_map,
                tile_shape,
            )

        # ------------time step-------------------
        dt = time_step.compute_new_timestep(
            queue,
            time_step_kernels,
            u,
            v,
            w,
            tile_map,
            tile_shape,
            active_tile_counter_host,
            velocity_maxima,
            partial_velocity_maxima,
            delta,
            cfl,
            output_time_step,
        )
        # ------------Reference Frame Velocity Transfer-------------------
        if reference_velocity_transfer != 0.0 and reference_frame_animated:
            reference_matrix, reference_rate = update_masks.get_matrix_data(
                *reference_matrix_data, t
            )

            current_reference_velocity_transform = (
                np.linalg.inv(reference_matrix) @ reference_rate
            )[:3, :].astype(FIELD_DTYPE)

            reference_velocity_delta = (
                current_reference_velocity_transform
                - previous_reference_velocity_transform
            ) * np.asarray(reference_velocity_transfer, dtype=FIELD_DTYPE)

            if np.any(reference_velocity_delta != 0.0):
                local_work_size = (
                    kernel_config.TILE_SIZE,
                    kernel_config.TILE_SIZE,
                    kernel_config.TILE_SIZE,
                )

                global_work_size = (
                    tile_shape[0] * kernel_config.TILE_SIZE,
                    tile_shape[1] * kernel_config.TILE_SIZE,
                    tile_shape[2] * kernel_config.TILE_SIZE,
                )

                reference_frame_kernels["transfer_velocity"](
                    queue,
                    global_work_size,
                    local_work_size,
                    u,
                    v,
                    w,
                    tile_map,
                    *[np.float32(value) for value in reference_velocity_delta.ravel()],
                    np.float32(origin_x),
                    np.float32(origin_y),
                    np.float32(origin_z),
                    np.float32(delta),
                    np.int32(nx),
                    np.int32(ny),
                    np.int32(nz),
                    np.int32(tile_shape[1]),
                    np.int32(tile_shape[2]),
                )

            previous_reference_velocity_transform = current_reference_velocity_transform

        # ------------BCs-------------------
        # ------------Domain BC-------------------
        bc_config = simulation.get("domain", {}).get("boundary_conditions", {})

        u, v, w, p, temperature, smoke, fuel = domain_bc.domain_bc(
            queue,
            domain_bc_kernels,
            u,
            v,
            w,
            p,
            temperature,
            smoke,
            fuel,
            bc_config,
            tile_map,
            tile_shape,
            reference_temperature,
            u_initial,
            v_initial,
            w_initial,
            nx,
            ny,
            nz,
        )

        # ------------Source BC-------------------
        for source_idx, source_mask in enumerate(geometry_source_masks):
            velocity_local = bool(source_values["velocity_local"][source_idx])
            if velocity_local:
                sparse_managment.reset_pools(
                    queue,
                    (scratch_A, scratch_B, scratch_C),
                    zero_pool,
                )

                update_masks.update_source_velocity(
                    queue,
                    update_masks_kernels,
                    source_base_masks[source_idx],
                    t,
                    delta,
                    origin_x,
                    origin_y,
                    origin_z,
                    tile_map,
                    tile_shape,
                    scratch_A,
                    scratch_B,
                    scratch_C,
                    source_values["velocity_x"][source_idx],
                    source_values["velocity_y"][source_idx],
                    source_values["velocity_z"][source_idx],
                )

            source_bc_kernels["source_bc"](
                queue,
                global_work_size,
                local_work_size,
                u,
                v,
                w,
                temperature,
                smoke,
                fuel,
                tile_map,
                source_mask,
                np.float32(source_values["temperature"][source_idx]),
                np.float32(source_values["smoke"][source_idx]),
                np.float32(source_values["fuel"][source_idx]),
                np.float32(source_values["velocity_x"][source_idx]),
                np.float32(source_values["velocity_y"][source_idx]),
                np.float32(source_values["velocity_z"][source_idx]),
                np.int32(velocity_local),
                scratch_A,
                scratch_B,
                scratch_C,
                np.float32(source_values["noise_scale"][source_idx]),
                np.float32(source_values["noise_amplitude"][source_idx]),
                np.int32(source_values["noise_seed"][source_idx]),
                np.float32(dt),
                np.int32(tile_shape[0]),
                np.int32(tile_shape[1]),
                np.int32(tile_shape[2]),
            )

        # ------------Particle Source BC-------------------
        if has_particle_sources:
            for source_idx, particle_source_mask in enumerate(particle_source_masks):
                if not particle_source_flags[source_idx]:
                    continue

                source_bc_kernels["source_bc"](
                    queue,
                    global_work_size,
                    local_work_size,
                    u,
                    v,
                    w,
                    temperature,
                    smoke,
                    fuel,
                    tile_map,
                    particle_source_mask,
                    np.float32(source_values["temperature"][source_idx]),
                    np.float32(source_values["smoke"][source_idx]),
                    np.float32(source_values["fuel"][source_idx]),
                    np.float32(source_values["velocity_x"][source_idx]),
                    np.float32(source_values["velocity_y"][source_idx]),
                    np.float32(source_values["velocity_z"][source_idx]),
                    np.int32(0),
                    scratch_A,
                    scratch_B,
                    scratch_C,
                    np.float32(source_values["noise_scale"][source_idx]),
                    np.float32(source_values["noise_amplitude"][source_idx]),
                    np.int32(source_values["noise_seed"][source_idx]),
                    np.float32(dt),
                    np.int32(tile_shape[0]),
                    np.int32(tile_shape[1]),
                    np.int32(tile_shape[2]),
                )

                if (
                    source_values["velocity_x"][source_idx] == 0.0
                    and source_values["velocity_y"][source_idx] == 0.0
                    and source_values["velocity_z"][source_idx] == 0.0
                ):
                    particles.reset_particle_velocity(
                        queue,
                        particles_kernels["reset_particle_velocity_kernel"],
                        u,
                        v,
                        w,
                        particle_sources[source_idx],
                        t,
                        delta,
                        origin,
                        tile_map,
                        tile_shape,
                    )

            particles.transfer_particle_velocities(
                queue,
                particles_kernels["transfer_particle_velocities"],
                u,
                v,
                w,
                particle_sources,
                t,
                delta,
                origin,
                tile_map,
                tile_shape,
            )

        # ------------Update obstacle masks-------------------
        if obstacle_base_masks:
            update_masks.update_obstacle_mask(
                queue,
                update_masks_kernels,
                obstacle_mask,
                obstacle_base_masks,
                t,
                delta,
                origin_x,
                origin_y,
                origin_z,
                tile_map,
                tile_shape,
                scratch_A,
                scratch_B,
                scratch_C,
            )

        # ------------Obstacle BC-------------------
        obstacle_bc_kernels["obstacle_bc"](
            queue,
            global_work_size,
            local_work_size,
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
            np.int32(tile_shape[0]),
            np.int32(tile_shape[1]),
            np.int32(tile_shape[2]),
        )

        # ------------Clear scratch-------------------
        sparse_managment.reset_pools(
            queue,
            (scratch_A, scratch_B, scratch_C),
            zero_pool,
        )

        # ------------Vorticity-------------------
        if physics_values["extras"]["vorticity"] > 0.0:
            vorticity_kernels["compute_vorticity"](
                queue,
                global_work_size,
                local_work_size,
                u,
                v,
                w,
                np.float32(u_initial),
                np.float32(v_initial),
                np.float32(w_initial),
                obstacle_mask,
                vorticity_magnitude,
                np.float32(delta),
                tile_map,
                np.int32(nx),
                np.int32(ny),
                np.int32(nz),
                np.int32(tile_shape[0]),
                np.int32(tile_shape[1]),
                np.int32(tile_shape[2]),
            )

        # ------------force params-------------------
        fx_const, fy_const, fz_const = forces.constant_force(simulation, t)
        swirl_config, has_swirl_nodes = forces.swirl_force(simulation, t)
        turbulence_config, has_turbulence_nodes = forces.turbulence_force(simulation, t)

        swirl_config_device = helper.to_device(
            context,
            np.ascontiguousarray(
                np.asarray(
                    swirl_config,
                    dtype=FIELD_DTYPE,
                ).reshape((-1, 8))
            ),
        )

        turbulence_config_device = helper.to_device(
            context,
            np.ascontiguousarray(
                np.asarray(
                    turbulence_config,
                    dtype=FIELD_DTYPE,
                ).reshape((-1, 4))
            ),
        )

        swirl_count = len(swirl_config)
        turbulence_count = len(turbulence_config)
        # ------------Velocity update-------------------
        sparse_managment.copy_pools(
            queue,
            (
                (u_work, u),
                (v_work, v),
                (w_work, w),
            ),
            next_tile_index_counter_host,
        )

        velocity_update_kernels["advect_velocity_semi_lagrangian"](
            queue,
            global_work_size,
            local_work_size,
            u,
            v,
            w,
            scratch_A,
            scratch_B,
            scratch_C,
            np.float32(dt),
            np.float32(delta),
            np.int32(advection_substeps),
            tile_map,
            np.float32(u_initial),
            np.float32(v_initial),
            np.float32(w_initial),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            np.int32(tile_shape[0]),
            np.int32(tile_shape[1]),
            np.int32(tile_shape[2]),
        )

        velocity_update_kernels["update_velocity_maccormack"](
            queue,
            global_work_size,
            local_work_size,
            u,
            v,
            w,
            obstacle_mask,
            scratch_A,
            scratch_B,
            scratch_C,
            np.float32(dt),
            u_work,
            v_work,
            w_work,
            np.float32(delta),
            np.float32(physics_values["fluid"]["density"]),
            np.int32(advection_substeps),
            np.float32(physics_values["fluid"]["viscosity"]),
            vorticity_magnitude,
            np.float32(physics_values["extras"]["vorticity"]),
            temperature,
            np.float32(physics_values["temperature"]["buoyancy"]),
            np.float32(reference_temperature),
            np.float32(gravity[0]),
            np.float32(gravity[1]),
            np.float32(gravity[2]),
            tile_map,
            np.float32(fx_const),
            np.float32(fy_const),
            np.float32(fz_const),
            np.int32(has_swirl_nodes),
            swirl_config_device,
            np.int32(swirl_count),
            np.float32(origin_x),
            np.float32(origin_y),
            np.float32(origin_z),
            np.int32(has_turbulence_nodes),
            turbulence_config_device,
            np.int32(turbulence_count),
            np.float32(t),
            np.float32(u_initial),
            np.float32(v_initial),
            np.float32(w_initial),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            np.int32(tile_shape[0]),
            np.int32(tile_shape[1]),
            np.int32(tile_shape[2]),
        )

        # ------------Velocity swap-------------------
        u, u_work = u_work, u
        v, v_work = v_work, v
        w, w_work = w_work, w

        # ------------Pressure solve-------------------
        p = pressure_solve.pressure_poisson_multigrid(
            pressure_solve_kernels,
            multigrid_kernels,
            queue,
            global_work_size,
            local_work_size,
            u,
            v,
            w,
            p,
            temperature,
            pressure_rhs,
            dt,
            geometry_source_masks,
            particle_source_masks,
            source_values["noise_scale"],
            source_values["noise_amplitude"],
            source_values["noise_seed"],
            source_values["extra_pressure"],
            delta,
            physics_values["fluid"]["density"],
            physics_values["temperature"]["expansion_rate"],
            reference_temperature,
            tile_map,
            tile_shape,
            u_initial,
            v_initial,
            w_initial,
            p_levels,
            b_levels,
            delta_levels,
            multigrid_tile_maps,
            multigrid_active_tiles,
            multigrid_active_tile_counts,
            multigrid_level_shapes,
            simulation.get("settings").get("iterations"),
            rhs_partial_sums,
            rhs_partial_counts,
            rhs_mean_buffer,
            nx,
            ny,
            nz,
        )
        # ------------Velocity projection-------------------
        pressure_solve_kernels["project_velocity_kernel"](
            queue,
            global_work_size,
            local_work_size,
            u,
            v,
            w,
            p,
            obstacle_mask,
            np.float32(dt),
            np.float32(delta),
            np.float32(physics_values["fluid"]["density"]),
            tile_map,
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            np.int32(tile_shape[0]),
            np.int32(tile_shape[1]),
            np.int32(tile_shape[2]),
        )

        # ------------Scalar update-------------------
        scalar_update_kernels["predict_scalar_fields_semi_lagrangian"](
            queue,
            global_work_size,
            local_work_size,
            temperature,
            smoke,
            fuel,
            u,
            v,
            w,
            np.float32(dt),
            scratch_A,
            scratch_B,
            scratch_C,
            np.float32(delta),
            np.int32(advection_substeps),
            np.float32(reference_temperature),
            tile_map,
            np.float32(u_initial),
            np.float32(v_initial),
            np.float32(w_initial),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            np.int32(tile_shape[0]),
            np.int32(tile_shape[1]),
            np.int32(tile_shape[2]),
        )

        scalar_update_kernels["update_scalar_fields_maccormack"](
            queue,
            global_work_size,
            local_work_size,
            temperature,
            smoke,
            fuel,
            scratch_A,
            scratch_B,
            scratch_C,
            u,
            v,
            w,
            np.float32(dt),
            temperature_work,
            smoke_work,
            fuel_work,
            flame,
            np.float32(delta),
            np.int32(advection_substeps),
            np.float32(physics_values["temperature"]["dissipation"]),
            np.float32(physics_values["temperature"]["production_rate"]),
            np.float32(physics_values["smoke"]["dissipation"]),
            np.float32(physics_values["smoke"]["production_rate"]),
            np.float32(physics_values["fuel"]["dissipation"]),
            np.float32(physics_values["fuel"]["burn_rate"]),
            np.float32(physics_values["fuel"]["ignition_temperature"]),
            np.float32(physics_values["burning"]["scale"]),
            np.float32(physics_values["burning"]["amplitude"]),
            np.float32(reference_temperature),
            tile_map,
            np.float32(u_initial),
            np.float32(v_initial),
            np.float32(w_initial),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            np.int32(tile_shape[0]),
            np.int32(tile_shape[1]),
            np.int32(tile_shape[2]),
        )

        # ------------Swap-------------------
        temperature, temperature_work = temperature_work, temperature
        smoke, smoke_work = smoke_work, smoke
        fuel, fuel_work = fuel_work, fuel

        # ------------time update-------------------
        t = t + dt
        time_step_count += 1

        # ------------Output-------------------
        while t >= next_output_time:
            output_index += 1
            next_output_time += output_time_step

            # ------------(V)RAM Track-------------------
            allocated_vram = helper.opencl_buffer_bytes(*locals().values())

            total_vram = device.global_mem_size

            emit_message(
                {
                    "type": "stats",
                    "frame": output_index,
                    "active_tiles": active_tile_counter_host,
                    "total_tiles": total_tile_count,
                    "active_cells": active_tile_counter_host
                    * kernel_config.TILE_SIZE**3,
                    "total_cells": total_tile_count * kernel_config.TILE_SIZE**3,
                    "vram_used_mb": allocated_vram / 1024**2,
                    "vram_total_mb": total_vram / 1024**2,
                }
            )

    # ------------Conclusion-------------------
    if cancel_requested:
        print("Simulation cancelled after clean shutdown.")
    else:
        print("Simulation finished!")
