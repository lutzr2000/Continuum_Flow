import math
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

from Solver.General.main import emit_message
import Solver.General.forces as forces

import Solver.Kernel_CPU.kernel_config as kernel_config
import Solver.Kernel_CPU.multigrid as multigrid
import Solver.Kernel_CPU.output as output
import Solver.Kernel_CPU.pressure_solve as pressure_solve
import Solver.Kernel_CPU.scalar_update as scalar_update
import Solver.Kernel_CPU.sparse_managment as sparse_managment
import Solver.Kernel_CPU.time_step as time_step
from Solver.Kernel_CPU.timing import profiled_run
import Solver.Kernel_CPU.update_masks as update_masks
import Solver.Kernel_CPU.velocity_update as velocity_update
import Solver.Kernel_CPU.vorticity as vorticity
import Solver.Kernel_CPU.voxelise_mesh as voxelise_mesh

import Solver.Kernel_CPU.Boundary_Conditions.domain_bc as BC
import Solver.Kernel_CPU.Boundary_Conditions.obstacle_bc as obstacle_bc
import Solver.Kernel_CPU.Boundary_Conditions.source_bc as source_bc

warnings.filterwarnings("ignore")

CPU_FIELD_DTYPE = kernel_config.CPU_FIELD_DTYPE


def get_source_values(
    simulation: dict[str, Any],
    var_name: str,
    t: float,
    index: int | None = None,
    dtype: DTypeLike = CPU_FIELD_DTYPE,
) -> NDArray:
    """
    Resolve source values for a simulation variable at a given time.

    For each configured source, the function reads ``var_name`` from the
    source definition. If animation data is available for that variable,
    the value from the animation sample nearest to ``t`` is used instead.
    """
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

    return (
        total_u * inv_count,
        total_v * inv_count,
        total_w * inv_count,
    )


def is_animated(base_masks: list[dict[str, Any]]) -> bool:
    """
    Determine whether any mesh represented by the base masks is animated.

    A mesh is considered animated when its transform animation contains
    multiple world matrices and at least one matrix differs from the first
    sample. Repeated samples of an identical transform are therefore treated
    as static.
    """
    for entry in base_masks:
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


@profiled_run
def solver(
    config: dict,
    *,
    timings: Any = None,
) -> None:
    r"""
    The solver initializes sparse tile storage, voxelizes source and obstacle
    geometry, allocates the required CPU fields, sets up the multigrid pressure
    solver, and initializes the asynchronous output pipeline.

    During each simulation step, the solver performs the following operations:

    1. Reset temporary scratch pools.
    2. Update the source tile activity mask.
    3. Update sparse tile activity, release inactive tile slots, reuse freed
       slots, and grow CPU field pools when additional capacity is required.
    4. Update animated source masks and the obstacle mask.
    5. Compute the timestep from the current velocity field using the configured
       CFL number.
    6. Apply domain, obstacle, and source boundary conditions.
    7. Reset temporary scratch pools before the physics update.
    8. Compute vorticity magnitude when vorticity confinement is enabled.
    9. Evaluate constant, swirl, and turbulence force parameters.
    10. Advect and update the velocity field using the MacCormack scheme.
    11. Solve the pressure Poisson equation using multigrid.
    12. Project the velocity field using the computed pressure.
    13. Advect and update temperature, smoke, fuel, and flame fields using the
        MacCormack scheme and configured combustion behavior.
    14. Advance simulation time and enqueue output frames whenever an output
        time is reached.
    15. Emit simulation statistics including active tile counts.

    The simulation loop also checks for a cancellation flag before each step.
    When cancellation is requested, the loop exits cleanly and the output
    pipeline is flushed and shut down.
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

    origin = (
        origin_x,
        origin_y,
        origin_z,
    )

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

    tile_shape = (
        tile_size_i,
        tile_size_j,
        tile_size_k,
    )

    total_tile_count = int(tile_size_i * tile_size_j * tile_size_k)

    if simulate_sparsely:
        tile_map = np.full(
            tile_shape,
            -1,
            dtype=np.int32,
        )

        base_tile_map = np.full(
            tile_shape,
            -1,
            dtype=np.int32,
        )

        initial_next_tile_index = 0
        initial_active_tile_count = 0

    else:
        tile_map = np.arange(
            total_tile_count,
            dtype=np.int32,
        ).reshape(tile_shape)

        base_tile_map = np.ones(
            tile_shape,
            dtype=np.int32,
        )

        initial_next_tile_index = total_tile_count
        initial_active_tile_count = total_tile_count

    with timings.section(
        "solver",
        "initialize_tile_buffers",
    ):
        free_slot_stack = np.full(
            total_tile_count,
            -1,
            dtype=np.int32,
        )

        free_slot_count = np.zeros(
            1,
            dtype=np.int32,
        )

        reused_slot_stack = np.full(
            total_tile_count,
            -1,
            dtype=np.int32,
        )

        reused_slot_count = np.zeros(
            1,
            dtype=np.int32,
        )

        next_tile_index_counter = np.asarray(
            [initial_next_tile_index],
            dtype=np.int32,
        )

        active_tile_counter = np.asarray(
            [initial_active_tile_count],
            dtype=np.int32,
        )

        active_tile_coords = np.empty((total_tile_count, 3), dtype=np.int32)
        active_tile_slots = np.empty(total_tile_count, dtype=np.int32)

        if not simulate_sparsely:
            active_tile_coords[:] = np.stack(
                np.unravel_index(np.arange(total_tile_count), tile_shape), axis=1
            )
            active_tile_slots[:] = np.arange(total_tile_count, dtype=np.int32)

    tile_growth_size = max(
        1,
        math.ceil(
            total_tile_count * (float(kernel_config.SPARSE_TILE_GROWTH_PERCENT) / 100.0)
        ),
    )

    print("################################################################")
    print("Initialise")
    print("Total tiles: ", total_tile_count)
    print(
        "Maximum number of cells: ",
        total_tile_count * kernel_config.TILE_SIZE**3,
    )

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

    # masks
    bake_path = simulation["outputs"][0]["output_path"]

    sources = simulation.get("sources") or []
    obstacles = simulation.get("obstacles") or []

    # add times
    for entry in sources + obstacles:
        for obj in entry.get("geometry_inputs") or []:
            obj["animation_timeline"] = simulation["animation_timeline"]

    with timings.section(
        "solver",
        "voxelise_mesh.source_masks",
    ):
        source_base_masks = []

        for source in sources:
            source_base_masks.append(
                voxelise_mesh.voxelise_all_meshes(
                    delta,
                    source.get("geometry_inputs"),
                    bake_path,
                )
            )

    with timings.section(
        "solver",
        "voxelise_mesh.obstacle_masks",
    ):
        obstacle_base_masks = []

        for obstacle in obstacles:
            obstacle_base_masks.extend(
                voxelise_mesh.voxelise_all_meshes(
                    delta,
                    obstacle.get("geometry_inputs"),
                    bake_path,
                )
            )

    for base_masks in source_base_masks:
        for entry in base_masks:
            times, matrices, rates = update_masks.prepare_matrix_data(
                entry["mesh_object"]
            )

            entry["matrix_times"] = times
            entry["matrix_matrices"] = matrices
            entry["matrix_rates"] = rates

    for entry in obstacle_base_masks:
        times, matrices, rates = update_masks.prepare_matrix_data(entry["mesh_object"])

        entry["matrix_times"] = times
        entry["matrix_matrices"] = matrices
        entry["matrix_rates"] = rates

    with timings.section(
        "solver",
        "initialize_fields_and_masks",
    ):
        zero_pool = np.zeros(
            sparse_pool_shape,
            dtype=CPU_FIELD_DTYPE,
        )

        # velocity
        u_initial, v_initial, w_initial = compute_inital_velocity(simulation)

        u = np.full(
            sparse_pool_shape,
            u_initial,
            dtype=CPU_FIELD_DTYPE,
        )

        v = np.full(
            sparse_pool_shape,
            v_initial,
            dtype=CPU_FIELD_DTYPE,
        )

        w = np.full(
            sparse_pool_shape,
            w_initial,
            dtype=CPU_FIELD_DTYPE,
        )

        u_work = np.full(
            sparse_pool_shape,
            u_initial,
            dtype=CPU_FIELD_DTYPE,
        )

        v_work = np.full(
            sparse_pool_shape,
            v_initial,
            dtype=CPU_FIELD_DTYPE,
        )

        w_work = np.full(
            sparse_pool_shape,
            w_initial,
            dtype=CPU_FIELD_DTYPE,
        )

        velocity_maxima = np.zeros(
            3,
            dtype=CPU_FIELD_DTYPE,
        )

        # scalars
        temperature = np.full(
            sparse_pool_shape,
            reference_temperature,
            dtype=CPU_FIELD_DTYPE,
        )

        smoke = np.zeros(
            sparse_pool_shape,
            dtype=CPU_FIELD_DTYPE,
        )

        fuel = np.zeros(
            sparse_pool_shape,
            dtype=CPU_FIELD_DTYPE,
        )

        temperature_work = np.full(
            sparse_pool_shape,
            reference_temperature,
            dtype=CPU_FIELD_DTYPE,
        )

        smoke_work = np.zeros(
            sparse_pool_shape,
            dtype=CPU_FIELD_DTYPE,
        )

        fuel_work = np.zeros(
            sparse_pool_shape,
            dtype=CPU_FIELD_DTYPE,
        )

        # flame
        flame = np.zeros(
            sparse_pool_shape,
            dtype=CPU_FIELD_DTYPE,
        )

        # vorticity
        vorticity_magnitude = np.zeros(
            sparse_pool_shape,
            dtype=CPU_FIELD_DTYPE,
        )

        # scratch
        scratch_A = np.full(
            sparse_pool_shape,
            reference_temperature,
            dtype=CPU_FIELD_DTYPE,
        )

        scratch_B = np.zeros(
            sparse_pool_shape,
            dtype=CPU_FIELD_DTYPE,
        )

        scratch_C = np.zeros(
            sparse_pool_shape,
            dtype=CPU_FIELD_DTYPE,
        )

        # pressure
        p = np.zeros(
            sparse_pool_shape,
            dtype=CPU_FIELD_DTYPE,
        )

        pressure_rhs = np.zeros(
            sparse_pool_shape,
            dtype=CPU_FIELD_DTYPE,
        )

        # CPU reduction stores one partial per logical tile.
        rhs_partial_sums = np.zeros(
            total_tile_count,
            dtype=CPU_FIELD_DTYPE,
        )

        rhs_partial_counts = np.zeros(
            total_tile_count,
            dtype=CPU_FIELD_DTYPE,
        )

        rhs_mean_buffer = np.zeros(
            1,
            dtype=CPU_FIELD_DTYPE,
        )

        # masks
        source_masks = []

        for _ in sources:
            source_masks.append(
                np.zeros(
                    sparse_pool_shape,
                    dtype=np.bool_,
                )
            )

        source_tile_mask = np.zeros(
            tile_shape,
            dtype=np.bool_,
        )

        # this one is needed for determining which tiles
        # are active due to source activity

        obstacle_mask = np.zeros(
            sparse_pool_shape,
            dtype=np.bool_,
        )

        animated_sources = [is_animated(base_masks) for base_masks in source_base_masks]

        has_animated_sources = any(animated_sources)

        # multigrid levels
        (
            p_levels,
            b_levels,
            delta_levels,
            zero_levels,
        ) = multigrid.create_multigrid_levels(
            shape,
            delta,
            min_size=8,
        )

    # ------------output------------------
    output_cfg = ((simulation.get("outputs") or [None])[0]) or {}

    output_time_step = 1.0 / int(
        output_cfg.get(
            "fps",
            24,
        )
    )

    with timings.section(
        "solver",
        "output.setup_output",
    ):
        shared_memory_blocks, writer_slots = output.setup_output(
            simulation,
            shape,
            tile_shape,
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

            print("Bake cancellation requested. " "Stopping the simulation cleanly...")

            break

        # ------------Clear scratch-------------------
        with timings.section(
            "solver",
            "sparse_managment.reset_pools",
        ):
            sparse_managment.reset_pools(
                (
                    scratch_A,
                    scratch_B,
                    scratch_C,
                ),
                zero_pool,
                next_tile_index_counter_host,
            )

        # ------------Update source tile mask-------------------
        with timings.section(
            "solver",
            "update_masks.update_source_tile_mask",
        ):
            update_masks.update_source_tile_mask(
                source_tile_mask,
                source_base_masks,
                t,
                delta,
                origin,
            )

        # ------------Start Active tiles-------------------
        if simulate_sparsely:
            with timings.section(
                "solver",
                "sparse_managment.build_activity_mask",
            ):
                sparse_managment.build_activity_mask(
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

            active_tile_counter[0] = 0
            free_slot_count[0] = 0
            reused_slot_count[0] = 0

            with timings.section(
                "solver",
                "sparse_managment.release_inactive_tile_slots",
            ):
                sparse_managment.release_inactive_tile_slots(
                    base_tile_map,
                    tile_map,
                    kernel_config.TILE_DILATE,
                    free_slot_stack,
                    free_slot_count,
                )

            with timings.section(
                "solver",
                "sparse_managment.activate_tiles_with_reuse",
            ):
                sparse_managment.activate_tiles_with_reuse(
                    base_tile_map,
                    tile_map,
                    kernel_config.TILE_DILATE,
                    free_slot_stack,
                    free_slot_count,
                    reused_slot_stack,
                    reused_slot_count,
                    next_tile_index_counter,
                    active_tile_counter,
                    active_tile_coords,
                    active_tile_slots,
                )

            reused_slot_count_host = int(reused_slot_count[0])

            if reused_slot_count_host > 0:
                with timings.section(
                    "solver",
                    "sparse_managment.reset_reused_pool_slots",
                ):
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

            next_tile_index_counter_host = int(next_tile_index_counter[0])

            if next_tile_index_counter_host > sparse_tile_capacity:
                with timings.section(
                    "solver",
                    "sparse_managment.required_pool_capacity",
                ):
                    next_sparse_tile_capacity = sparse_managment.required_pool_capacity(
                        sparse_tile_capacity,
                        next_tile_index_counter_host,
                        tile_growth_size,
                    )

                with timings.section(
                    "solver",
                    "sparse_managment.ensure_pool_capacities",
                ):
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

            active_tile_counter_host = int(active_tile_counter[0])

        else:
            active_tile_counter_host = total_tile_count
            next_tile_index_counter_host = total_tile_count

        # ------------Update masks-------------------
        if time_step_count == 0 or has_animated_sources:
            with timings.section(
                "solver",
                "update_masks.update_source_masks",
            ):
                update_masks.update_source_masks(
                    source_masks,
                    source_base_masks,
                    animated_sources,
                    time_step_count == 0,
                    t,
                    delta,
                    origin_x,
                    origin_y,
                    origin_z,
                    tile_map,
                )

        with timings.section(
            "solver",
            "update_masks.update_obstacle_mask",
        ):
            if obstacle_base_masks:
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
        velocity_maxima[:] = 0.0

        with timings.section(
            "solver",
            "time_step.compute_new_timestep_cpu",
        ):
            dt = time_step.compute_new_timestep_cpu(
                u,
                v,
                w,
                tile_map,
                active_tile_slots,
                active_tile_counter_host,
                velocity_maxima,
                delta,
                cfl,
                output_time_step,
            )

        # ------------BCs-------------------
        # ------------Domain BC-------------------
        bc_config = simulation.get(
            "domain",
            {},
        ).get(
            "boundary_conditions",
            {},
        )

        with timings.section(
            "solver",
            "BC.domain_bc",
        ):
            (
                u,
                v,
                w,
                p,
                temperature,
                smoke,
                fuel,
            ) = BC.domain_bc(
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
        with timings.section(
            "solver",
            "obstacle_bc.obstacle_bc",
        ):
            obstacle_bc.obstacle_bc(
                active_tile_coords,
                active_tile_slots,
                active_tile_counter_host,
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
        with timings.section(
            "solver",
            "get_source_values",
        ):
            source_temperature_values = get_source_values(
                simulation,
                "temperature",
                t,
            )

            source_smoke_values = get_source_values(
                simulation,
                "smoke",
                t,
            )

            source_fuel_values = get_source_values(
                simulation,
                "fuel",
                t,
            )

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
                simulation,
                "source_noise",
                t,
                dtype=np.bool_,
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

            source_extra_pressure = get_source_values(
                simulation,
                "extra_pressure",
                t,
            )

        for source_idx, source_mask in enumerate(source_masks):
            with timings.section(
                "solver",
                "source_bc.source_bc",
            ):
                source_bc.source_bc(
                    active_tile_coords,
                    active_tile_slots,
                    active_tile_counter_host,
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
        with timings.section(
            "solver",
            "sparse_managment.reset_pools",
        ):
            sparse_managment.reset_pools(
                (
                    scratch_A,
                    scratch_B,
                    scratch_C,
                ),
                zero_pool,
                next_tile_index_counter_host,
            )

        # ------------Vorticity-------------------
        if simulation.get("physics").get("extras").get("vorticity") > 0.0:
            with timings.section(
                "solver",
                "vorticity.compute_vorticity",
            ):
                vorticity.compute_vorticity(
                    active_tile_coords,
                    active_tile_slots,
                    active_tile_counter_host,
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
        with timings.section(
            "solver",
            "get_force_params",
        ):
            fx_const, fy_const, fz_const = forces.constant_force(
                simulation,
                t,
            )

            swirl_config, has_swirl_nodes = forces.swirl_force(
                simulation,
                t,
            )

            (
                turbulence_config,
                has_turbulence_nodes,
            ) = forces.turbulence_force(
                simulation,
                t,
            )

            swirl_config_cpu = np.ascontiguousarray(
                np.asarray(
                    swirl_config,
                    dtype=CPU_FIELD_DTYPE,
                ).reshape((-1, 8))
            )

            turbulence_config_cpu = np.ascontiguousarray(
                np.asarray(
                    turbulence_config,
                    dtype=CPU_FIELD_DTYPE,
                ).reshape((-1, 4))
            )

        # ------------Velocity update-------------------
        with timings.section(
            "solver",
            "sparse_managment.copy_pools",
        ):
            sparse_managment.copy_pools(
                (
                    (u_work, u),
                    (v_work, v),
                    (w_work, w),
                ),
                next_tile_index_counter_host,
            )

        with timings.section(
            "solver",
            "velocity_update.advect_velocity_semi_lagrangian",
        ):
            velocity_update.advect_velocity_semi_lagrangian(
                active_tile_coords,
                active_tile_slots,
                active_tile_counter_host,
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

        with timings.section(
            "solver",
            "velocity_update.update_velocity_maccormack",
        ):
            velocity_update.update_velocity_maccormack(
                active_tile_coords,
                active_tile_slots,
                active_tile_counter_host,
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
                swirl_config_cpu,
                origin_x,
                origin_y,
                origin_z,
                has_turbulence_nodes,
                turbulence_config_cpu,
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
        with timings.section(
            "solver",
            "pressure_solve.pressure_poisson_multigrid",
        ):
            p = pressure_solve.pressure_poisson_multigrid(
                active_tile_coords,
                active_tile_slots,
                active_tile_counter_host,
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
                source_extra_pressure,
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
                rhs_partial_sums,
                rhs_partial_counts,
                rhs_mean_buffer,
                zero_levels,
                nx,
                ny,
                nz,
                timings=timings,
            )

        # ------------Velocity projection-------------------
        with timings.section(
            "solver",
            "pressure_solve.project_velocity_kernel",
        ):
            pressure_solve.project_velocity_kernel(
                active_tile_coords,
                active_tile_slots,
                active_tile_counter_host,
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
        with timings.section(
            "solver",
            "scalar_update.predict_scalar_fields_semi_lagrangian",
        ):
            scalar_update.predict_scalar_fields_semi_lagrangian(
                active_tile_coords,
                active_tile_slots,
                active_tile_counter_host,
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

        with timings.section(
            "solver",
            "scalar_update.update_scalar_fields_maccormack",
        ):
            scalar_update.update_scalar_fields_maccormack(
                active_tile_coords,
                active_tile_slots,
                active_tile_counter_host,
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
        (
            temperature,
            temperature_work,
        ) = (
            temperature_work,
            temperature,
        )

        smoke, smoke_work = (
            smoke_work,
            smoke,
        )

        fuel, fuel_work = (
            fuel_work,
            fuel,
        )

        # ------------time updated-------------------
        t = t + dt
        time_step_count += 1

        # ------------Output-------------------
        cpu_fields = {
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
            with timings.section(
                "solver",
                "output.enqueue_device_output",
            ):
                output.enqueue_device_output(
                    simulation,
                    writer_slots,
                    cpu_fields,
                    tile_map,
                    kernel_config.TILE_SIZE,
                    active_tile_counter_host,
                    next_tile_index_counter_host,
                    output_index,
                    t,
                )

            emit_message(
                {
                    "type": "stats",
                    "frame": output_index,
                    "active_tiles": active_tile_counter_host,
                    "total_tiles": total_tile_count,
                    "active_cells": (
                        active_tile_counter_host * kernel_config.TILE_SIZE**3
                    ),
                    "total_cells": (total_tile_count * kernel_config.TILE_SIZE**3),
                }
            )

            output_index += 1
            next_output_time += output_time_step

    # ------------Shutdown output-------------------
    with timings.section(
        "solver",
        "output.shutdown_output",
    ):
        output.shutdown_output(
            shared_memory_blocks,
            writer_slots,
        )

    # ------------Conclusion-------------------
    if cancel_requested:
        print("Simulation cancelled after clean shutdown.")
    else:
        print("Simulation finished!")
