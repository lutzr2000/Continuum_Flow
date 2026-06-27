import numpy as np

import Solver.General.voxelise_mesh as voxelise_mesh_module
import Solver.Kernel_GPU.kernel as gpu_kernel_module


def _collect_mesh_objects(entries):
    mesh_objects = []

    for entry in entries or ():
        for geometry_input in entry.get("geometry_inputs", ()):
            if geometry_input.get("mesh_file"):
                mesh_objects.append(geometry_input)

    return mesh_objects


def _has_animated_mesh_objects(mesh_objects):
    for mesh_object in mesh_objects or ():
        transform_animation = mesh_object.get("transform_animation") or {}
        matrices = transform_animation.get("matrices_world") or ()
        if len(matrices) <= 1:
            continue

        first_matrix = np.asarray(matrices[0], dtype=np.float32)
        if any(
            not np.allclose(np.asarray(matrix, dtype=np.float32), first_matrix)
            for matrix in matrices[1:]
        ):
            return True

    return False


def main(config=None):

    simulations = config.get("simulations") or []

    simulation_cfg = simulations[0]
    domain_cfg = simulation_cfg.get("domain") or {}
    grid_cfg = domain_cfg.get("grid") or {}

    nx = int(grid_cfg.get("nx", 0))
    ny = int(grid_cfg.get("ny", 0))
    nz = int(grid_cfg.get("nz", 0))
    delta = float(domain_cfg.get("resolution", 0.0))
    origin_x = -0.5 * nx * delta
    origin_y = -0.5 * ny * delta
    origin_z = 0.0

    obstacle_mesh_objects = _collect_mesh_objects(simulation_cfg.get("obstacles"))
    source_entries = simulation_cfg.get("sources") or []
    source_mesh_objects = _collect_mesh_objects(source_entries)

    animated_obstacles = bool(obstacle_mesh_objects) and _has_animated_mesh_objects(
        obstacle_mesh_objects
    )
    animated_sources = bool(source_mesh_objects) and _has_animated_mesh_objects(
        source_mesh_objects
    )

    obstacle_base_masks, obstacle_mask = voxelise_mesh_module.voxelise_mesh_all(
        nx,
        ny,
        nz,
        delta,
        obstacle_mesh_objects,
        origin_x=origin_x,
        origin_y=origin_y,
        origin_z=origin_z,
    )

    source_base_masks = []
    source_masks = []
    for source_entry in source_entries:
        source_mesh_objects = _collect_mesh_objects([source_entry])
        source_base_mask, source_mask = voxelise_mesh_module.voxelise_mesh_all(
            nx,
            ny,
            nz,
            delta,
            source_mesh_objects,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_z=origin_z,
        )
        source_base_masks.append(source_base_mask)
        source_masks.append(source_mask)

    return gpu_kernel_module.solver(
        config,
        obstacle_base_masks,
        obstacle_mask,
        source_base_masks,
        source_masks,
        animated_obstacles,
        animated_sources,
    )

if __name__ == "__main__":
    import json
    import sys
    import traceback
    from pathlib import Path

    try:
        if len(sys.argv) < 2:
            raise ValueError("Expected bake directory path as first argument.")

        bake_directory = Path(sys.argv[1]).resolve()
        voxelise_mesh_module.set_config_dir(bake_directory)

        config = json.load(sys.stdin)
        main(config)

    except Exception:
        traceback.print_exc()
        sys.exit(1)
