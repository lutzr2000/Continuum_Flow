from time import perf_counter

voxelise_mesh_module = None
gpu_kernel_module = None
np = None


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


def _cancel_requested(config):
    cancel_flag_path = ((config.get("meta") or {}).get("cancel_flag_path") or "").strip()
    return bool(cancel_flag_path) and Path(cancel_flag_path).exists()


def main(config=None):
    global np, voxelise_mesh_module, gpu_kernel_module

    total_start_time = perf_counter()
    try:
        if np is None:
            import numpy as _np
            np = _np
        if voxelise_mesh_module is None:
            import Solver.General.voxelise_mesh as _voxelise_mesh_module
            voxelise_mesh_module = _voxelise_mesh_module
        voxelise_mesh_module.set_config_dir(config.get("bake_directory", ""))
        if gpu_kernel_module is None:
            import Solver.Kernel_GPU.kernel as _gpu_kernel_module
            gpu_kernel_module = _gpu_kernel_module

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
        if _cancel_requested(config):
            print("Bake cancelled during preprocessing.")
            return

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

            if _cancel_requested(config):
                print("Bake cancelled during preprocessing.")
                return

        return gpu_kernel_module.solver(
            config,
            obstacle_base_masks,
            obstacle_mask,
            source_base_masks,
            source_masks,
            animated_obstacles,
            animated_sources,
        )
    finally:
        total_runtime = perf_counter() - total_start_time
        print(f"Bake runtime: {total_runtime:.3f} s")
        print("################################################################")

if __name__ == "__main__":
    import json
    import sys
    import traceback
    from pathlib import Path

    try:
        if len(sys.argv) < 2:
            raise ValueError("Expected bake directory path as first argument.")

        bake_directory = Path(sys.argv[1]).resolve()
        config = json.load(sys.stdin)
        config["bake_directory"] = str(bake_directory)
        main(config)

    except Exception:
        traceback.print_exc()
        sys.exit(1)
