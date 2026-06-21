import Solver.General.voxelise_mesh as voxelise_mesh_module
import Solver.Kernel_GPU.kernel as gpu_kernel_module


def _collect_mesh_objects(entries):
    mesh_objects = []
    for entry in entries or ():
        mesh_cfg = entry.get("mesh", {})
        mesh_objects.extend(mesh_cfg.get("objects", ()))
    return mesh_objects


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

    viewers = simulation_cfg.get("viewers") or []
    debug_enabled = any(bool(viewer_cfg.get("debug", False)) for viewer_cfg in viewers)
    solver_fn = gpu_kernel_module.solver_debug if debug_enabled else gpu_kernel_module.solver

    return solver_fn(
        config,
        obstacle_base_masks,
        obstacle_mask,
        source_base_masks,
        source_masks,
    )
