from __future__ import annotations

import numpy as np
import bpy


def _iter_source_objects(geometry_nodes):
    """Yield valid Blender objects referenced by geometry nodes."""
    for geometry_node in geometry_nodes:
        source_object = getattr(geometry_node, "source_object", None)
        if source_object is None:
            continue
        yield source_object


def _object_to_worldspace_triangles(source_object, depsgraph=None):
    """
    Return a triangulated world-space vertex array for one Blender object.

    The evaluated object is used so modifiers are baked into the exported mesh.
    """
    if source_object is None:
        return np.empty((0, 3, 3), dtype=np.float32)

    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()

    object_eval = source_object.evaluated_get(depsgraph)
    mesh = object_eval.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
    if mesh is None:
        return np.empty((0, 3, 3), dtype=np.float32)

    try:
        mesh.calc_loop_triangles()
        if not mesh.loop_triangles:
            return np.empty((0, 3, 3), dtype=np.float32)

        matrix_world = object_eval.matrix_world.copy()
        world_vertices = np.asarray(
            [(matrix_world @ vertex.co)[:] for vertex in mesh.vertices],
            dtype=np.float32,
        )
        triangles = np.asarray(
            [[world_vertices[index] for index in triangle.vertices] for triangle in mesh.loop_triangles],
            dtype=np.float32,
        )
        return triangles
    finally:
        object_eval.to_mesh_clear()


def export_object_geometry(source_object, depsgraph=None):
    """Serialize one Blender object as world-space triangle data."""
    triangles = _object_to_worldspace_triangles(source_object, depsgraph=depsgraph)
    if triangles.size == 0:
        bounds_min = [0.0, 0.0, 0.0]
        bounds_max = [0.0, 0.0, 0.0]
    else:
        flat_vertices = triangles.reshape(-1, 3)
        bounds_min = flat_vertices.min(axis=0).astype(np.float32).tolist()
        bounds_max = flat_vertices.max(axis=0).astype(np.float32).tolist()

    return {
        "object_name": source_object.name if source_object is not None else None,
        "triangle_count": int(triangles.shape[0]),
        "bounds": {
            "min": bounds_min,
            "max": bounds_max,
        },
        "triangles": triangles.tolist(),
    }


def export_geometry_nodes(geometry_nodes, depsgraph=None):
    """Serialize all linked geometry nodes into obstacle-ready triangle payloads."""
    exports = []
    for source_object in _iter_source_objects(geometry_nodes):
        exports.append(export_object_geometry(source_object, depsgraph=depsgraph))
    return exports
