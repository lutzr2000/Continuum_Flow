from __future__ import annotations

from pathlib import Path

import numpy as np
import bpy


def _iter_source_objects(geometry_nodes):
    """
    Yield valid Blender objects referenced by geometry nodes.
    """
    for geometry_node in geometry_nodes:
        source_object = getattr(geometry_node, "source_object", None)
        if source_object is None:
            continue
        yield source_object


def _object_to_localspace_triangles(source_object, depsgraph=None):
    """
    Return a triangulated local-space vertex array for one Blender object.

    The evaluated object is used so modifiers are baked into the exported mesh
    while the object transform itself stays separate for runtime playback.
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
        triangle_count = len(mesh.loop_triangles)
        vertex_count = len(mesh.vertices)
        if triangle_count == 0 or vertex_count == 0:
            return np.empty((0, 3, 3), dtype=np.float32)

        local_vertices = np.empty(vertex_count * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", local_vertices)
        local_vertices = local_vertices.reshape(vertex_count, 3)

        triangle_vertex_indices = np.empty(triangle_count * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", triangle_vertex_indices)
        triangle_vertex_indices = triangle_vertex_indices.reshape(triangle_count, 3)

        return np.ascontiguousarray(
            local_vertices[triangle_vertex_indices], dtype=np.float32
        )
    finally:
        object_eval.to_mesh_clear()


def _sanitize_file_stem(name):
    """
    Return a filesystem-friendly file stem for one Blender object name.
    """
    safe_name = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in str(name)
    )
    safe_name = safe_name.strip("._")
    return safe_name or "mesh"


def _serialize_triangles(triangles, source_object=None, storage_dir=None):
    """
    Serialize triangles either inline for JSON or to a binary cache file for bake startup.
    """
    if storage_dir is None:
        return {
            "triangles": triangles.tolist(),
        }

    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    object_name = (
        getattr(source_object, "name", None) if source_object is not None else None
    )
    triangle_file_path = storage_dir / f"{_sanitize_file_stem(object_name)}.npy"
    suffix = 1
    while triangle_file_path.exists():
        triangle_file_path = (
            storage_dir / f"{_sanitize_file_stem(object_name)}_{suffix}.npy"
        )
        suffix += 1
    np.save(
        triangle_file_path,
        np.ascontiguousarray(triangles, dtype=np.float32),
        allow_pickle=False,
    )
    return {
        "triangles_file": str(triangle_file_path),
        "triangles_shape": [int(axis_length) for axis_length in triangles.shape],
    }


def export_object_geometry(source_object, depsgraph=None, storage_dir=None):
    """
    Serialize one Blender object as local-space triangle data.
    """
    triangles = _object_to_localspace_triangles(source_object, depsgraph=depsgraph)
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
        **_serialize_triangles(
            triangles, source_object=source_object, storage_dir=storage_dir
        ),
    }


def export_geometry_nodes(geometry_nodes, depsgraph=None, storage_dir=None):
    """
    Serialize all linked geometry nodes into obstacle-ready triangle payloads.
    """
    exports = []
    for source_object in _iter_source_objects(geometry_nodes):
        exports.append(
            export_object_geometry(
                source_object, depsgraph=depsgraph, storage_dir=storage_dir
            )
        )
    return exports
