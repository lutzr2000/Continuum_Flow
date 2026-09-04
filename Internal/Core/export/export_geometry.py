from pathlib import Path
import contextlib
import re
import bpy

# -------------- avoid UI redraw ----------------
@contextlib.contextmanager
def suspend_continuum_frame_handler():
    """Avoid node and UI updates while Alembic evaluates every export frame."""
    from . import export_config

    handler = export_config.continuum_flow_frame_change_post
    handlers = bpy.app.handlers.frame_change_post
    was_registered = handler in handlers

    if was_registered:
        handlers.remove(handler)

    try:
        yield
    finally:
        if was_registered and handler not in handlers:
            handlers.append(handler)


# -------------- Alembic export ----------------
def sanitize_export_name(name, fallback="geometry"):
    sanitized = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(name or fallback).strip(),
    )
    sanitized = sanitized.strip("._-")
    return sanitized or fallback


def export_alembics(config_dict, export_directory):
    geometry_dir = Path(export_directory) / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    simulation = config_dict.get("simulation") or {}
    simulation_settings = simulation.get("settings") or {}
    start_frame = int(
        simulation_settings.get("start_frame", getattr(scene, "frame_current", 1))
    )
    end_frame = int(simulation_settings.get("end_frame", start_frame + 1))
    export_end_frame = max(start_frame, end_frame - 1)

    obstacle_objects = collect_group_objects(simulation.get("obstacles", []))
    if obstacle_objects:
        export_objects_as_alembic(
            obstacle_objects,
            geometry_dir / "obstacles.abc",
            start_frame=start_frame,
            end_frame=export_end_frame,
        )

    for source_entry in simulation.get("sources", []):
        source_objects = collect_entry_objects(source_entry)
        if not source_objects:
            continue

        source_name = sanitize_export_name(
            source_entry.get("node_name"),
            fallback="source",
        )
        export_objects_as_alembic(
            source_objects,
            geometry_dir / f"{source_name}.abc",
            start_frame=start_frame,
            end_frame=export_end_frame,
        )


def collect_group_objects(entries):
    collected = []
    seen = set()

    for entry in entries:
        for source_object in collect_entry_objects(entry):
            if source_object.name in seen:
                continue
            seen.add(source_object.name)
            collected.append(source_object)

    return collected


def collect_entry_objects(entry):
    collected = []
    seen = set()

    for geometry_input in entry.get("geometry_inputs", []):
        object_name = geometry_input.get("object_name")
        if not object_name or object_name in seen:
            continue

        source_object = bpy.data.objects.get(object_name)
        if source_object is None:
            continue

        seen.add(object_name)
        collected.append(source_object)

    return collected


def export_objects_as_alembic(source_objects, file_path, start_frame, end_frame):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    active_object = getattr(bpy.context.view_layer.objects, "active", None)
    selected_objects = list(getattr(bpy.context, "selected_objects", ()))

    try:
        bpy.ops.object.select_all(action="DESELECT")
        for source_object in source_objects:
            source_object.select_set(True)

        if source_objects:
            bpy.context.view_layer.objects.active = source_objects[0]

        with suspend_continuum_frame_handler():
            bpy.ops.wm.alembic_export(
                filepath=str(file_path),
                start=int(start_frame),
                end=int(end_frame),
                selected=True,
                flatten=False,
                uvs=False,
                normals=False,
                vcolors=False,
                apply_subdiv=False,
                curves_as_mesh=False,
                use_instancing=False,
                triangulate=True,
                as_background_job=False,
            )

    finally:
        bpy.ops.object.select_all(action="DESELECT")

        for selected_object in selected_objects:
            try:
                if selected_object.name in bpy.data.objects:
                    selected_object.select_set(True)
            except Exception:
                pass

        try:
            bpy.context.view_layer.objects.active = active_object
        except Exception:
            pass
