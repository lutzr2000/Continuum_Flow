from pathlib import Path
import contextlib
import io
import os
import re
from time import perf_counter
import bpy

# -------------- avoid UI redraw ----------------
@contextlib.contextmanager
def suspend_continuum_frame_handler():
    """Avoid node and UI updates while the exporter evaluates animation frames."""
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


@contextlib.contextmanager
def suppress_export_output():
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull_fd)


# -------------- USD export ----------------
def sanitize_export_name(name, fallback="geometry"):
    sanitized = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(name or fallback).strip(),
    )
    sanitized = sanitized.strip("._-")
    return sanitized or fallback


def export_usdc(config_dict, export_directory):
    export_start_time = perf_counter()
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
        export_objects_as_usdc(
            obstacle_objects,
            geometry_dir / "obstacles.usdc",
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
        export_objects_as_usdc(
            source_objects,
            geometry_dir / f"{source_name}.usdc",
            start_frame=start_frame,
            end_frame=export_end_frame,
        )

    print(f"Geometry export time: {perf_counter() - export_start_time:.2f} s")


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


def export_objects_as_usdc(source_objects, file_path, start_frame, end_frame):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    active_object = getattr(bpy.context.view_layer.objects, "active", None)
    selected_objects = list(getattr(bpy.context, "selected_objects", ()))
    scene = bpy.context.scene
    original_frame_start = scene.frame_start
    original_frame_end = scene.frame_end

    try:
        bpy.ops.object.select_all(action="DESELECT")
        for source_object in source_objects:
            source_object.select_set(True)

        if source_objects:
            bpy.context.view_layer.objects.active = source_objects[0]

        with suspend_continuum_frame_handler():
            scene.frame_start = int(start_frame)
            scene.frame_end = int(end_frame)
            with suppress_export_output():
                bpy.ops.wm.usd_export(
                    filepath=str(file_path),
                    selected_objects_only=True,
                    export_animation=True,
                    export_meshes=True,
                    export_lights=False,
                    export_cameras=False,
                    export_curves=False,
                    export_points=False,
                    export_volumes=False,
                    export_hair=False,
                    export_armatures=False,
                    export_materials=False,
                    export_normals=False,
                    export_uvmaps=False,
                    use_instancing=False,
                    triangulate_meshes=False,
                )

    finally:
        scene.frame_start = original_frame_start
        scene.frame_end = original_frame_end
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
