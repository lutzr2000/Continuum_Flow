from pathlib import Path
import contextlib
import io
import os
import bpy


@contextlib.contextmanager
def suppress_process_output():
    """
    Suppress Python- and C-level stdout/stderr during Blender operator calls.
    """
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


def export_object_as_local_stl(source_object, target_dir, depsgraph):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / f"{source_object.name}.stl"

    temp_object = None
    temp_mesh = None
    active_object = getattr(bpy.context.view_layer.objects, "active", None)
    selected_objects = list(getattr(bpy.context, "selected_objects", ()))
    scene_collection = getattr(getattr(bpy.context, "scene", None), "collection", None)

    try:
        temp_object, temp_mesh = mesh_to_temp_object(
            source_object,
            depsgraph,
        )
        scene_collection.objects.link(temp_object)

        bpy.ops.object.select_all(action="DESELECT")
        temp_object.select_set(True)
        bpy.context.view_layer.objects.active = temp_object

        with suppress_process_output():
            bpy.ops.wm.stl_export(
                filepath=str(file_path),
                export_selected_objects=True,
                apply_modifiers=False,
                ascii_format=False,
            )

    finally:
        bpy.ops.object.select_all(action="DESELECT")

        if temp_object is not None:
            try:
                if temp_object.name in bpy.data.objects:
                    bpy.data.objects.remove(temp_object, do_unlink=True)
            except Exception:
                pass

        if temp_mesh is not None:
            try:
                if temp_mesh.name in bpy.data.meshes:
                    bpy.data.meshes.remove(temp_mesh, do_unlink=True)
            except Exception:
                pass

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


def mesh_to_temp_object(source_object, depsgraph):
    """
    Build one temporary mesh object from the evaluated object state.
    """
    object_eval = source_object.evaluated_get(depsgraph)
    temp_mesh = bpy.data.meshes.new_from_object(
        object_eval,
        depsgraph=depsgraph,
        preserve_all_data_layers=False,
    )
    temp_object = bpy.data.objects.new(
        f"__continuum_flow_export__{source_object.name}",
        temp_mesh,
    )
    temp_object.matrix_world.identity()
    return temp_object, temp_mesh


