from pathlib import Path
import contextlib
import io
import bpy


def export_object_as_local_stl(source_object, target_dir):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / f"{source_object.name}.stl"

    old_matrix = source_object.matrix_world.copy()

    try:
        source_object.matrix_world.identity()

        bpy.ops.object.select_all(action="DESELECT")
        source_object.select_set(True)
        bpy.context.view_layer.objects.active = source_object

        # remove console output
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            bpy.ops.wm.stl_export(
                filepath=str(file_path),
                export_selected_objects=True,
                apply_modifiers=True,
                ascii_format=False,
            )

    finally:
        source_object.matrix_world = old_matrix

    return file_path