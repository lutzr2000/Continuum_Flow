from pathlib import Path

import bpy


def export_object_as_usdc(source_object, target_dir, start_frame, end_frame):
    """Export one animated object as USD while retaining polygon topology."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / f"{source_object.name}.usdc"
    scene = bpy.context.scene
    current_frame = int(scene.frame_current)
    current_start = int(scene.frame_start)
    current_end = int(scene.frame_end)
    active_object = bpy.context.view_layer.objects.active
    selected_objects = list(bpy.context.selected_objects)

    try:
        scene.frame_start = int(start_frame)
        scene.frame_end = max(int(start_frame), int(end_frame) - 1)
        bpy.ops.object.select_all(action="DESELECT")
        source_object.select_set(True)
        bpy.context.view_layer.objects.active = source_object
        bpy.ops.wm.usd_export(
            filepath=str(file_path),
            selected_objects_only=True,
            export_animation=True,
            export_materials=False,
            export_uvmaps=False,
            export_mesh_colors=False,
            export_normals=False,
            export_hair=False,
            export_armatures=False,
            export_shapekeys=False,
            export_custom_properties=False,
            export_lights=False,
            export_cameras=False,
            export_curves=False,
            export_points=False,
            export_volumes=False,
            triangulate_meshes=True,
            evaluation_mode="RENDER",
        )
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for selected_object in selected_objects:
            if selected_object.name in bpy.data.objects:
                selected_object.select_set(True)
        bpy.context.view_layer.objects.active = active_object
        scene.frame_start = current_start
        scene.frame_end = current_end
        scene.frame_set(current_frame)
