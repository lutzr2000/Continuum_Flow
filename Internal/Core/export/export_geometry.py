from pathlib import Path

import bpy


def make_object_exportable(source_object, scene, view_layer):
    """
    Temporarily make an object selectable in the active view layer.
    """
    linked_to_scene = source_object.name not in view_layer.objects
    if linked_to_scene:
        scene.collection.objects.link(source_object)

    state = {
        "linked_to_scene": linked_to_scene,
        "hide_viewport": source_object.hide_viewport,
        "hide_select": source_object.hide_select,
        "hidden": source_object.hide_get(view_layer=view_layer),
    }
    source_object.hide_viewport = False
    source_object.hide_select = False
    source_object.hide_set(False, view_layer=view_layer)
    return state


def restore_object_visibility(source_object, scene, view_layer, state):
    source_object.hide_set(state["hidden"], view_layer=view_layer)
    source_object.hide_select = state["hide_select"]
    source_object.hide_viewport = state["hide_viewport"]
    if state["linked_to_scene"]:
        scene.collection.objects.unlink(source_object)


def export_object_as_usdc(
    source_object, target_dir, start_frame, end_frame, deforming=False
):
    """Export an object's polygon topology and, when needed, deformed points."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / f"{source_object.name}.usdc"
    scene = bpy.context.scene
    current_frame = int(scene.frame_current)
    current_start = int(scene.frame_start)
    current_end = int(scene.frame_end)
    view_layer = bpy.context.view_layer
    active_object = view_layer.objects.active
    selected_objects = list(bpy.context.selected_objects)
    visibility_state = None

    try:
        scene.frame_start = int(start_frame)
        scene.frame_end = (
            max(int(start_frame), int(end_frame) - 1) if deforming else int(start_frame)
        )
        scene.frame_set(int(start_frame))
        bpy.ops.object.select_all(action="DESELECT")
        visibility_state = make_object_exportable(source_object, scene, view_layer)
        source_object.select_set(True)
        view_layer.objects.active = source_object
        bpy.ops.wm.usd_export(
            filepath=str(file_path),
            selected_objects_only=True,
            export_animation=bool(deforming),
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
            triangulate_meshes=False,
            evaluation_mode="RENDER",
        )
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        if visibility_state is not None:
            restore_object_visibility(
                source_object, scene, view_layer, visibility_state
            )
        for selected_object in selected_objects:
            if (
                selected_object.name in bpy.data.objects
                and selected_object.name in view_layer.objects
            ):
                selected_object.select_set(True)
        if active_object is None or active_object.name in view_layer.objects:
            view_layer.objects.active = active_object
        scene.frame_start = current_start
        scene.frame_end = current_end
        scene.frame_set(current_frame)
    return file_path
