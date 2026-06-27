import bpy
from bpy.app.handlers import persistent

from . import forces, main as bake_main, viewer
from .export_config import sync_all_continuum_flow_node_animations
from ..UI.node_tree import NODE_TREE_ID


def _tag_animation_editors_redraw():
    window_manager = getattr(bpy.context, "window_manager", None)
    if window_manager is None:
        return

    for window in window_manager.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type in {"NODE_EDITOR", "PROPERTIES"}:
                area.tag_redraw()


@persistent
def ensure_fake_user(_scene=None, _depsgraph=None):
    node_groups = getattr(bpy.data, "node_groups", None)
    if node_groups is None:
        return

    for tree in node_groups:
        if tree.bl_idname == NODE_TREE_ID and not tree.use_fake_user:
            tree.use_fake_user = True


@persistent
def sync_runtime_state(_scene=None):
    try:
        viewer.disable_domain_preview()
    except Exception:
        pass

    try:
        forces.disable_force_preview()
    except Exception:
        pass

    try:
        bake_main.refresh_bake_state_from_output_nodes()
    except Exception:
        pass


@persistent
def continuum_flow_frame_change_post(scene, _depsgraph=None):
    sync_all_continuum_flow_node_animations(scene)
    _tag_animation_editors_redraw()


def sync_ui_animation_state(scene=None):
    sync_all_continuum_flow_node_animations(scene)
    _tag_animation_editors_redraw()
