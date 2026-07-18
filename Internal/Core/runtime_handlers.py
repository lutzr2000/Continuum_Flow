import bpy
from bpy.app.handlers import persistent

from . import baking as bake_main, forces, viewer
from .export.export_config import sync_node_tree_animations
from ..UI.node_tree import NODE_TREE_ID

_FAKE_USER_INITIALIZED_KEY = "_continuum_flow_fake_user_initialized"


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
def initialize_fake_user_state(_scene=None, _depsgraph=None):
    node_groups = getattr(bpy.data, "node_groups", None)
    if node_groups is None:
        return

    for tree in node_groups:
        if tree.bl_idname == NODE_TREE_ID:
            tree[_FAKE_USER_INITIALIZED_KEY] = True


@persistent
def ensure_fake_user(_scene=None, _depsgraph=None):
    node_groups = getattr(bpy.data, "node_groups", None)
    if node_groups is None:
        return

    for tree in node_groups:
        if tree.bl_idname != NODE_TREE_ID:
            continue

        if tree.get(_FAKE_USER_INITIALIZED_KEY):
            continue

        if not tree.use_fake_user:
            tree.use_fake_user = True

        tree[_FAKE_USER_INITIALIZED_KEY] = True


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
    sync_node_tree_animations(scene)
    _tag_animation_editors_redraw()


def sync_ui_animation_state(scene=None):
    sync_node_tree_animations(scene)
    _tag_animation_editors_redraw()
