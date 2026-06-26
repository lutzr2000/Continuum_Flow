import bpy
from bpy.app.handlers import persistent
from nodeitems_utils import register_node_categories, unregister_node_categories

from ..Interface.node_tree import (
    ContinuumFlowNodeTree,
    CONTINUUM_FLOW_OT_reload,
    NODE_TREE_ID,
    NODE_CATEGORIES_ID,
    build_node_categories,
)

from ..Interface.node_geometry import ContinuumFlowGeometryNode


@persistent
def ensure_fake_user(_scene=None, _depsgraph=None):
    node_groups = getattr(bpy.data, "node_groups", None)
    if node_groups is None:
        return

    for tree in node_groups:
        if tree.bl_idname == NODE_TREE_ID and not tree.use_fake_user:
            tree.use_fake_user = True


classes = (
    ContinuumFlowNodeTree,
    CONTINUUM_FLOW_OT_reload,
    ContinuumFlowGeometryNode,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    register_node_categories(
        NODE_CATEGORIES_ID,
        build_node_categories(),
    )

    bpy.app.timers.register(ensure_fake_user, first_interval=0.1)

    if ensure_fake_user not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(ensure_fake_user)


def unregister():
    if ensure_fake_user in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(ensure_fake_user)

    try:
        unregister_node_categories(NODE_CATEGORIES_ID)
    except Exception:
        pass

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)