import bpy
from nodeitems_utils import register_node_categories, unregister_node_categories
import subprocess

from ..UI.node_tree import (
    ContinuumFlowNodeTree,
    CONTINUUM_FLOW_OT_reload,
    NODE_CATEGORIES_ID,
    build_node_categories,
)

from ..UI.sockets import (
    ContinuumFlowIntSocket,
    ContinuumFlowForceSocket,
    ContinuumFlowLinkSocket,
    ContinuumFlowResultSocket,
)

from ..UI.node_domain import ContinuumFlowDomainNode
from ..UI.node_forces import ContinuumFlowForceConstantNode,ContinuumFlowForceSwirlNode,ContinuumFlowForceTurbulenceNode
from ..UI.node_geometry import ContinuumFlowGeometryNode
from ..UI.node_obstacle import ContinuumFlowObstacleNode
from ..UI.node_output import (
    ContinuumFlowOutputNode,
    CONTINUUM_FLOW_OT_output_bake_button,
    CONTINUUM_FLOW_OT_output_free_bake_button,
)
from ..UI.node_physics import ContinuumFlowPhysicsNode0
from ..UI.node_simulation import ContinuumFlowSimulationNode
from ..UI.node_source import ContinuumFlowSourceNode
from ..UI.node_viewer import ContinuumFlowViewerNode
from ..UI.node_preset_tree import ContinuumFlow_OT_add_basic_setup
from ..Core.runtime_handlers import (
    continuum_flow_frame_change_post,
    ensure_fake_user,
    initialize_fake_user_state,
    sync_runtime_state,
    sync_ui_animation_state,
)
from ..Core.main import main, CONTINUUM_FLOW_OT_cancel_bake, CONTINUUM_FLOW_OT_free_bake
from ..Core import forces
from ..Core.viewer import ContinuumFlow_OT_viewer_toggle_domain
from ..Core import solver_status


classes = (
    ContinuumFlowNodeTree,
    CONTINUUM_FLOW_OT_reload,

    ContinuumFlowIntSocket,
    ContinuumFlowForceSocket,
    ContinuumFlowLinkSocket,
    ContinuumFlowResultSocket,

    ContinuumFlowDomainNode,
    ContinuumFlowGeometryNode,
    ContinuumFlowOutputNode,
    CONTINUUM_FLOW_OT_output_bake_button,
    CONTINUUM_FLOW_OT_output_free_bake_button,
    ContinuumFlowPhysicsNode0,
    ContinuumFlowSimulationNode,
    ContinuumFlowSourceNode,
    ContinuumFlowObstacleNode,
    ContinuumFlowViewerNode,

    ContinuumFlowForceConstantNode,
    ContinuumFlowForceSwirlNode,
    ContinuumFlowForceTurbulenceNode,

    ContinuumFlow_OT_add_basic_setup,
    ContinuumFlow_OT_viewer_toggle_domain,
    CONTINUUM_FLOW_OT_cancel_bake,
    CONTINUUM_FLOW_OT_free_bake,
    main,
)


def safe_unregister_class(cls):
    try:
        bpy.utils.unregister_class(cls)
    except RuntimeError:
        pass


def safe_register_class(cls):
    safe_unregister_class(cls)
    bpy.utils.register_class(cls)


def register():

    for cls in classes:
        safe_register_class(cls)

    try:
        unregister_node_categories(NODE_CATEGORIES_ID)
    except Exception:
        pass

    register_node_categories(
        NODE_CATEGORIES_ID,
        build_node_categories(),
    )

    initialize_fake_user_state()

    if not bpy.app.timers.is_registered(ensure_fake_user):
        bpy.app.timers.register(ensure_fake_user, first_interval=0.1)

    if not bpy.app.timers.is_registered(forces.force_preview_timer):
        bpy.app.timers.register(forces.force_preview_timer, first_interval=0.1)

    if ensure_fake_user not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(ensure_fake_user)

    if sync_runtime_state not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(sync_runtime_state)

    if initialize_fake_user_state not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(initialize_fake_user_state)

    if continuum_flow_frame_change_post not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(continuum_flow_frame_change_post)

    sync_runtime_state()
    sync_ui_animation_state(getattr(bpy.context, "scene", None))


def unregister():
    solver_status.gpu_available = False

    if hasattr(bpy.types.WindowManager, "continuum_flow_bake_progress"):
        del bpy.types.WindowManager.continuum_flow_bake_progress

    if ensure_fake_user in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(ensure_fake_user)

    if continuum_flow_frame_change_post in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(continuum_flow_frame_change_post)

    if bpy.app.timers.is_registered(forces.force_preview_timer):
        bpy.app.timers.unregister(forces.force_preview_timer)

    if sync_runtime_state in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(sync_runtime_state)

    if initialize_fake_user_state in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(initialize_fake_user_state)

    try:
        unregister_node_categories(NODE_CATEGORIES_ID)
    except Exception:
        pass

    try:
        sync_runtime_state()
    except Exception:
        pass

    for cls in reversed(classes):
        safe_unregister_class(cls)
