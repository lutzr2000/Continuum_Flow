import bpy
from bpy.app.handlers import persistent
from nodeitems_utils import register_node_categories, unregister_node_categories
from pathlib import Path
import subprocess

from ..UI.node_tree import (
    ContinuumFlowNodeTree,
    CONTINUUM_FLOW_OT_reload,
    NODE_TREE_ID,
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
from ..UI.node_output import ContinuumFlowOutputNode, CONTINUUM_FLOW_OT_output_bake_button
from ..UI.node_physics import ContinuumFlowPhysicsNode0
from ..UI.node_simulation import ContinuumFlowSimulationNode
from ..UI.node_source import ContinuumFlowSourceNode
from ..UI.node_viewer import ContinuumFlowViewerNode
from ..UI.node_preset_tree import ContinuumFlow_OT_add_basic_setup
from ..Core.main import main, CONTINUUM_FLOW_OT_cancel_bake
from ..Core import viewer
from ..Core import main as bake_main
from ..Core.viewer import ContinuumFlow_OT_viewer_toggle_domain
from ..Core import solver_status


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
        bake_main.refresh_bake_state_from_output_nodes()
    except Exception:
        pass


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
    main,
)


def check_solver_status():
    addon_root = Path(__file__).resolve().parents[2]
    py = addon_root / "ContinuumFlow_env" / "Scripts" / "python.exe"

    solver_status.environment_ready = py.exists()
    solver_status.gpu_available = False

    if solver_status.environment_ready:
        result = subprocess.run(
            [
                str(py),
                "-c",
                "from numba import cuda; cuda.get_current_device()",
            ],
            capture_output=True,
        )

        solver_status.gpu_available = result.returncode == 0


def safe_unregister_class(cls):
    try:
        bpy.utils.unregister_class(cls)
    except RuntimeError:
        pass


def safe_register_class(cls):
    safe_unregister_class(cls)
    bpy.utils.register_class(cls)


def register():
    check_solver_status()

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

    if not bpy.app.timers.is_registered(ensure_fake_user):
        bpy.app.timers.register(ensure_fake_user, first_interval=0.1)

    if ensure_fake_user not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(ensure_fake_user)

    if sync_runtime_state not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(sync_runtime_state)

    sync_runtime_state()


def unregister():
    solver_status.environment_ready = False
    solver_status.gpu_available = False


    if hasattr(bpy.types.WindowManager, "continuum_flow_bake_progress"):
        del bpy.types.WindowManager.continuum_flow_bake_progress

    if ensure_fake_user in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(ensure_fake_user)

    if sync_runtime_state in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(sync_runtime_state)

    try:
        unregister_node_categories(NODE_CATEGORIES_ID)
    except Exception:
        pass

    try:
        viewer.disable_domain_preview()
    except Exception:
        pass

    for cls in reversed(classes):
        safe_unregister_class(cls)

