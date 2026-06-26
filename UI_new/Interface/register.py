import bpy
from bpy.app.handlers import persistent
from nodeitems_utils import register_node_categories, unregister_node_categories
from pathlib import Path
import subprocess

from ..Interface.node_tree import (
    ContinuumFlowNodeTree,
    CONTINUUM_FLOW_OT_reload,
    NODE_TREE_ID,
    NODE_CATEGORIES_ID,
    build_node_categories,
)

from ..Interface.sockets import (
    ContinuumFlowIntSocket,
    ContinuumFlowForceSocket,
    ContinuumFlowLinkSocket,
    ContinuumFlowResultSocket,
)

from ..Interface.node_domain import ContinuumFlowDomainNode
from ..Interface.node_forces import ContinuumFlowForceConstantNode,ContinuumFlowForceSwirlNode,ContinuumFlowForceTurbulenceNode
from ..Interface.node_geometry import ContinuumFlowGeometryNode
from ..Interface.node_obstacle import ContinuumFlowObstacleNode
from ..Interface.node_output import ContinuumFlowOutputNode
from ..Interface.node_physics import ContinuumFlowPhysicsNode0
from ..Interface.node_simulation import ContinuumFlowSimulationNode
from ..Interface.node_source import ContinuumFlowSourceNode
from ..Interface.node_viewer import ContinuumFlowViewerNode
from ..Interface.node_preset_tree import ContinuumFlow_OT_add_basic_setup
from ..Core.main import main
from ..Core import solver_status


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

    ContinuumFlowIntSocket,
    ContinuumFlowForceSocket,
    ContinuumFlowLinkSocket,
    ContinuumFlowResultSocket,

    ContinuumFlowDomainNode,
    ContinuumFlowGeometryNode,
    ContinuumFlowOutputNode,
    ContinuumFlowPhysicsNode0,
    ContinuumFlowSimulationNode,
    ContinuumFlowSourceNode,
    ContinuumFlowObstacleNode,
    ContinuumFlowViewerNode,

    ContinuumFlowForceConstantNode,
    ContinuumFlowForceSwirlNode,
    ContinuumFlowForceTurbulenceNode,

    ContinuumFlow_OT_add_basic_setup,
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


def unregister():
    solver_status.environment_ready = False
    solver_status.gpu_available = False
    if ensure_fake_user in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(ensure_fake_user)

    try:
        unregister_node_categories(NODE_CATEGORIES_ID)
    except Exception:
        pass

    for cls in reversed(classes):
        safe_unregister_class(cls)