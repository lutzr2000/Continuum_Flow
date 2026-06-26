import bpy
import importlib.util
from pathlib import Path
from bpy.app.handlers import persistent


# Load node_tree.py
HERE = Path(bpy.data.texts["register.py"].filepath).parent
NODE_TREE_PATH = HERE / "node_tree.py"

spec = importlib.util.spec_from_file_location("node_tree", NODE_TREE_PATH)
node_tree = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node_tree)


@persistent
def ensure_fake_user(_scene=None, _depsgraph=None):
    for tree in bpy.data.node_groups:
        if (
            tree.bl_idname == node_tree.NODE_TREE_ID
            and not tree.use_fake_user
        ):
            tree.use_fake_user = True


def register():
    bpy.utils.register_class(node_tree.ContinuumFlowNodeTree)

    ensure_fake_user()

    if ensure_fake_user not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(ensure_fake_user)


def unregister():
    if ensure_fake_user in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(ensure_fake_user)

    bpy.utils.unregister_class(node_tree.ContinuumFlowNodeTree)


if __name__ == "__main__":
    try:
        unregister()
    except Exception:
        pass

    register()