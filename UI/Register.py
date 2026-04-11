import bpy
import importlib.util
import sys
from pathlib import Path
from nodeitems_utils import register_node_categories, unregister_node_categories


def _load_module(file_name, module_name):
    """Load a UI module from disk or from an open Blender text block."""
    candidate_paths = []

    if "__file__" in globals():
        candidate_paths.append(Path(__file__).resolve().with_name(file_name))

    current_text = getattr(getattr(bpy.context, "space_data", None), "text", None)
    if current_text is not None and current_text.filepath:
        candidate_paths.append(Path(bpy.path.abspath(current_text.filepath)).resolve().with_name(file_name))

    candidate_paths.append((Path.cwd() / "UI" / file_name).resolve())

    seen = set()
    for candidate in candidate_paths:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)

        if candidate.exists():
            spec = importlib.util.spec_from_file_location(module_name, candidate)
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

    text_block = bpy.data.texts.get(file_name)
    if text_block is not None:
        return text_block.as_module()

    raise ImportError(f"{file_name} konnte nicht geladen werden.")


node_tree_module = _load_module("NodeTree.py", "blendercfd_nodetree")
socket_module = _load_module("Sockets.py", "blendercfd_sockets")
nodes_module = _load_module("Nodes.py", "blendercfd_nodes")
viewer_module = _load_module("Viewer.py", "blendercfd_viewer")
config_module = _load_module("Create_Config_Dict.py", "blendercfd_create_config_dict")


def register():
    """Register node tree classes, node classes, and add-menu categories."""
    for cls in node_tree_module.classes:
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
        bpy.utils.register_class(cls)

    for cls in socket_module.classes:
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
        bpy.utils.register_class(cls)

    for cls in nodes_module.classes:
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
        bpy.utils.register_class(cls)

    for cls in viewer_module.classes:
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
        bpy.utils.register_class(cls)

    for cls in config_module.classes:
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
        bpy.utils.register_class(cls)
    try:
        unregister_node_categories(node_tree_module.NODE_CATEGORIES_ID)
    except Exception:
        pass
    register_node_categories(node_tree_module.NODE_CATEGORIES_ID, node_tree_module.build_node_categories())


def unregister():
    """Unregister menu categories and all BlenderCFD UI classes."""
    try:
        unregister_node_categories(node_tree_module.NODE_CATEGORIES_ID)
    except Exception:
        pass

    try:
        viewer_module.disable_domain_preview()
    except Exception:
        pass

    for cls in reversed(viewer_module.classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass

    for cls in reversed(config_module.classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass

    for cls in reversed(nodes_module.classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass

    for cls in reversed(socket_module.classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass

    for cls in reversed(node_tree_module.classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass


if __name__ == "__main__":
    register()
