import bpy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from nodeitems_utils import register_node_categories, unregister_node_categories


def _load_module(file_names, module_name):
    """Load a UI module from disk or from an open Blender text block."""
    if isinstance(file_names, str):
        file_names = (file_names,)

    candidate_paths = []

    for file_name in file_names:
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

    for file_name in file_names:
        text_block = bpy.data.texts.get(file_name)
        if text_block is not None:
            return text_block.as_module()

    readable_names = ", ".join(file_names)
    raise ImportError(f"Keines dieser Module konnte geladen werden: {readable_names}")


def _load_module_from_relative_path(relative_path, module_name):
    """Load a UI module from a path relative to the UI directory."""
    relative_path = Path(relative_path)
    candidate_paths = []

    if "__file__" in globals():
        candidate_paths.append(Path(__file__).resolve().parent / relative_path)

    current_text = getattr(getattr(bpy.context, "space_data", None), "text", None)
    if current_text is not None and current_text.filepath:
        candidate_paths.append(Path(bpy.path.abspath(current_text.filepath)).resolve().parent / relative_path)

    candidate_paths.append((Path.cwd() / "UI" / relative_path).resolve())

    seen = set()
    for candidate in candidate_paths:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)

        if not candidate.exists():
            continue

        spec = importlib.util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    raise ImportError(f"Keines dieser Module konnte geladen werden: {relative_path}")


node_tree_module = _load_module(("Node_Tree.py", "NodeTree.py"), "blendercfd_nodetree")
socket_module = _load_module("Sockets.py", "blendercfd_sockets")
viewer_module = _load_module("Viewer.py", "blendercfd_viewer")
config_module = _load_module(("Config_Export.py", "Create_Config_Dict.py"), "blendercfd_create_config_dict")
general_nodes_module = _load_module_from_relative_path(Path("Nodes") / "General_Nodes.py", "blendercfd_general_nodes")
source_nodes_module = _load_module_from_relative_path(Path("Nodes") / "Source_and_Obstacle_Nodes.py", "blendercfd_source_obstacle_nodes")
force_nodes_module = _load_module_from_relative_path(Path("Nodes") / "Force_Nodes.py", "blendercfd_force_nodes")
output_nodes_module = _load_module_from_relative_path(Path("Nodes") / "Output_Nodes.py", "blendercfd_output_nodes")
nodes_module = SimpleNamespace(
    classes=(
        *general_nodes_module.classes,
        *source_nodes_module.classes,
        *force_nodes_module.classes,
        *output_nodes_module.classes,
    )
)


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
