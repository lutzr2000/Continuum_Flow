"""Core registration entrypoint for BlenderCFD UI classes and node categories."""

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

        candidate_paths.append((Path.cwd() / "UI" / "Core" / file_name).resolve())

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
        candidate_paths.append(Path(__file__).resolve().parent.parent / relative_path)

    current_text = getattr(getattr(bpy.context, "space_data", None), "text", None)
    if current_text is not None and current_text.filepath:
        candidate_paths.append(Path(bpy.path.abspath(current_text.filepath)).resolve().parent.parent / relative_path)

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


node_tree_module = _load_module("node_tree.py", "blendercfd_nodetree")
socket_module = _load_module("sockets.py", "blendercfd_sockets")
viewer_module = _load_module("viewer.py", "blendercfd_viewer")
config_module = _load_module_from_relative_path("Config_Export.py", "blendercfd_create_config_dict")
general_nodes_module = _load_module_from_relative_path(Path("Nodes") / "general.py", "blendercfd_general_nodes")
force_nodes_module = _load_module_from_relative_path(Path("Nodes") / "forces.py", "blendercfd_force_nodes")
output_node_module = _load_module_from_relative_path(Path("Nodes") / "output.py", "blendercfd_output_node")
bake_runtime_module = _load_module_from_relative_path(Path("Nodes") / "bake_runtime.py", "blendercfd_bake_runtime")
nodes_module = SimpleNamespace(
    classes=(
        *general_nodes_module.classes,
        *force_nodes_module.classes,
        *output_node_module.classes,
        *bake_runtime_module.classes,
    )
)


def _remove_blendercfd_frame_change_handlers():
    """Remove stale BlenderCFD frame-change handlers from previous reloads."""
    handlers = bpy.app.handlers.frame_change_post
    kept_handlers = []
    for handler in handlers:
        handler_name = getattr(handler, "__name__", "")
        handler_module = getattr(handler, "__module__", "")
        if handler_name == "blendercfd_frame_change_post":
            continue
        if "blendercfd_general_nodes" in handler_module:
            continue
        kept_handlers.append(handler)

    if len(kept_handlers) != len(handlers):
        handlers[:] = kept_handlers


def register():
    """Register node tree classes, node classes, and add-menu categories."""
    _remove_blendercfd_frame_change_handlers()

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
    try:
        viewer_module.enable_force_preview()
    except Exception:
        pass

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
    frame_change_handler = getattr(general_nodes_module, "blendercfd_frame_change_post", None)
    if frame_change_handler is not None:
        bpy.app.handlers.frame_change_post.append(frame_change_handler)
    sync_animations = getattr(general_nodes_module, "sync_all_blendercfd_node_animations", None)
    if callable(sync_animations):
        sync_animations(getattr(bpy.context, "scene", None))


def unregister():
    """Unregister menu categories and all BlenderCFD UI classes."""
    _remove_blendercfd_frame_change_handlers()

    try:
        unregister_node_categories(node_tree_module.NODE_CATEGORIES_ID)
    except Exception:
        pass

    try:
        viewer_module.disable_domain_preview()
    except Exception:
        pass
    try:
        viewer_module.disable_force_preview()
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
