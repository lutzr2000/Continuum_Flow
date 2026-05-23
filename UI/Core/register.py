import bpy
import importlib.util
import sys
from pathlib import Path
from nodeitems_utils import register_node_categories, unregister_node_categories

MODULE_NAME_ALIASES = {}


def _register_module_aliases(module, module_name):
    """
    Expose one loaded module under its canonical import name.
    """
    sys.modules[module_name] = module
    for alias in MODULE_NAME_ALIASES.get(module_name, ()):
        sys.modules[alias] = module


def _candidate_core_paths(file_names):
    """
    Yield candidate paths for modules that live inside UI/Core.
    """
    for file_name in file_names:
        if "__file__" in globals():
            yield Path(__file__).resolve().with_name(file_name)

        current_text = getattr(getattr(bpy.context, "space_data", None), "text", None)
        if current_text is not None and current_text.filepath:
            yield Path(bpy.path.abspath(current_text.filepath)).resolve().with_name(
                file_name
            )

        yield (Path.cwd() / "UI" / "Core" / file_name).resolve()


def _candidate_ui_paths(relative_path):
    """
    Yield candidate paths for modules that live under the UI root.
    """
    relative_path = Path(relative_path)
    if "__file__" in globals():
        yield Path(__file__).resolve().parent.parent / relative_path

    current_text = getattr(getattr(bpy.context, "space_data", None), "text", None)
    if current_text is not None and current_text.filepath:
        yield Path(
            bpy.path.abspath(current_text.filepath)
        ).resolve().parent.parent / relative_path

    yield (Path.cwd() / "UI" / relative_path).resolve()


def _load_module_from_candidates(candidate_paths, module_name, readable_name):
    """
    Load a module from the first existing candidate path.
    """
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
        _register_module_aliases(module, module_name)
        spec.loader.exec_module(module)
        return module

    raise ImportError(f"None of these modules could be loaded: {readable_name}")


def _load_module(file_names, module_name):
    """
    Load a UI/Core module from disk or from an open Blender text block.
    """
    if isinstance(file_names, str):
        file_names = (file_names,)

    try:
        return _load_module_from_candidates(
            _candidate_core_paths(file_names),
            module_name,
            ", ".join(file_names),
        )
    except ImportError:
        pass

    for file_name in file_names:
        text_block = bpy.data.texts.get(file_name)
        if text_block is not None:
            return text_block.as_module()

    readable_names = ", ".join(file_names)
    raise ImportError(f"None of these modules could be loaded: {readable_names}")


def _load_module_from_relative_path(relative_path, module_name):
    """
    Load a UI module from a path relative to the UI directory.
    """
    return _load_module_from_candidates(
        _candidate_ui_paths(relative_path),
        module_name,
        Path(relative_path),
    )


MODULES = {
    "node_tree": _load_module("node_tree.py", "continuum_flow_nodetree"),
    "sockets": _load_module("sockets.py", "continuum_flow_sockets"),
    "viewer": _load_module("viewer.py", "continuum_flow_viewer"),
    "config": _load_module_from_relative_path(
        Path("Export") / "config.py", "continuum_flow_create_config_dict"
    ),
    "general_nodes": _load_module_from_relative_path(
        Path("Nodes") / "general.py", "continuum_flow_general_nodes"
    ),
    "force_nodes": _load_module_from_relative_path(
        Path("Nodes") / "forces.py", "continuum_flow_force_nodes"
    ),
    "output_node": _load_module_from_relative_path(
        Path("Nodes") / "output.py", "continuum_flow_output_node"
    ),
    "bake_runtime": _load_module_from_relative_path(
        Path("Nodes") / "bake_runtime.py", "continuum_flow_bake_runtime"
    ),
}
NODE_CLASSES = (
    *MODULES["general_nodes"].classes,
    *MODULES["force_nodes"].classes,
    *MODULES["output_node"].classes,
    *MODULES["bake_runtime"].classes,
)


def _remove_continuum_flow_frame_change_handlers():
    """
    Remove stale Continuum Flow frame-change handlers from previous reloads.
    """
    handlers = bpy.app.handlers.frame_change_post
    kept_handlers = []
    for handler in handlers:
        handler_name = getattr(handler, "__name__", "")
        handler_module = getattr(handler, "__module__", "")
        if handler_name in {
            "continuum_flow_frame_change_post",
        }:
            continue
        if "continuum_flow_general_nodes" in handler_module:
            continue
        kept_handlers.append(handler)

    if len(kept_handlers) != len(handlers):
        handlers[:] = kept_handlers


def _reregister_classes(classes):
    """
    Unregister and register one class sequence to survive Blender reloads.
    """
    for cls in classes:
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
        bpy.utils.register_class(cls)


def _unregister_classes(classes):
    """
    Unregister one class sequence in reverse order.
    """
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass


def register():
    """
    Register node tree classes, node classes, and add-menu categories.
    """
    _remove_continuum_flow_frame_change_handlers()
    _reregister_classes(MODULES["node_tree"].classes)
    _reregister_classes(MODULES["sockets"].classes)
    _reregister_classes(NODE_CLASSES)
    _reregister_classes(MODULES["viewer"].classes)
    try:
        MODULES["viewer"].enable_force_preview()
    except Exception:
        pass
    _reregister_classes(MODULES["config"].classes)
    try:
        unregister_node_categories(MODULES["node_tree"].NODE_CATEGORIES_ID)
    except Exception:
        pass
    register_node_categories(
        MODULES["node_tree"].NODE_CATEGORIES_ID,
        MODULES["node_tree"].build_node_categories(),
    )
    frame_change_handler = getattr(
        MODULES["general_nodes"],
        "continuum_flow_frame_change_post",
        None,
    )
    if frame_change_handler is not None:
        bpy.app.handlers.frame_change_post.append(frame_change_handler)
    sync_animations = getattr(
        MODULES["general_nodes"],
        "sync_all_continuum_flow_node_animations",
        None,
    )
    if callable(sync_animations):
        sync_animations(getattr(bpy.context, "scene", None))


def unregister():
    """
    Unregister menu categories and all Continuum Flow UI classes.
    """
    _remove_continuum_flow_frame_change_handlers()

    try:
        unregister_node_categories(MODULES["node_tree"].NODE_CATEGORIES_ID)
    except Exception:
        pass

    try:
        MODULES["viewer"].disable_domain_preview()
    except Exception:
        pass
    try:
        MODULES["viewer"].disable_force_preview()
    except Exception:
        pass
    _unregister_classes(MODULES["viewer"].classes)
    _unregister_classes(MODULES["config"].classes)
    _unregister_classes(NODE_CLASSES)
    _unregister_classes(MODULES["sockets"].classes)
    _unregister_classes(MODULES["node_tree"].classes)


if __name__ == "__main__":
    register()
