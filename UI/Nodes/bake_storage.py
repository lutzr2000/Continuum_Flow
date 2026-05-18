"""Storage and import helpers for Continuum Flow bake output directories and VDB files."""

import copy
import shutil
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Matrix


_LAST_BAKE_CONFIG_DICT = None
_BAKE_OUTPUT_DIRECTORY_PREFIX = ".continuum_flow_bake_"
_LEGACY_BAKE_OUTPUT_DIRECTORY_PREFIX = ".blendercfd_bake_"
_BAKE_CANCEL_FILENAME = ".continuum_flow_cancel"
_LEGACY_BAKE_CANCEL_FILENAME = ".blendercfd_cancel"
_AUTO_IMPORT_KEY = "continuum_flow_auto_import"
_LEGACY_AUTO_IMPORT_KEY = "blendercfd_auto_import"
_OUTPUT_PATH_KEY = "continuum_flow_output_path"
_LEGACY_OUTPUT_PATH_KEY = "blendercfd_output_path"


def _iter_simulation_dicts(config_dict):
    """Yield only valid simulation dictionaries from one config payload."""
    if not isinstance(config_dict, dict):
        return
    for simulation_cfg in config_dict.get("simulations", ()) or ():
        if isinstance(simulation_cfg, dict):
            yield simulation_cfg


def _iter_output_dicts(simulation_cfg):
    """Yield only valid output dictionaries from one simulation entry."""
    if not isinstance(simulation_cfg, dict):
        return
    for output_cfg in simulation_cfg.get("outputs", ()) or ():
        if isinstance(output_cfg, dict):
            yield output_cfg


def _iter_viewer_dicts(simulation_cfg):
    """Yield only valid viewer dictionaries from one simulation entry."""
    if not isinstance(simulation_cfg, dict):
        return
    for viewer_cfg in simulation_cfg.get("viewers", ()) or ():
        if isinstance(viewer_cfg, dict):
            yield viewer_cfg


def _tag_all_areas_redraw(context=None):
    window_manager = getattr(context, "window_manager", None) if context is not None else None
    if window_manager is None:
        window_manager = getattr(bpy.context, "window_manager", None)
    if window_manager is None:
        return

    for window in window_manager.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            area.tag_redraw()


def _set_bake_state_active(active, context=None, config_dict=None):
    global _LAST_BAKE_CONFIG_DICT
    if active and config_dict is not None:
        _LAST_BAKE_CONFIG_DICT = config_dict
    elif not active:
        _LAST_BAKE_CONFIG_DICT = None
    _tag_all_areas_redraw(context)


def _iter_output_directory_vdb_files(output_directory):
    output_directory = Path(output_directory).resolve()
    if not output_directory.exists():
        return

    for vdb_file in sorted(output_directory.glob("*.vdb")):
        if vdb_file.is_file():
            yield vdb_file

    for child_directory in sorted(output_directory.iterdir()):
        if not child_directory.is_dir():
            continue
        if not (
            child_directory.name.startswith(_BAKE_OUTPUT_DIRECTORY_PREFIX) or
            child_directory.name.startswith(_LEGACY_BAKE_OUTPUT_DIRECTORY_PREFIX)
        ):
            continue
        for vdb_file in sorted(child_directory.glob("*.vdb")):
            if vdb_file.is_file():
                yield vdb_file


def _output_directory_has_vdbs(output_directory):
    return any(_iter_output_directory_vdb_files(output_directory))


def _path_is_same_or_within_directory(path_value, directory):
    try:
        resolved_path = Path(path_value).resolve()
        resolved_directory = Path(directory).resolve()
    except Exception:
        return False
    if resolved_path == resolved_directory:
        return True
    try:
        return resolved_directory in resolved_path.parents
    except Exception:
        return False


def _display_output_directory_name(output_directory):
    output_directory = Path(output_directory)
    if output_directory.name.startswith(_BAKE_OUTPUT_DIRECTORY_PREFIX) or output_directory.name.startswith(_LEGACY_BAKE_OUTPUT_DIRECTORY_PREFIX):
        return output_directory.parent.name or output_directory.name
    return output_directory.name


def _prepared_output_directory(base_output_path, bake_token):
    return Path(base_output_path).resolve() / f"{_BAKE_OUTPUT_DIRECTORY_PREFIX}{bake_token}"


def _bake_geometry_cache_directory(output_directory, bake_token):
    output_directory = Path(output_directory).resolve()
    return _prepared_output_directory(output_directory, bake_token) / "geometry"


def _bake_cancel_flag_path(output_directory, bake_token):
    output_directory = Path(output_directory).resolve()
    return _prepared_output_directory(output_directory, bake_token) / _BAKE_CANCEL_FILENAME


def _cleanup_geometry_cache(config_dict):
    if not config_dict:
        return
    geometry_cache_dir = (((config_dict.get("meta") or {}).get("geometry_cache_dir")) or "").strip()
    if not geometry_cache_dir:
        return

    geometry_cache_path = Path(geometry_cache_dir).resolve()
    managed_output_directories = _output_directories_from_config(config_dict)
    if not any(
        _path_is_same_or_within_directory(geometry_cache_path, output_directory)
        for output_directory in managed_output_directories
    ):
        return

    shutil.rmtree(geometry_cache_path, ignore_errors=True)

    current_path = geometry_cache_path.parent
    while True:
        if not any(
            _path_is_same_or_within_directory(current_path, output_directory)
            for output_directory in managed_output_directories
        ):
            break

        try:
            current_path.rmdir()
        except OSError:
            break

        if any(current_path == output_directory for output_directory in managed_output_directories):
            break
        current_path = current_path.parent


def _clear_bake_cancel_flag(config_dict):
    if not config_dict:
        return
    meta_cfg = config_dict.get("meta") or {}
    cancel_flag_path = (meta_cfg.get("cancel_flag_path") or "").strip()
    if not cancel_flag_path:
        return
    try:
        Path(cancel_flag_path).unlink(missing_ok=True)
    except OSError:
        pass


def _prepare_config_for_bake(config_dict, bake_token=None):
    prepared_config = copy.deepcopy(config_dict)
    if not bake_token:
        bake_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    meta_cfg = prepared_config.setdefault("meta", {})
    meta_cfg["bake_token"] = bake_token
    cancel_flag_path = None
    for simulation_cfg in _iter_simulation_dicts(prepared_config):
        for output_cfg in _iter_output_dicts(simulation_cfg):
            output_path = output_cfg.get("output_path", "")
            if not output_path:
                continue
            base_output_path = str(Path(output_path).resolve())
            output_cfg["base_output_path"] = base_output_path
            output_cfg["output_path"] = str(_prepared_output_directory(base_output_path, bake_token))
            if cancel_flag_path is None:
                cancel_flag_path = _bake_cancel_flag_path(base_output_path, bake_token)
    if cancel_flag_path is not None:
        meta_cfg["cancel_flag_path"] = str(cancel_flag_path)
    return prepared_config


def _output_directories_from_config(config_dict):
    output_directories = []
    seen_paths = set()
    for simulation_cfg in _iter_simulation_dicts(config_dict):
        for output_cfg in _iter_output_dicts(simulation_cfg):
            output_path = output_cfg.get("output_path", "")
            if not output_path:
                continue
            resolved_path = str(Path(output_path).resolve())
            if resolved_path in seen_paths:
                continue
            seen_paths.add(resolved_path)
            output_directories.append(Path(resolved_path))
    return output_directories


def _start_frame_for_output_directory(config_dict, output_directory):
    output_directory = Path(output_directory).resolve()
    for simulation_cfg in _iter_simulation_dicts(config_dict):
        settings = simulation_cfg.get("settings") or {}
        start_frame = int(settings.get("start_frame", 1))
        for output_cfg in _iter_output_dicts(simulation_cfg):
            output_path = output_cfg.get("output_path", "")
            if not output_path:
                continue
            if Path(output_path).resolve() == output_directory:
                return start_frame
    return 1


def _config_has_baked_vdbs(config_dict):
    return any(
        _output_directory_has_vdbs(output_directory)
        for output_directory in _output_directories_from_config(config_dict)
    )


def _resolve_output_directory_hint(output_path_hint):
    output_path_hint = (output_path_hint or "").strip()
    if not output_path_hint:
        return None
    resolved_output_path = bpy.path.abspath(output_path_hint)
    if not resolved_output_path:
        return None
    return Path(resolved_output_path).resolve()


def _baked_vdb_files(output_directory):
    frame_files = sorted(output_directory.glob("frame_*.vdb"))
    if frame_files:
        return frame_files
    return sorted(output_directory.glob("*.vdb"))


def _volume_data_matches_output_directory(volume_data, output_directory):
    if volume_data is None:
        return False
    filepath = getattr(volume_data, "filepath", "")
    if not filepath:
        return False
    try:
        resolved_filepath = Path(bpy.path.abspath(filepath)).resolve()
    except Exception:
        try:
            resolved_filepath = Path(filepath).resolve()
        except Exception:
            return False
    return _path_is_same_or_within_directory(resolved_filepath, output_directory)


def _object_matches_output_directory(volume_object, output_directory):
    if volume_object.type != "VOLUME":
        return False
    if volume_object.get(_OUTPUT_PATH_KEY) == str(output_directory):
        return True
    if volume_object.get(_LEGACY_OUTPUT_PATH_KEY) == str(output_directory):
        return True
    if volume_object.get(_AUTO_IMPORT_KEY) or volume_object.get(_LEGACY_AUTO_IMPORT_KEY):
        return _volume_data_matches_output_directory(volume_object.data, output_directory)
    return False


def _remove_previous_baked_volume(output_directory):
    output_directory = Path(output_directory).resolve()
    removed_count = 0
    matched_volume_data = set()
    for volume_object in list(bpy.data.objects):
        if not _object_matches_output_directory(volume_object, output_directory):
            continue
        volume_data = volume_object.data if volume_object.type == "VOLUME" else None
        if volume_data is not None:
            matched_volume_data.add(volume_data)
        bpy.data.objects.remove(volume_object, do_unlink=True)
        removed_count += 1
    for volume_data in list(bpy.data.volumes):
        if volume_data not in matched_volume_data and not _volume_data_matches_output_directory(
            volume_data,
            output_directory,
        ):
            continue
        try:
            bpy.data.volumes.remove(volume_data)
        except TypeError:
            bpy.data.volumes.remove(volume_data)
    return removed_count


def _delete_baked_vdb_files(output_directory):
    output_directory = Path(output_directory).resolve()
    if not output_directory.exists():
        return 0
    deleted_files = 0
    for vdb_file in list(_iter_output_directory_vdb_files(output_directory)):
        vdb_file.unlink(missing_ok=True)
        deleted_files += 1
    return deleted_files


def _remove_empty_bake_subdirectories(output_directory):
    """Remove empty Continuum Flow bake subdirectories below one base output directory."""
    output_directory = Path(output_directory).resolve()
    if not output_directory.exists() or not output_directory.is_dir():
        return 0

    removed_directories = 0
    for child_directory in sorted(output_directory.iterdir(), reverse=True):
        if not child_directory.is_dir():
            continue
        if not (
            child_directory.name.startswith(_BAKE_OUTPUT_DIRECTORY_PREFIX) or
            child_directory.name.startswith(_LEGACY_BAKE_OUTPUT_DIRECTORY_PREFIX)
        ):
            continue
        try:
            child_directory.rmdir()
            removed_directories += 1
        except OSError:
            pass
    return removed_directories


def _remove_bake_output_directory(output_directory):
    output_directory = Path(output_directory).resolve()
    if not output_directory.exists() or not output_directory.is_dir():
        return
    if output_directory.name.startswith(_BAKE_OUTPUT_DIRECTORY_PREFIX) or output_directory.name.startswith(_LEGACY_BAKE_OUTPUT_DIRECTORY_PREFIX):
        shutil.rmtree(output_directory, ignore_errors=True)


def _refresh_after_bake_cleanup(context):
    scene = getattr(context, "scene", None) or getattr(bpy.context, "scene", None)
    if scene is not None:
        scene.frame_set(scene.frame_current)
    _tag_all_areas_redraw(context)


def _refresh_after_live_preview_update(context=None, frame_value=None):
    scene = getattr(context, "scene", None) if context is not None else None
    if scene is None:
        scene = getattr(bpy.context, "scene", None)
    if scene is not None and frame_value is not None:
        try:
            scene.frame_set(int(frame_value))
        except Exception:
            pass
    _tag_all_areas_redraw(context)


def _free_bake(context, config_dict=None, output_directories=None):
    cleanup_directories = []
    seen_paths = set()

    for output_directory in output_directories or ():
        resolved_output_directory = Path(output_directory).resolve()
        resolved_output_directory_str = str(resolved_output_directory)
        if resolved_output_directory_str in seen_paths:
            continue
        seen_paths.add(resolved_output_directory_str)
        cleanup_directories.append(resolved_output_directory)

    if not cleanup_directories:
        cleanup_config = config_dict or _LAST_BAKE_CONFIG_DICT
        cleanup_directories = _output_directories_from_config(cleanup_config) if cleanup_config else []

    removed_volumes = 0
    deleted_files = 0
    for output_directory in cleanup_directories:
        removed_volumes += _remove_previous_baked_volume(output_directory)
        deleted_files += _delete_baked_vdb_files(output_directory)
        _remove_empty_bake_subdirectories(output_directory)
        _remove_bake_output_directory(output_directory)
    _refresh_after_bake_cleanup(context)
    _set_bake_state_active(False, context=context)
    return removed_volumes, deleted_files


def _apply_imported_volume_sequence_settings(volume_data, start_frame, frame_count):
    if volume_data is None:
        return
    volume_data.is_sequence = True
    volume_data.frame_duration = max(1, int(frame_count))
    volume_data.frame_start = int(start_frame)
    if hasattr(volume_data, "frame_offset"):
        volume_data.frame_offset = 0
    if hasattr(volume_data, "sequence_mode"):
        volume_data.sequence_mode = "CLIP"


def _managed_volume_display_name(output_directory):
    return f"Continuum Flow {_display_output_directory_name(output_directory)}"


def _apply_imported_volume_object_settings(volume_object, output_directory):
    if volume_object is None or volume_object.type != "VOLUME":
        return
    volume_object.location = (0.0, 0.0, 0.0)
    volume_object.matrix_world = Matrix.Identity(4)
    volume_object.name = _managed_volume_display_name(output_directory)
    volume_object[_AUTO_IMPORT_KEY] = True
    volume_object[_LEGACY_AUTO_IMPORT_KEY] = True
    volume_object[_OUTPUT_PATH_KEY] = str(output_directory)
    volume_object[_LEGACY_OUTPUT_PATH_KEY] = str(output_directory)


def _volume_collection_for_import():
    context_collection = getattr(bpy.context, "collection", None)
    if context_collection is not None:
        return context_collection
    scene = getattr(bpy.context, "scene", None)
    if scene is not None:
        return getattr(scene, "collection", None)
    return None


def _reload_volume_data(volume_data):
    if volume_data is None:
        return
    reload_method = getattr(volume_data, "reload", None)
    if callable(reload_method):
        reload_method()
        return
    grids = getattr(volume_data, "grids", None)
    load_method = getattr(grids, "load", None)
    if callable(load_method):
        load_method()


def _managed_volume_objects(output_directory):
    output_directory = Path(output_directory).resolve()
    return [
        volume_object
        for volume_object in bpy.data.objects
        if _object_matches_output_directory(volume_object, output_directory)
    ]


def _refresh_imported_vdb_sequence(output_directory, vdb_files, start_frame):
    first_file = vdb_files[0]
    for imported_object in _managed_volume_objects(output_directory):
        if imported_object.type != "VOLUME" or imported_object.data is None:
            continue
        _apply_imported_volume_sequence_settings(imported_object.data, start_frame, len(vdb_files))
        imported_object.data.filepath = str(first_file)
        _reload_volume_data(imported_object.data)
        _apply_imported_volume_object_settings(imported_object, output_directory)
        return 1
    return 0


def _import_baked_vdb_sequence(output_directory, vdb_files, start_frame):
    _remove_previous_baked_volume(output_directory)
    first_file = vdb_files[0]
    volume_name = _managed_volume_display_name(output_directory)
    volume_data = bpy.data.volumes.new(volume_name)
    volume_object = None
    try:
        volume_data.filepath = str(first_file)
        _apply_imported_volume_sequence_settings(volume_data, start_frame, len(vdb_files))
        _reload_volume_data(volume_data)
        volume_object = bpy.data.objects.new(volume_name, volume_data)
        _apply_imported_volume_object_settings(volume_object, output_directory)
        target_collection = _volume_collection_for_import()
        if target_collection is None:
            raise RuntimeError("No active Blender collection is available for the imported volume.")
        target_collection.objects.link(volume_object)
        return 1
    except Exception:
        if volume_object is not None:
            try:
                bpy.data.objects.remove(volume_object)
            except Exception:
                pass
        try:
            bpy.data.volumes.remove(volume_data)
        except Exception:
            pass
        raise


def _ensure_baked_vdb_sequence_imported(output_directory, vdb_files, start_frame):
    refreshed_count = _refresh_imported_vdb_sequence(output_directory, vdb_files, start_frame)
    if refreshed_count > 0:
        return refreshed_count
    return _import_baked_vdb_sequence(output_directory, vdb_files, start_frame)


def _iter_live_preview_output_entries(config_dict):
    for simulation_cfg in _iter_simulation_dicts(config_dict):
        viewers = tuple(_iter_viewer_dicts(simulation_cfg))
        if not viewers:
            continue
        if not any(bool(viewer_cfg.get("live_preview", True)) for viewer_cfg in viewers):
            continue
        settings = simulation_cfg.get("settings") or {}
        start_frame = int(settings.get("start_frame", 1))
        for output_cfg in _iter_output_dicts(simulation_cfg):
            output_path = output_cfg.get("output_path", "")
            if not output_path:
                continue
            yield Path(output_path).resolve(), start_frame, output_cfg


def _live_preview_refresh_interval(config_dict, default_fps=24):
    intervals = []
    for _output_directory, _start_frame, output_cfg in _iter_live_preview_output_entries(config_dict):
        try:
            output_fps = max(1.0, float(output_cfg.get("fps", default_fps)))
        except (TypeError, ValueError):
            output_fps = float(default_fps)
        intervals.append(1.0 / output_fps)
    if not intervals:
        return None
    return min(intervals)


def _refresh_live_preview(config_dict, context=None, state=None):
    preview_state = state if isinstance(state, dict) else {}
    refreshed_count = 0
    latest_frame = None

    for output_directory, start_frame, _output_cfg in _iter_live_preview_output_entries(config_dict):
        if not output_directory.exists():
            continue
        vdb_files = _baked_vdb_files(output_directory)
        frame_count = len(vdb_files)
        if frame_count <= 0:
            continue

        output_key = str(output_directory)
        previous_frame_count = int((preview_state.get(output_key) or {}).get("frame_count", 0))
        if frame_count <= previous_frame_count:
            continue

        refreshed_count += _ensure_baked_vdb_sequence_imported(output_directory, vdb_files, start_frame)
        preview_state[output_key] = {"frame_count": frame_count}
        current_latest_frame = int(start_frame) + frame_count - 1
        latest_frame = current_latest_frame if latest_frame is None else max(latest_frame, current_latest_frame)

    if refreshed_count > 0:
        _refresh_after_live_preview_update(context=context, frame_value=latest_frame)
    return refreshed_count


def _import_baked_vdbs(config_dict):
    imported_count = 0
    missing_directories = []
    for output_directory in _output_directories_from_config(config_dict):
        if not output_directory.exists():
            missing_directories.append(str(output_directory))
            continue
        vdb_files = _baked_vdb_files(output_directory)
        if not vdb_files:
            continue
        start_frame = _start_frame_for_output_directory(config_dict, output_directory)
        imported_count += _ensure_baked_vdb_sequence_imported(output_directory, vdb_files, start_frame)
    return imported_count, missing_directories
