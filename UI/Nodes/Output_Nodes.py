import bpy
import copy
import json
import queue
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty

import blendercfd_general_nodes as GeneralNodes


BlenderCFDConfigModule = GeneralNodes._load_ui_module(
    "blendercfd_create_config_dict",
    ("Config_Export.py", "Create_Config_Dict.py"),
    (".Config_Export", "UI.Config_Export", "Config_Export", "UI.Create_Config_Dict", "Create_Config_Dict"),
)
BlenderCFDHostWriterModule = GeneralNodes._load_ui_module(
    "blendercfd_host_writer",
    "Host_Writer.py",
    (".Host_Writer", "UI.Host_Writer", "Host_Writer"),
)

BlenderCFDNodeTree = GeneralNodes.BlenderCFDNodeTree
BlenderCFDResultSocket = GeneralNodes.BlenderCFDResultSocket
tree_has_invalid_links = GeneralNodes._sockets_module.tree_has_invalid_links
is_bake_running = GeneralNodes.is_bake_running
set_bake_running = GeneralNodes.set_bake_running

PROGRESS_EVENT_PREFIX = "__BLENDERCFD_PROGRESS__ "
_STATUS_PROGRESS_PERCENT = 0.0
_STATUS_PROGRESS_FACTOR = 0.0
_ACTIVE_BAKE_OPERATOR = None
_LAST_BAKE_CONFIG_DICT = None
_BAKE_OUTPUT_DIRECTORY_PREFIX = ".blendercfd_bake_"
_SOLVER_DIVERGENCE_MESSAGES = (
    "solver divergence",
    "solver diverged",
)


def _draw_blendercfd_status_progress(header, context):
    layout = header.layout
    layout.separator_spacer()
    layout.label(text="BlenderCFD Bake:")
    row = layout.row()
    row.ui_units_x = 10
    row.progress(factor=_STATUS_PROGRESS_FACTOR, text=f"{_STATUS_PROGRESS_PERCENT:.1f}%", type="BAR")
    row = layout.row(align=True)
    row.operator("blendercfd.cancel_bake", text="", icon="PANEL_CLOSE", emboss=False)
    layout.separator_spacer()


def _set_status_progress_values(percent):
    global _STATUS_PROGRESS_PERCENT, _STATUS_PROGRESS_FACTOR
    _STATUS_PROGRESS_PERCENT = max(0.0, min(100.0, float(percent)))
    _STATUS_PROGRESS_FACTOR = _STATUS_PROGRESS_PERCENT / 100.0


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
        if not child_directory.name.startswith(_BAKE_OUTPUT_DIRECTORY_PREFIX):
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
    if output_directory.name.startswith(_BAKE_OUTPUT_DIRECTORY_PREFIX):
        return output_directory.parent.name or output_directory.name
    return output_directory.name


def _prepared_output_directory(base_output_path, bake_token):
    return Path(base_output_path).resolve() / f"{_BAKE_OUTPUT_DIRECTORY_PREFIX}{bake_token}"


def _bake_geometry_cache_directory(bake_token):
    return _project_root_directory() / ".blendercfd_bake_cache" / bake_token / "geometry"


def _cleanup_geometry_cache(config_dict):
    if not config_dict:
        return
    geometry_cache_dir = (((config_dict.get("meta") or {}).get("geometry_cache_dir")) or "").strip()
    if not geometry_cache_dir:
        return

    cache_root = (_project_root_directory() / ".blendercfd_bake_cache").resolve()
    geometry_cache_path = Path(geometry_cache_dir).resolve()

    try:
        geometry_cache_path.relative_to(cache_root)
    except ValueError:
        return

    shutil.rmtree(geometry_cache_path, ignore_errors=True)

    current_path = geometry_cache_path.parent
    while True:
        try:
            current_path.relative_to(cache_root)
        except ValueError:
            break

        try:
            current_path.rmdir()
        except OSError:
            break

        if current_path == cache_root:
            break
        current_path = current_path.parent


def _prepare_config_for_bake(config_dict, bake_token=None):
    prepared_config = copy.deepcopy(config_dict)
    if not bake_token:
        bake_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prepared_config.setdefault("meta", {})["bake_token"] = bake_token
    for simulation_cfg in prepared_config.get("simulations", ()):
        for output_cfg in simulation_cfg.get("outputs", ()):
            output_path = output_cfg.get("output_path", "")
            if not output_path:
                continue
            base_output_path = str(Path(output_path).resolve())
            output_cfg["base_output_path"] = base_output_path
            output_cfg["output_path"] = str(_prepared_output_directory(base_output_path, bake_token))
    return prepared_config


def _kernel_directory():
    if "__file__" in globals():
        return Path(__file__).resolve().parents[2] / "Kernel"
    return (Path.cwd() / "Kernel").resolve()


def _project_root_directory():
    if "__file__" in globals():
        return Path(__file__).resolve().parents[2]
    return Path.cwd().resolve()


def _resolve_python_executable():
    python_executable = _project_root_directory() / "BlenderCFD_env" / "Scripts" / "python.exe"
    if not python_executable.exists():
        raise FileNotFoundError(f"Python executable not found: {python_executable}")
    return str(python_executable)


def _run_kernel(config_dict):
    python_executable = _resolve_python_executable()
    kernel_dir = _kernel_directory()
    project_root = _project_root_directory()
    bootstrap_code = (
        "import json, sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "from Kernel import Kernel_GPU; "
        "Kernel_GPU.main(json.load(sys.stdin))"
    )
    process = subprocess.Popen(
        [python_executable, "-u", "-c", bootstrap_code, str(project_root)],
        cwd=str(kernel_dir),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    process.stdin.write(json.dumps(config_dict))
    process.stdin.close()
    return process, python_executable


def _handle_kernel_output_line(line, output_tail):
    output_tail.append(line.rstrip())
    if len(output_tail) > 40:
        del output_tail[0]
    if line.startswith(PROGRESS_EVENT_PREFIX):
        try:
            payload = json.loads(line[len(PROGRESS_EVENT_PREFIX):])
            return max(0.0, min(100.0, float(payload.get("percent", 0.0))))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    sys.stdout.write(line)
    sys.stdout.flush()
    return None


def _solver_divergence_message_from_line(line):
    normalized_line = line.strip().lower()
    if any(message in normalized_line for message in _SOLVER_DIVERGENCE_MESSAGES):
        return "Bake stopped: the solver diverged."
    return None


def _writer_process_count_from_config(config_dict):
    simulations = config_dict.get("simulations") or ()
    if not simulations:
        return 4
    outputs = simulations[0].get("outputs") or ()
    if not outputs:
        return 4
    performance_cfg = outputs[0].get("performance", {})
    return max(1, int(performance_cfg.get("writer_processes", 4)))


def _read_kernel_output(process, output_queue):
    try:
        if process.stdout is not None:
            for line in process.stdout:
                output_queue.put(("line", line))
    except Exception as exc:
        output_queue.put(("error", str(exc)))


def _start_bake_session(config_dict):
    writer_server = BlenderCFDHostWriterModule.HostVDBWriterServer(writer_process_count=_writer_process_count_from_config(config_dict))
    writer_server.start()
    config_dict["_host_vdb_writer"] = writer_server.endpoint()
    output_queue = queue.Queue()
    try:
        process, python_executable = _run_kernel(config_dict)
    except Exception:
        writer_server.stop()
        raise
    output_thread = threading.Thread(target=_read_kernel_output, args=(process, output_queue), daemon=True)
    output_thread.start()
    return {
        "writer_server": writer_server,
        "process": process,
        "python_executable": python_executable,
        "output_queue": output_queue,
        "output_thread": output_thread,
    }


def _output_directories_from_config(config_dict):
    output_directories = []
    seen_paths = set()
    for simulation_cfg in config_dict.get("simulations", ()):
        for output_cfg in simulation_cfg.get("outputs", ()):
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
    for simulation_cfg in config_dict.get("simulations", ()):
        settings = simulation_cfg.get("settings", {})
        start_frame = int(settings.get("start_frame", 1))
        for output_cfg in simulation_cfg.get("outputs", ()):
            output_path = output_cfg.get("output_path", "")
            if not output_path:
                continue
            if Path(output_path).resolve() == output_directory:
                return start_frame
    return 1


def _config_has_baked_vdbs(config_dict):
    return any(_output_directory_has_vdbs(output_directory) for output_directory in _output_directories_from_config(config_dict))


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
    if volume_object.get("blendercfd_output_path") == str(output_directory):
        return True
    if volume_object.get("blendercfd_auto_import"):
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
        if volume_data not in matched_volume_data and not _volume_data_matches_output_directory(volume_data, output_directory):
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
    """Remove empty BlenderCFD bake subdirectories below one base output directory."""
    output_directory = Path(output_directory).resolve()
    if not output_directory.exists() or not output_directory.is_dir():
        return 0

    removed_directories = 0
    for child_directory in sorted(output_directory.iterdir(), reverse=True):
        if not child_directory.is_dir():
            continue
        if not child_directory.name.startswith(_BAKE_OUTPUT_DIRECTORY_PREFIX):
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
    if output_directory.name.startswith(_BAKE_OUTPUT_DIRECTORY_PREFIX):
        shutil.rmtree(output_directory, ignore_errors=True)


def _purge_blender_baked_data(context):
    try:
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except (RuntimeError, TypeError):
        pass
    scene = getattr(context, "scene", None) or getattr(bpy.context, "scene", None)
    if scene is not None:
        scene.frame_set(scene.frame_current)
    _tag_all_areas_redraw(context)


def _free_bake(context, config_dict=None):
    cleanup_config = config_dict or _LAST_BAKE_CONFIG_DICT
    output_directories = _output_directories_from_config(cleanup_config) if cleanup_config else []
    removed_volumes = 0
    deleted_files = 0
    for output_directory in output_directories:
        removed_volumes += _remove_previous_baked_volume(output_directory)
        deleted_files += _delete_baked_vdb_files(output_directory)
        _remove_empty_bake_subdirectories(output_directory)
        _remove_bake_output_directory(output_directory)
    _purge_blender_baked_data(context)
    _set_bake_state_active(False, context=context)
    return removed_volumes, deleted_files


def _apply_imported_volume_sequence_settings(volume_data, start_frame, frame_count):
    if volume_data is None:
        return
    volume_data.is_sequence = True
    volume_data.frame_duration = max(1, int(frame_count))
    volume_data.frame_start = int(start_frame)
    if hasattr(volume_data, "sequence_mode"):
        volume_data.sequence_mode = "CLIP"


def _import_baked_vdb_sequence(output_directory, vdb_files, start_frame):
    _remove_previous_baked_volume(output_directory)
    existing_objects = set(bpy.data.objects)
    first_file = vdb_files[0]
    bpy.ops.object.volume_import(
        filepath=str(first_file),
        directory=str(output_directory),
        files=[{"name": vdb_file.name} for vdb_file in vdb_files],
        use_sequence_detection=True,
    )
    imported_objects = [obj for obj in bpy.data.objects if obj not in existing_objects]
    if not imported_objects and bpy.context.object is not None:
        imported_objects = [bpy.context.object]
    for imported_object in imported_objects:
        if imported_object.type != "VOLUME":
            continue
        _apply_imported_volume_sequence_settings(imported_object.data, start_frame, len(vdb_files))
        imported_object.name = f"BlenderCFD {_display_output_directory_name(output_directory)}"
        imported_object["blendercfd_auto_import"] = True
        imported_object["blendercfd_output_path"] = str(output_directory)
    return len([obj for obj in imported_objects if obj.type == "VOLUME"])


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
        imported_count += _import_baked_vdb_sequence(output_directory, vdb_files, start_frame)
    return imported_count, missing_directories


class BlenderCFDOutputNode(bpy.types.Node):
    """Node used to configure which simulation results should be written to disk."""

    bl_idname = "BLENDERCFD_OUTPUT_NODE"
    bl_label = "Output"
    bl_icon = "OUTPUT"
    bl_width_default = 260.0
    bl_width_min = 240.0
    bl_width_max = 420.0

    fps: IntProperty(name="FPS", default=24, min=1, max=240, soft_min=1, description="Output frame rate",options=set()) # type: ignore
    writer_processes: IntProperty(name="Writers", default=4, min=1, max=16, soft_min=1, soft_max=8, description="How many writer processes are launched, usually four give the best performance",options=set())  # type: ignore
    output_precision: EnumProperty(  # type: ignore
        name="Precision",
        items=(
            ("float16", "Half (float16)", "Write VDB grids with 16-bit floating point values"),
            ("float32", "Full (float32)", "Write VDB grids with 32-bit floating point values"),
        ),
        default="float16",
    )
    export_velocity: BoolProperty(name="velocity", default=False)  # type: ignore
    sparse_velocity: BoolProperty(name="sparse", default=True)  # type: ignore
    export_p: BoolProperty(name="pressure", default=False)  # type: ignore
    sparse_p: BoolProperty(name="sparse", default=True)  # type: ignore
    export_t: BoolProperty(name="temperature", default=False)  # type: ignore
    sparse_t: BoolProperty(name="sparse", default=True)  # type: ignore
    export_smoke: BoolProperty(name="density", default=True)  # type: ignore
    sparse_smoke: BoolProperty(name="sparse", default=True)  # type: ignore
    export_fuel: BoolProperty(name="fuel", default=False)  # type: ignore
    sparse_fuel: BoolProperty(name="sparse", default=True)  # type: ignore
    export_flame: BoolProperty(name="flame", default=True)  # type: ignore
    sparse_flame: BoolProperty(name="sparse", default=True)  # type: ignore
    output_path: StringProperty(name="Path", default="", subtype="DIR_PATH")  # type: ignore

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_input_socket(self):
        socket = self.inputs.get("Result")
        if socket is None:
            socket = self.inputs.new(BlenderCFDResultSocket.bl_idname, "Result")
        return socket

    def _sync_input_socket(self):
        self._ensure_input_socket()

    def _sync_defaults_from_scene(self, context):
        scene = getattr(context, "scene", None) or getattr(bpy.context, "scene", None)
        if scene is None:
            return
        render = getattr(scene, "render", None)
        if render is None:
            return
        fps = getattr(render, "fps", None)
        if fps is not None and self.fps == 24:
            self.fps = fps

    def init(self, context):
        self._sync_input_socket()
        self._sync_defaults_from_scene(context)

    def copy(self, node):
        self._sync_input_socket()

    def update(self):
        self._sync_input_socket()

    def _draw_field_row(self, layout, export_attr, sparse_attr, label=None):
        row = layout.row(align=True)
        row.prop(self, export_attr, text=label)
        sparse_row = row.row(align=True)
        sparse_row.enabled = bool(getattr(self, export_attr))
        sparse_row.prop(self, sparse_attr, text="sparse")

    def draw_buttons(self, context, layout):
        layout.enabled = not is_bake_running(context)
        layout.prop(self, "fps")
        layout.prop(self, "writer_processes")
        layout.prop(self, "output_precision")
        fields_box = layout.box()
        fields_box.label(text="Fields")
        fields_col = fields_box.column(align=True)
        self._draw_field_row(fields_col, "export_velocity", "sparse_velocity", label="velocity")
        self._draw_field_row(fields_col, "export_p", "sparse_p", label="pressure")
        self._draw_field_row(fields_col, "export_t", "sparse_t", label="temperature")
        self._draw_field_row(fields_col, "export_smoke", "sparse_smoke", label="density")
        self._draw_field_row(fields_col, "export_fuel", "sparse_fuel", label="fuel")
        self._draw_field_row(fields_col, "export_flame", "sparse_flame", label="flame")
        layout.prop(self, "output_path")
        layout.separator()
        resolved_output_path = bpy.path.abspath(self.output_path) if self.output_path else ""
        button_is_free_bake = bool(resolved_output_path) and _output_directory_has_vdbs(resolved_output_path)
        has_invalid_links = tree_has_invalid_links(getattr(self, "id_data", None))
        if has_invalid_links:
            layout.label(text="Bake disabled: invalid socket connection", icon="ERROR")
        button_row = layout.row()
        button_row.enabled = (not is_bake_running(context)) and not has_invalid_links
        button_row.operator("blendercfd.bake", text="Free Bake" if button_is_free_bake else "Bake", icon="TRASH" if button_is_free_bake else "RENDER_STILL")


class BlenderCFD_OT_bake(bpy.types.Operator):
    """Operator that exports the active node tree config and starts the kernel."""

    bl_idname = "blendercfd.bake"
    bl_label = "Bake BlenderCFD"
    bl_description = "Start the BlenderCFD bake process"

    _timer = None
    _session = None
    _config_dict = None
    _output_tail = None
    _progress_percent = 0.0
    _progress_factor = 0.0
    _reader_error = None
    _solver_divergence_message = None
    _window_manager = None
    _workspace = None
    _cancel_requested = False

    def _drain_kernel_output(self):
        output_queue = self._session["output_queue"]
        while True:
            try:
                event_type, payload = output_queue.get_nowait()
            except queue.Empty:
                break
            if event_type == "line":
                percent = _handle_kernel_output_line(payload, self._output_tail)
                if percent is not None:
                    self._progress_percent = percent
                    self._progress_factor = percent / 100.0
                    _set_status_progress_values(percent)
                divergence_message = _solver_divergence_message_from_line(payload)
                if divergence_message is not None:
                    self._solver_divergence_message = divergence_message
            elif event_type == "error":
                self._reader_error = payload

    def _set_status_progress(self, context):
        if self._workspace is None:
            self._workspace = context.workspace
        self._workspace.status_text_set(_draw_blendercfd_status_progress)
        self._tag_status_bar_redraw(context)

    def _clear_status_progress(self, context):
        workspace = self._workspace or context.workspace
        workspace.status_text_set(None)
        self._workspace = None
        self._tag_status_bar_redraw(context)

    def _tag_status_bar_redraw(self, context):
        for window in context.window_manager.windows:
            screen = getattr(window, "screen", None)
            if screen is None:
                continue
            for area in screen.areas:
                if area.type == "STATUSBAR":
                    area.tag_redraw()

    def _recent_kernel_output_detail(self):
        if not self._output_tail:
            return ""
        recent_output = "\n".join(self._output_tail[-10:])
        return f"\nRecent kernel output:\n{recent_output}" if recent_output else ""

    def _cleanup_bake(self, context):
        global _ACTIVE_BAKE_OPERATOR
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        self._clear_status_progress(context)
        self._window_manager = None
        if self._session is not None:
            self._session["writer_server"].stop()
            self._session = None
        _cleanup_geometry_cache(self._config_dict)
        if _ACTIVE_BAKE_OPERATOR is self:
            _ACTIVE_BAKE_OPERATOR = None
        set_bake_running(False, context=context)

    def _cancel_running_process(self):
        if self._session is None:
            return
        process = self._session["process"]
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def _request_cancel(self):
        self._cancel_requested = True
        self._cancel_running_process()

    def modal(self, context, event):
        if event.type == "ESC":
            self._request_cancel()
        if self._cancel_requested:
            self._cleanup_bake(context)
            self.report({"WARNING"}, "BlenderCFD bake cancelled.")
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        self._drain_kernel_output()
        self._set_status_progress(context)
        process = self._session["process"]
        return_code = process.poll()
        if return_code is None:
            return {"RUNNING_MODAL"}
        self._drain_kernel_output()
        reader_error = self._reader_error
        self._cleanup_bake(context)
        if reader_error:
            self.report({"ERROR"}, f"Bake failed while reading kernel output: {reader_error}")
            return {"CANCELLED"}
        if self._solver_divergence_message is not None:
            self.report({"WARNING"}, self._solver_divergence_message)
            return {"CANCELLED"}
        if return_code != 0:
            self.report({"ERROR"}, f"Bake failed: Kernel process failed with exit code {return_code}{self._recent_kernel_output_detail()}")
            return {"CANCELLED"}
        try:
            imported_count, missing_directories = _import_baked_vdbs(self._config_dict)
        except Exception as exc:
            self.report({"WARNING"}, f"Bake finished, but VDB import failed: {exc}")
            return {"FINISHED"}
        if imported_count > 0:
            self.report({"INFO"}, f"Bake finished. Imported {imported_count} VDB volume(s).")
        elif missing_directories:
            self.report({"WARNING"}, "Bake finished, but no VDB output directory was found.")
        else:
            self.report({"WARNING"}, "Bake finished, but no VDB files were found to import.")
        return {"FINISHED"}

    def execute(self, context):
        resolve_node_tree = getattr(BlenderCFDConfigModule, "_resolve_node_tree", None)
        node_tree = resolve_node_tree(context) if callable(resolve_node_tree) else None
        if tree_has_invalid_links(node_tree):
            self.report({"ERROR"}, "Bake disabled: invalid socket connection in the node tree.")
            return {"CANCELLED"}
        global _ACTIVE_BAKE_OPERATOR
        bake_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        geometry_cache_dir = _bake_geometry_cache_directory(bake_token)
        try:
            live_config_dict = BlenderCFDConfigModule.build_config_dict(
                context,
                geometry_storage_dir=str(geometry_cache_dir),
            )
            live_config_dict.setdefault("meta", {})["geometry_cache_dir"] = str(geometry_cache_dir)
        except Exception as exc:
            shutil.rmtree(geometry_cache_dir, ignore_errors=True)
            self.report({"ERROR"}, f"Bake failed: {exc}")
            return {"CANCELLED"}
        if _config_has_baked_vdbs(live_config_dict):
            active_operator = _ACTIVE_BAKE_OPERATOR
            if active_operator is not None and active_operator._session is not None:
                _cleanup_geometry_cache(live_config_dict)
                active_operator._request_cancel()
                self.report({"INFO"}, "Bake cancellation requested. Press Free Bake again after the bake stops.")
                return {"FINISHED"}
            removed_volumes, deleted_files = _free_bake(context, config_dict=live_config_dict)
            _cleanup_geometry_cache(live_config_dict)
            self.report({"INFO"}, f"Freed bake data. Removed {removed_volumes} volume object(s) and deleted {deleted_files} VDB file(s).")
            return {"FINISHED"}
        try:
            self._config_dict = _prepare_config_for_bake(live_config_dict, bake_token=bake_token)
            self._session = _start_bake_session(self._config_dict)
        except ModuleNotFoundError as exc:
            _cleanup_geometry_cache(live_config_dict)
            set_bake_running(False, context=context)
            missing_module = getattr(exc, "name", None) or str(exc)
            self.report({"ERROR"}, f"Bake failed: missing Python module '{missing_module}' in {sys.executable}")
            return {"CANCELLED"}
        except Exception as exc:
            _cleanup_geometry_cache(live_config_dict)
            set_bake_running(False, context=context)
            self.report({"ERROR"}, f"Bake failed: {exc}")
            return {"CANCELLED"}
        _set_bake_state_active(True, context=context, config_dict=self._config_dict)
        set_bake_running(True, context=context)
        self._output_tail = []
        self._progress_percent = 0.0
        self._progress_factor = 0.0
        _set_status_progress_values(0.0)
        self._reader_error = None
        self._solver_divergence_message = None
        self._window_manager = context.window_manager
        self._workspace = context.workspace
        self._cancel_requested = False
        _ACTIVE_BAKE_OPERATOR = self
        self._set_status_progress(context)
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}


class BlenderCFD_OT_cancel_bake(bpy.types.Operator):
    """Cancel the currently running BlenderCFD bake from the status bar."""

    bl_idname = "blendercfd.cancel_bake"
    bl_label = "Cancel BlenderCFD Bake"
    bl_description = "Cancel the currently running BlenderCFD bake"

    def execute(self, context):
        active_operator = _ACTIVE_BAKE_OPERATOR
        if active_operator is None or active_operator._session is None:
            self.report({"WARNING"}, "No BlenderCFD bake is currently running.")
            return {"CANCELLED"}
        active_operator._request_cancel()
        active_operator._tag_status_bar_redraw(context)
        return {"FINISHED"}


classes = (
    BlenderCFDOutputNode,
    BlenderCFD_OT_bake,
    BlenderCFD_OT_cancel_bake,
)
