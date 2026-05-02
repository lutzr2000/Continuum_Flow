"""Runtime helpers and operators for starting, monitoring, and cancelling BlenderCFD bakes."""

import importlib.util
import json
import queue
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import bpy
import blendercfd_general_nodes as GeneralNodes
from bpy.props import StringProperty


def _load_sibling_module(module_name, file_name):
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    if "__file__" in globals():
        module_path = Path(__file__).resolve().with_name(file_name)
    else:
        module_path = (Path.cwd() / "UI" / "Nodes" / file_name).resolve()

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module '{module_name}' from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


BakeStorage = _load_sibling_module("blendercfd_bake_storage", "bake_storage.py")

BlenderCFDConfigModule = GeneralNodes._load_ui_module(
    "blendercfd_create_config_dict",
    str(Path("Export") / "config.py"),
    (".Export.config", "UI.Export.config", "Export.config"),
)
BlenderCFDHostWriterModule = GeneralNodes._load_ui_module(
    "blendercfd_host_writer",
    "Bake/writer_manager.py",
    (".Bake.writer_manager", "UI.Bake.writer_manager", "Bake.writer_manager"),
)

tree_has_invalid_links = GeneralNodes._sockets_module.tree_has_invalid_links
set_bake_running = GeneralNodes.set_bake_running

PROGRESS_EVENT_PREFIX = "__BLENDERCFD_PROGRESS__ "
_STATUS_PROGRESS_PERCENT = 0.0
_STATUS_PROGRESS_FACTOR = 0.0
_ACTIVE_BAKE_OPERATOR = None
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


def _kernel_directory():
    if "__file__" in globals():
        return Path(__file__).resolve().parents[2] / "Solver" / "Kernel_GPU"
    return (Path.cwd() / "Solver" / "Kernel_GPU").resolve()


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
        "import Solver.Kernel_GPU.kernel as KernelMain; "
        "KernelMain.main(json.load(sys.stdin))"
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
    writer_server = BlenderCFDHostWriterModule.HostVDBWriterServer(
        writer_process_count=_writer_process_count_from_config(config_dict)
    )
    writer_server.start()
    config_dict["_host_vdb_writer"] = writer_server.endpoint()
    output_queue = queue.Queue()
    try:
        process, python_executable = _run_kernel(config_dict)
    except Exception:
        writer_server.stop()
        raise
    output_thread = threading.Thread(
        target=_read_kernel_output,
        args=(process, output_queue),
        daemon=True,
    )
    output_thread.start()
    return {
        "writer_server": writer_server,
        "process": process,
        "python_executable": python_executable,
        "output_queue": output_queue,
        "output_thread": output_thread,
    }


class BlenderCFD_OT_bake(bpy.types.Operator):
    """Operator that exports the active node tree config and starts the kernel."""

    bl_idname = "blendercfd.bake"
    bl_label = "Bake BlenderCFD"
    bl_description = "Start the BlenderCFD bake process"
    output_path_hint: StringProperty(default="", options={"SKIP_SAVE"})  # type: ignore

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
        BakeStorage._clear_bake_cancel_flag(self._config_dict)
        BakeStorage._cleanup_geometry_cache(self._config_dict)
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
        if not self._config_dict:
            return
        cancel_flag_path = (((self._config_dict.get("meta") or {}).get("cancel_flag_path")) or "").strip()
        if not cancel_flag_path:
            return
        try:
            Path(cancel_flag_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cancel_flag_path).write_text("cancel\n", encoding="utf-8")
        except OSError:
            self._cancel_running_process()

    def modal(self, context, event):
        if event.type == "ESC":
            self._request_cancel()
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
        if return_code != 0:
            self.report(
                {"ERROR"},
                f"Bake failed: Kernel process failed with exit code {return_code}{self._recent_kernel_output_detail()}",
            )
            return {"CANCELLED"}
        try:
            imported_count, missing_directories = BakeStorage._import_baked_vdbs(self._config_dict)
        except Exception as exc:
            if self._solver_divergence_message is not None:
                self.report({"WARNING"}, f"{self._solver_divergence_message} VDB import failed: {exc}")
            else:
                self.report({"WARNING"}, f"Bake finished, but VDB import failed: {exc}")
            return {"FINISHED"}
        if self._solver_divergence_message is not None:
            if imported_count > 0:
                self.report(
                    {"WARNING"},
                    f"{self._solver_divergence_message} Imported {imported_count} VDB volume(s) up to that point.",
                )
            elif missing_directories:
                self.report({"WARNING"}, f"{self._solver_divergence_message} No VDB output directory was found.")
            else:
                self.report({"WARNING"}, f"{self._solver_divergence_message} No VDB files were found to import.")
            return {"FINISHED"}
        if self._cancel_requested:
            if imported_count > 0:
                self.report(
                    {"WARNING"},
                    f"BlenderCFD bake cancelled. Imported {imported_count} VDB volume(s) up to that point.",
                )
            elif missing_directories:
                self.report({"WARNING"}, "BlenderCFD bake cancelled. No VDB output directory was found.")
            else:
                self.report({"WARNING"}, "BlenderCFD bake cancelled. No VDB files were found to import.")
            return {"FINISHED"}
        if imported_count > 0:
            self.report({"INFO"}, f"Bake finished. Imported {imported_count} VDB volume(s).")
        elif missing_directories:
            self.report({"WARNING"}, "Bake finished, but no VDB output directory was found.")
        else:
            self.report({"WARNING"}, "Bake finished, but no VDB files were found to import.")
        return {"FINISHED"}

    def execute(self, context):
        hinted_output_directory = BakeStorage._resolve_output_directory_hint(self.output_path_hint)
        if hinted_output_directory is not None and BakeStorage._output_directory_has_vdbs(hinted_output_directory):
            removed_volumes, deleted_files = BakeStorage._free_bake(
                context,
                output_directories=(hinted_output_directory,),
            )
            self.report(
                {"INFO"},
                f"Freed bake data. Removed {removed_volumes} volume object(s) and deleted {deleted_files} VDB file(s).",
            )
            return {"FINISHED"}

        resolve_node_tree = getattr(BlenderCFDConfigModule, "_resolve_node_tree", None)
        node_tree = resolve_node_tree(context) if callable(resolve_node_tree) else None
        if tree_has_invalid_links(node_tree):
            self.report({"ERROR"}, "Bake disabled: invalid socket connection in the node tree.")
            return {"CANCELLED"}

        global _ACTIVE_BAKE_OPERATOR
        bake_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        geometry_output_directory = hinted_output_directory or _project_root_directory()
        geometry_cache_dir = BakeStorage._bake_geometry_cache_directory(geometry_output_directory, bake_token)
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
        if BakeStorage._config_has_baked_vdbs(live_config_dict):
            active_operator = _ACTIVE_BAKE_OPERATOR
            if active_operator is not None and active_operator._session is not None:
                BakeStorage._cleanup_geometry_cache(live_config_dict)
                active_operator._request_cancel()
                self.report({"INFO"}, "Bake cancellation requested. Press Free Bake again after the bake stops.")
                return {"FINISHED"}
            removed_volumes, deleted_files = BakeStorage._free_bake(context, config_dict=live_config_dict)
            BakeStorage._cleanup_geometry_cache(live_config_dict)
            self.report(
                {"INFO"},
                f"Freed bake data. Removed {removed_volumes} volume object(s) and deleted {deleted_files} VDB file(s).",
            )
            return {"FINISHED"}
        try:
            self._config_dict = BakeStorage._prepare_config_for_bake(live_config_dict, bake_token=bake_token)
            self._session = _start_bake_session(self._config_dict)
        except ModuleNotFoundError as exc:
            BakeStorage._cleanup_geometry_cache(live_config_dict)
            set_bake_running(False, context=context)
            missing_module = getattr(exc, "name", None) or str(exc)
            self.report({"ERROR"}, f"Bake failed: missing Python module '{missing_module}' in {sys.executable}")
            return {"CANCELLED"}
        except Exception as exc:
            BakeStorage._cleanup_geometry_cache(live_config_dict)
            set_bake_running(False, context=context)
            self.report({"ERROR"}, f"Bake failed: {exc}")
            return {"CANCELLED"}
        BakeStorage._set_bake_state_active(True, context=context, config_dict=self._config_dict)
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
        self.report({"INFO"}, "Bake cancellation requested.")
        return {"FINISHED"}


classes = (
    BlenderCFD_OT_bake,
    BlenderCFD_OT_cancel_bake,
)
