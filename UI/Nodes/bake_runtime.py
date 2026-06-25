import importlib.util
import json
import queue
import shutil
import subprocess
import sys
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import bpy

import continuum_flow_general_nodes as GeneralNodes
from bpy.props import StringProperty


def _load_sibling_module(module_name, file_name, legacy_names=()):
    for known_name in (module_name, *legacy_names):
        module = sys.modules.get(known_name)
        if module is not None:
            sys.modules[module_name] = module
            for legacy_name in legacy_names:
                sys.modules[legacy_name] = module
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
    for legacy_name in legacy_names:
        sys.modules[legacy_name] = module
    spec.loader.exec_module(module)
    return module


BakeStorage = _load_sibling_module(
    "continuum_flow_bake_storage",
    "bake_storage.py",
)

ContinuumFlowConfigModule = GeneralNodes._load_ui_module(
    "continuum_flow_create_config_dict",
    str(Path("Export") / "config.py"),
    (".Export.config", "UI.Export.config", "Export.config"),
)
ContinuumFlowHostWriterModule = GeneralNodes._load_ui_module(
    "continuum_flow_host_writer",
    "Bake/writer_manager.py",
    (".Bake.writer_manager", "UI.Bake.writer_manager", "Bake.writer_manager"),
)
ContinuumFlowEnvironmentModule = GeneralNodes._load_ui_module(
    "continuum_flow_environment",
    "Core/environment.py",
    (".Core.environment", "UI.Core.environment", "Core.environment"),
)

tree_has_invalid_links = GeneralNodes._sockets_module.tree_has_invalid_links
set_bake_running = GeneralNodes.set_bake_running

PROGRESS_EVENT_PREFIX = "__CONTINUUM_FLOW_PROGRESS__ "
_STATUS_PROGRESS_PERCENT = 0.0
_STATUS_PROGRESS_FACTOR = 0.0
_STATUS_PROGRESS_ETA = ""
_STATUS_PROGRESS_ETA_PLACEHOLDER = "Remaining: 00:00:00"
_STATUS_PROGRESS_ETA_DELAY_SECONDS = 2.0
_ACTIVE_BAKE_OPERATOR = None
_SOLVER_DIVERGENCE_MESSAGES = (
    "solver divergence",
    "solver diverged",
)


def _gpu_solver_available():
    """
    Return whether a usable CUDA backend is available in the solver environment.
    """
    try:
        python_executable = _resolve_python_executable()
        result = subprocess.run(
            [
                python_executable,
                "-c",
                "from numba import cuda; print(int(cuda.is_available()))",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        return result.returncode == 0 and result.stdout.strip() == "1"
    except Exception:
        return False


def _draw_continuum_flow_status_progress(header, context):
    layout = header.layout
    layout.separator_spacer()
    layout.label(text="Continuum Flow Bake:")
    row = layout.row(align=True)
    row.ui_units_x = 20
    split = row.split(factor=0.65, align=True)
    progress_row = split.row(align=True)
    progress_row.progress(
        factor=_STATUS_PROGRESS_FACTOR,
        text=f"{_STATUS_PROGRESS_PERCENT:.1f}%",
        type="BAR",
    )
    eta_row = split.row(align=True)
    eta_row.ui_units_x = 7
    eta_row.label(text=_STATUS_PROGRESS_ETA or _STATUS_PROGRESS_ETA_PLACEHOLDER)
    row = layout.row(align=True)
    row.operator("continuum_flow.cancel_bake", text="", icon="PANEL_CLOSE", emboss=False)
    layout.separator_spacer()


def _set_status_progress_values(percent, eta_text=""):
    global _STATUS_PROGRESS_PERCENT, _STATUS_PROGRESS_FACTOR, _STATUS_PROGRESS_ETA
    _STATUS_PROGRESS_PERCENT = max(0.0, min(100.0, float(percent)))
    _STATUS_PROGRESS_FACTOR = _STATUS_PROGRESS_PERCENT / 100.0
    _STATUS_PROGRESS_ETA = str(eta_text or "")


def _format_eta_seconds(seconds_remaining):
    total_seconds = max(0, int(round(float(seconds_remaining))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"Remaining: {hours:02d}:{minutes:02d}:{seconds:02d}"


def _solver_backend_from_config(config_dict):
    simulations = config_dict.get("simulations") or ()
    if not simulations:
        return "CPU"
    settings = simulations[0].get("settings") or {}
    backend = str(settings.get("solver_backend", "CPU")).strip().upper()
    if backend not in {"GPU", "CPU"}:
        return "CPU"
    if backend == "GPU" and not _gpu_solver_available():
        return "CPU"
    return backend


def _kernel_directory(solver_backend="GPU"):
    kernel_folder = (
        "Kernel_CPU" if str(solver_backend).upper() == "CPU" else "Kernel_GPU"
    )
    if "__file__" in globals():
        return Path(__file__).resolve().parents[2] / "Solver" / kernel_folder
    return (Path.cwd() / "Solver" / kernel_folder).resolve()


def _project_root_directory():
    if "__file__" in globals():
        return Path(__file__).resolve().parents[2]
    return Path.cwd().resolve()


def _resolve_python_executable():
    python_executable = Path(
        ContinuumFlowEnvironmentModule.solver_python_executable(__file__)
    )
    if python_executable.exists():
        return str(python_executable)
    raise FileNotFoundError(
        "Solver environment not found. Open the Continuum Flow add-on preferences "
        "and run 'Install Solver Environment' first."
    )


def _run_kernel(config_dict):
    python_executable = _resolve_python_executable()
    solver_backend = _solver_backend_from_config(config_dict)
    kernel_dir = _kernel_directory(solver_backend)
    project_root = _project_root_directory()
    kernel_module = "Solver.General.main"
    bootstrap_code = "\n".join(
        (
            "import json, sys",
            "sys.path.insert(0, sys.argv[1])",
            f"import {kernel_module} as KernelMain",
            "config = json.load(sys.stdin)",
            "KernelMain.main(config)",
        )
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
    try:
        process.stdin.write(json.dumps(config_dict))
        process.stdin.close()
    except BrokenPipeError as exc:
        kernel_output = ""
        if process.stdout is not None:
            try:
                kernel_output = process.stdout.read().strip()
            except Exception:
                kernel_output = ""
        process.wait(timeout=5.0)
        detail = f" Kernel startup output:\n{kernel_output}" if kernel_output else ""
        raise RuntimeError(
            f"Kernel process exited before accepting the bake config.{detail}"
        ) from exc
    return process


def _handle_kernel_output_line(line, output_tail):
    output_tail.append(line.rstrip())
    if len(output_tail) > 40:
        del output_tail[0]
    if line.startswith(PROGRESS_EVENT_PREFIX):
        try:
            payload = json.loads(line[len(PROGRESS_EVENT_PREFIX) :])
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



def _debug_config_export_path_from_config(config_dict):
    simulations = config_dict.get("simulations") or ()
    if not simulations:
        return None
    outputs = simulations[0].get("outputs") or ()
    if not outputs:
        return None
    output_dir = (outputs[0].get("output_path") or "").strip()
    if not output_dir:
        return None
    node_tree_name = ((config_dict.get("meta") or {}).get("node_tree_name") or "").strip()
    if not node_tree_name:
        node_tree_name = "continuum_flow"
    return Path(output_dir) / f"{node_tree_name}_config_export.json"


def _debug_enabled_from_config(config_dict):
    simulations = config_dict.get("simulations") or ()
    if not simulations:
        return False
    viewers = simulations[0].get("viewers") or ()
    return any(bool(viewer_cfg.get("debug", False)) for viewer_cfg in viewers)


def _write_debug_config_snapshot(config_dict):
    file_path = _debug_config_export_path_from_config(config_dict)
    if file_path is None:
        return None
    export_config = {
        key: deepcopy(value)
        for key, value in config_dict.items()
        if not str(key).startswith("_")
    }
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(export_config, indent=2), encoding="utf-8")
    return file_path


def _writer_config_export_path_from_config(config_dict):
    simulations = config_dict.get("simulations") or ()
    if not simulations:
        return None
    outputs = simulations[0].get("outputs") or ()
    if not outputs:
        return None
    output_dir = (outputs[0].get("output_path") or "").strip()
    if not output_dir:
        return None
    return Path(output_dir) / "config.json"


def _write_writer_config_snapshot(config_dict):
    file_path = _writer_config_export_path_from_config(config_dict)
    if file_path is None:
        return None
    export_config = {
        key: deepcopy(value)
        for key, value in config_dict.items()
        if not str(key).startswith("_")
    }
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(export_config, indent=2), encoding="utf-8")
    return file_path


def _read_kernel_output(process, output_queue):
    try:
        if process.stdout is not None:
            for line in process.stdout:
                output_queue.put(("line", line))
    except Exception as exc:
        output_queue.put(("error", str(exc)))


def _start_bake_session(config_dict):
    writer_config_path = _write_writer_config_snapshot(config_dict)
    writer_server = ContinuumFlowHostWriterModule.HostVDBWriterServer(
        writer_process_count=_writer_process_count_from_config(config_dict),
        config_path=writer_config_path,
    )
    writer_server.start()
    config_dict["simulations"][0]["outputs"][0]["host_vdb_writer"] = writer_server.endpoint()
    output_queue = queue.Queue()
    try:
        if _debug_enabled_from_config(config_dict):
            _write_debug_config_snapshot(config_dict)
        process = _run_kernel(config_dict)
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
        "output_queue": output_queue,
        "output_thread": output_thread,
    }


class ContinuumFlow_OT_bake(bpy.types.Operator):
    """
    Operator that exports the active node tree config and starts the kernel.
    """

    bl_idname = "continuum_flow.bake"
    bl_label = "Bake Continuum Flow"
    bl_description = "Start the Continuum Flow bake process"
    output_path_hint: StringProperty(default="", options={"SKIP_SAVE"})  # type: ignore

    _timer = None
    _session = None
    _config_dict = None
    _output_tail = None
    _progress_percent = 0.0
    _progress_factor = 0.0
    _live_preview_state = None
    _live_preview_enabled = False
    _live_preview_interval = None
    _next_live_preview_sync_at = 0.0
    _live_preview_warning_reported = False
    _reader_error = None
    _solver_divergence_message = None
    _window_manager = None
    _workspace = None
    _cancel_requested = False
    _bake_started_at = None

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
                    _set_status_progress_values(
                        percent, self._progress_eta_text(percent)
                    )
                divergence_message = _solver_divergence_message_from_line(payload)
                if divergence_message is not None:
                    self._solver_divergence_message = divergence_message
            elif event_type == "error":
                self._reader_error = payload

    def _progress_eta_text(self, percent):
        if self._bake_started_at is None or percent <= 0.0 or percent >= 100.0:
            return ""
        elapsed_seconds = perf_counter() - self._bake_started_at
        if elapsed_seconds < _STATUS_PROGRESS_ETA_DELAY_SECONDS:
            return ""
        remaining_seconds = elapsed_seconds * (100.0 - percent) / percent
        return _format_eta_seconds(remaining_seconds)

    def _set_status_progress(self, context):
        if self._workspace is None:
            self._workspace = context.workspace
        self._workspace.status_text_set(_draw_continuum_flow_status_progress)
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
        self._bake_started_at = None
        self._live_preview_state = None
        self._live_preview_enabled = False
        self._live_preview_interval = None
        self._next_live_preview_sync_at = 0.0
        self._live_preview_warning_reported = False
        _set_status_progress_values(0.0, "")
        self._clear_status_progress(context)
        self._window_manager = None
        if self._session is not None:
            process = self._session.get("process")
            output_thread = self._session.get("output_thread")
            writer_server = self._session.get("writer_server")

            if process is not None:
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass

                for stream_name in ("stdin", "stdout", "stderr"):
                    stream = getattr(process, stream_name, None)
                    if stream is None:
                        continue
                    try:
                        stream.close()
                    except OSError:
                        pass

            if output_thread is not None and output_thread.is_alive():
                output_thread.join(timeout=0.5)

            if writer_server is not None:
                writer_server.stop()
            self._session = None
        BakeStorage._clear_bake_cancel_flag(self._config_dict)
        BakeStorage._cleanup_geometry_cache(self._config_dict)
        self._config_dict = None
        self._output_tail = None
        self._reader_error = None
        self._solver_divergence_message = None
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
        cancel_flag_path = (
            ((self._config_dict.get("meta") or {}).get("cancel_flag_path")) or ""
        ).strip()
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
        if (
            return_code is None
            and self._live_preview_enabled
            and self._live_preview_interval is not None
            and perf_counter() >= self._next_live_preview_sync_at
        ):
            try:
                BakeStorage._refresh_live_preview(
                    self._config_dict,
                    context=context,
                    state=self._live_preview_state,
                )
            except Exception as exc:
                if not self._live_preview_warning_reported:
                    self.report(
                        {"WARNING"},
                        f"Live preview was disabled after a refresh failed: {exc}",
                    )
                    self._live_preview_warning_reported = True
                self._live_preview_enabled = False
            finally:
                self._next_live_preview_sync_at = perf_counter() + float(
                    self._live_preview_interval
                )
        if return_code is None:
            return {"RUNNING_MODAL"}
        self._drain_kernel_output()
        reader_error = self._reader_error
        config_dict = self._config_dict
        solver_divergence_message = self._solver_divergence_message
        cancel_requested = self._cancel_requested
        self._cleanup_bake(context)
        if reader_error:
            self.report(
                {"ERROR"}, f"Bake failed while reading kernel output: {reader_error}"
            )
            return {"CANCELLED"}
        if return_code != 0:
            self.report(
                {"ERROR"},
                f"Bake failed: Kernel process failed with exit code {return_code}{self._recent_kernel_output_detail()}",
            )
            return {"CANCELLED"}
        try:
            imported_count, missing_directories = BakeStorage._import_baked_vdbs(
                config_dict
            )
        except Exception as exc:
            if solver_divergence_message is not None:
                self.report(
                    {"WARNING"}, f"{solver_divergence_message} VDB import failed: {exc}"
                )
            else:
                self.report({"WARNING"}, f"Bake finished, but VDB import failed: {exc}")
            return {"FINISHED"}
        if solver_divergence_message is not None:
            if imported_count > 0:
                self.report(
                    {"WARNING"},
                    f"{solver_divergence_message} Imported {imported_count} VDB volume(s) up to that point.",
                )
            elif missing_directories:
                self.report(
                    {"WARNING"},
                    f"{solver_divergence_message} No VDB output directory was found.",
                )
            else:
                self.report(
                    {"WARNING"},
                    f"{solver_divergence_message} No VDB files were found to import.",
                )
            return {"FINISHED"}
        if cancel_requested:
            if imported_count > 0:
                self.report(
                    {"WARNING"},
                    f"Continuum Flow bake cancelled. Imported {imported_count} VDB volume(s) up to that point.",
                )
            elif missing_directories:
                self.report(
                    {"WARNING"},
                    "Continuum Flow bake cancelled. No VDB output directory was found.",
                )
            else:
                self.report(
                    {"WARNING"},
                    "Continuum Flow bake cancelled. No VDB files were found to import.",
                )
            return {"FINISHED"}
        if imported_count > 0:
            self.report(
                {"INFO"}, f"Bake finished. Imported {imported_count} VDB volume(s)."
            )
        elif missing_directories:
            self.report(
                {"WARNING"}, "Bake finished, but no VDB output directory was found."
            )
        else:
            self.report(
                {"WARNING"}, "Bake finished, but no VDB files were found to import."
            )
        return {"FINISHED"}

    def execute(self, context):
        hinted_output_directory = BakeStorage._resolve_output_directory_hint(
            self.output_path_hint
        )
        if (
            hinted_output_directory is not None
            and BakeStorage._output_directory_has_vdbs(hinted_output_directory)
        ):
            removed_volumes, deleted_files = BakeStorage._free_bake(
                context,
                output_directories=(hinted_output_directory,),
            )
            self.report(
                {"INFO"},
                f"Freed bake data. Removed {removed_volumes} volume object(s) and deleted {deleted_files} VDB file(s).",
            )
            return {"FINISHED"}

        resolve_node_tree = getattr(ContinuumFlowConfigModule, "_resolve_node_tree", None)
        node_tree = resolve_node_tree(context) if callable(resolve_node_tree) else None
        if tree_has_invalid_links(node_tree):
            self.report(
                {"ERROR"}, "Bake disabled: invalid socket connection in the node tree."
            )
            return {"CANCELLED"}

        global _ACTIVE_BAKE_OPERATOR
        bake_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        geometry_output_directory = hinted_output_directory or _project_root_directory()
        geometry_cache_dir = BakeStorage._bake_geometry_cache_directory(
            geometry_output_directory, bake_token
        )
        try:
            live_config_dict = ContinuumFlowConfigModule.build_config_dict(
                context,
                geometry_storage_dir=str(geometry_cache_dir),
            )
            live_config_dict.setdefault("meta", {})["geometry_cache_dir"] = str(
                geometry_cache_dir
            )
        except Exception as exc:
            shutil.rmtree(geometry_cache_dir, ignore_errors=True)
            self.report({"ERROR"}, f"Bake failed: {exc}")
            return {"CANCELLED"}
        if BakeStorage._config_has_baked_vdbs(live_config_dict):
            active_operator = _ACTIVE_BAKE_OPERATOR
            if active_operator is not None and active_operator._session is not None:
                BakeStorage._cleanup_geometry_cache(live_config_dict)
                active_operator._request_cancel()
                self.report(
                    {"INFO"},
                    "Bake cancellation requested. Press Free Bake again after the bake stops.",
                )
                return {"FINISHED"}
            removed_volumes, deleted_files = BakeStorage._free_bake(
                context, config_dict=live_config_dict
            )
            BakeStorage._cleanup_geometry_cache(live_config_dict)
            self.report(
                {"INFO"},
                f"Freed bake data. Removed {removed_volumes} volume object(s) and deleted {deleted_files} VDB file(s).",
            )
            return {"FINISHED"}
        try:
            self._config_dict = BakeStorage._prepare_config_for_bake(
                live_config_dict, bake_token=bake_token
            )
            self._session = _start_bake_session(self._config_dict)
        except ModuleNotFoundError as exc:
            BakeStorage._cleanup_geometry_cache(live_config_dict)
            set_bake_running(False, context=context)
            missing_module = getattr(exc, "name", None) or str(exc)
            self.report(
                {"ERROR"},
                f"Bake failed: missing Python module '{missing_module}' in {sys.executable}",
            )
            return {"CANCELLED"}
        except FileNotFoundError as exc:
            BakeStorage._cleanup_geometry_cache(live_config_dict)
            set_bake_running(False, context=context)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        except Exception as exc:
            BakeStorage._cleanup_geometry_cache(live_config_dict)
            set_bake_running(False, context=context)
            self.report({"ERROR"}, f"Bake failed: {exc}")
            return {"CANCELLED"}
        BakeStorage._set_bake_state_active(
            True, context=context, config_dict=self._config_dict
        )
        set_bake_running(True, context=context)
        self._output_tail = []
        self._progress_percent = 0.0
        self._progress_factor = 0.0
        _set_status_progress_values(0.0, "")
        self._reader_error = None
        self._solver_divergence_message = None
        self._window_manager = context.window_manager
        self._workspace = context.workspace
        self._cancel_requested = False
        self._bake_started_at = perf_counter()
        self._live_preview_state = {}
        self._live_preview_interval = BakeStorage._live_preview_refresh_interval(
            self._config_dict
        )
        self._live_preview_enabled = self._live_preview_interval is not None
        self._next_live_preview_sync_at = perf_counter()
        self._live_preview_warning_reported = False
        _ACTIVE_BAKE_OPERATOR = self
        self._set_status_progress(context)
        timer_interval = 0.1
        if self._live_preview_interval is not None:
            timer_interval = min(timer_interval, float(self._live_preview_interval))
        self._timer = context.window_manager.event_timer_add(
            timer_interval, window=context.window
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}


class ContinuumFlow_OT_cancel_bake(bpy.types.Operator):
    """
    Cancel the currently running Continuum Flow bake from the status bar.
    """

    bl_idname = "continuum_flow.cancel_bake"
    bl_label = "Cancel Continuum Flow Bake"
    bl_description = "Cancel the currently running Continuum Flow bake"

    def execute(self, context):
        active_operator = _ACTIVE_BAKE_OPERATOR
        if active_operator is None or active_operator._session is None:
            self.report({"WARNING"}, "No Continuum Flow bake is currently running.")
            return {"CANCELLED"}
        active_operator._request_cancel()
        active_operator._tag_status_bar_redraw(context)
        self.report({"INFO"}, "Bake cancellation requested.")
        return {"FINISHED"}


classes = (
    ContinuumFlow_OT_bake,
    ContinuumFlow_OT_cancel_bake,
)

