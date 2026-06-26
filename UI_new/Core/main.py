import bpy
import json
import shutil
import subprocess
import threading
from pathlib import Path

from . import export_config
from . import writer_manager
from . import load_result
from . import solver_status


def _venv_python_path():
    addon_root = Path(__file__).resolve().parents[2]
    return addon_root / "ContinuumFlow_env" / "Scripts" / "python.exe"


def _read_solver_stdout(process):
    for line in process.stdout:
        print("[Solver]", line, end="")


_vdb_watcher = load_result.VDBWatcher()


def draw_bake_progress(self, context):
    if not solver_status.bake_running:
        return

    window_manager = getattr(context, "window_manager", None)
    if window_manager is None:
        return

    row = self.layout.row(align=True)

    # Linker flexibler Spacer
    row.separator_spacer()

    center = row.row(align=True)
    center.alignment = 'CENTER'

    center.label(text="Bake:")

    progress_row = center.row(align=True)
    progress_row.ui_units_x = 13

    if hasattr(progress_row, "progress"):
        progress_row.progress(
            factor=float(solver_status.progress),
            type='BAR',
            text=solver_status.progress_text,
        )
    elif hasattr(window_manager, "continuum_flow_bake_progress"):
        fallback_row = progress_row.row(align=True)
        fallback_row.enabled = False
        fallback_row.prop(
            window_manager,
            "continuum_flow_bake_progress",
            text=solver_status.progress_text,
            slider=True,
        )

    # Rechter flexibler Spacer
    row.separator_spacer()

    # Cancel-Button rechts außen
    cancel_row = row.row(align=True)
    cancel_row.operator(
        "continuum_flow.cancel_bake",
        text="",
        icon="PANEL_CLOSE",
        emboss=False,
    )


def _tag_ui_redraw():
    window_manager = getattr(bpy.context, "window_manager", None)
    if window_manager is None:
        return

    for window in window_manager.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue

        for area in screen.areas:
            area.tag_redraw()


def _set_bake_progress(current_frames, total_frames):
    current_frames = max(0, int(current_frames or 0))
    total_frames = max(0, int(total_frames or 0))

    solver_status.progress_current_frames = current_frames
    solver_status.progress_total_frames = total_frames

    if total_frames > 0:
        progress = min(1.0, float(current_frames) / float(total_frames))
        percent = int(round(progress * 100.0))
        solver_status.progress = progress
        solver_status.progress_text = f"{percent}%"
    else:
        solver_status.progress = 0.0
        solver_status.progress_text = "0%"

    window_manager = getattr(bpy.context, "window_manager", None)
    if window_manager is None:
        return

    if hasattr(window_manager, "continuum_flow_bake_progress"):
        window_manager.continuum_flow_bake_progress = float(solver_status.progress)
    _tag_ui_redraw()


def _set_bake_available_state(is_available, output_directory=None):
    solver_status.bake_available = bool(is_available)
    solver_status.last_output_directory = str(output_directory) if output_directory else None
    _tag_ui_redraw()


def _update_bake_available_from_output(output_directory):
    if output_directory is None:
        _set_bake_available_state(False)
        return False

    output_directory = Path(output_directory).resolve()
    has_vdbs = output_directory.exists() and any(output_directory.glob("*.vdb"))
    _set_bake_available_state(has_vdbs, output_directory if has_vdbs else None)
    return has_vdbs


def _clear_baked_vdbs(output_directory):
    if output_directory is None:
        return 0

    output_directory = Path(output_directory).resolve()
    if not output_directory.exists() or not output_directory.is_dir():
        return 0

    deleted_count = 0
    for vdb_path in output_directory.glob("*.vdb"):
        try:
            vdb_path.unlink()
            deleted_count += 1
        except OSError as exc:
            print("Failed to remove VDB:", vdb_path, exc)

    print("Removed VDB files:", deleted_count)
    return deleted_count


class CONTINUUM_FLOW_OT_cancel_bake(bpy.types.Operator):
    bl_idname = "continuum_flow.cancel_bake"
    bl_label = "Cancel Bake"

    def execute(self, context):
        active_operator = solver_status.active_bake_operator
        if not solver_status.bake_running or active_operator is None:
            self.report({'WARNING'}, "No bake is currently running")
            return {'CANCELLED'}

        print("Bake cancelled by user")
        active_operator.cancel_bake()
        return {'FINISHED'}


class main(bpy.types.Operator):
    bl_idname = "continuum_flow.bake"
    bl_label = "Bake"

    def execute(self, context):
        if solver_status.bake_running:
            self.report({'WARNING'}, "Bake is already running")
            return {'CANCELLED'}

        if solver_status.bake_available:
            deleted_count = self.free_bake()
            if deleted_count:
                self.report({'INFO'}, f"Removed {deleted_count} VDB files")
            else:
                self.report({'INFO'}, "No VDB files found to remove")
            return {'FINISHED'}

        self.process = None
        self.writer_server = None
        self.config_path = None
        self.output_directory = None
        self._cleanup_done = False
        self._cleanup_lock = threading.Lock()

        solver_status.bake_running = True
        solver_status.active_bake_operator = self
        _tag_ui_redraw()

        self.do_bake(context)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            print("Bake cancelled by user")
            self.cancel_bake()
            return {'CANCELLED'}

        if self.process and self.process.poll() is not None:
            self.cleanup()
            return {'FINISHED'}

        return {'PASS_THROUGH'}

    def cleanup(self):
        with self._cleanup_lock:
            if self._cleanup_done:
                return

            self._cleanup_done = True
            solver_status.bake_running = False
            if solver_status.active_bake_operator is self:
                solver_status.active_bake_operator = None

            _vdb_watcher.stop()

            if self.writer_server:
                try:
                    self.writer_server.stop()
                except Exception as exc:
                    print("Failed to stop writer server:", exc)

            if self.config_path:
                try:
                    self.config_path.unlink(missing_ok=True)
                except OSError:
                    pass

                geometry_dir = self.config_path.parent / "geometry"
                try:
                    shutil.rmtree(geometry_dir, ignore_errors=True)
                except OSError:
                    pass

            _update_bake_available_from_output(self.output_directory)
            _set_bake_progress(0, 0)
            print("Bake cleanup finished")

    def cancel_bake(self):
        _vdb_watcher.stop()

        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as exc:
                print("Failed to terminate solver:", exc)

        self.cleanup()

    def free_bake(self):
        output_directory = solver_status.last_output_directory
        _vdb_watcher.clear_loaded_sequence()
        deleted_count = _clear_baked_vdbs(output_directory)
        _update_bake_available_from_output(output_directory)
        return deleted_count

    def _update_progress_from_loaded_frames(self, loaded_frame_count):
        _set_bake_progress(loaded_frame_count, solver_status.progress_total_frames)

    def launch_writer_manager(self, config_dict):
        output_config = ((config_dict.get("simulations") or [{}])[0].get("outputs") or [{}])[0]
        performance_config = output_config.get("performance") or {}
        writer_process_count = int(performance_config.get("writer_processes", 4))
        config_path = output_config.get("_writer_config_path")

        server = writer_manager.HostVDBWriterServer(
            writer_process_count=writer_process_count,
            config_path=config_path,
        )
        server.start()
        return server

    def do_bake(self, context):
        config_dict = export_config.build_config_dict()
        config_path, config_dict = export_config.export_config_dict(config_dict)
        config_dict["simulations"][0]["outputs"][0]["_writer_config_path"] = str(config_path)

        writer_server = self.launch_writer_manager(config_dict)
        config_dict["simulations"][0]["outputs"][0]["host_vdb_writer"] = writer_server.endpoint()
        config_dict["simulations"][0]["outputs"][0].pop("_writer_config_path", None)
        config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

        self.config_path = config_path
        self.writer_server = writer_server

        output_config = config_dict["simulations"][0]["outputs"][0]
        simulation_settings = config_dict["simulations"][0].get("settings") or {}
        start_frame = int(simulation_settings.get("start_frame", 1))
        end_frame = int(simulation_settings.get("end_frame", start_frame))
        total_frames = max(0, (end_frame - start_frame) + 1)
        _set_bake_progress(0, total_frames)

        vdb_output_dir = Path(output_config["output_path"]).resolve()
        self.output_directory = vdb_output_dir

        _vdb_watcher.start(
            vdb_output_dir,
            progress_callback=self._update_progress_from_loaded_frames,
        )

        addon_root = Path(__file__).resolve().parents[2]

        self.process = subprocess.Popen(
            [
                str(_venv_python_path()),
                "-m",
                "Solver.General.main",
                str(config_path),
            ],
            cwd=str(addon_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        threading.Thread(
            target=_read_solver_stdout,
            args=(self.process,),
            daemon=True,
        ).start()

        print("Solver started:", self.process.pid)

