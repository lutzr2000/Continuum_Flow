import bpy
import json
import shutil
import subprocess
import threading
from pathlib import Path

from . import export_config
from . import writer_manager
from . import load_result


def _venv_python_path():
    addon_root = Path(__file__).resolve().parents[2]
    return addon_root / "ContinuumFlow_env" / "Scripts" / "python.exe"


def _read_solver_stdout(process):
    for line in process.stdout:
        print("[Solver]", line, end="")


_vdb_watcher = load_result.VDBWatcher()


class main(bpy.types.Operator):
    bl_idname = "continuum_flow.bake"
    bl_label = "Bake"

    def execute(self, context):
        self.process = None
        self.writer_server = None
        self.config_path = None
        self._cleanup_done = False
        self._cleanup_lock = threading.Lock()

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

                try:
                    shutil.rmtree(self.config_path.parent / "geometry", ignore_errors=True)
                except OSError:
                    pass

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

    def _wait_solver_finished(self):
        exit_code = self.process.wait()
        print("Solver Exit Code:", exit_code)
        self.cleanup()

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
        vdb_output_dir = Path(output_config["output_path"]).resolve()

        _vdb_watcher.start(vdb_output_dir)

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

        threading.Thread(
            target=self._wait_solver_finished,
            daemon=True,
        ).start()

        print("Solver started:", self.process.pid)