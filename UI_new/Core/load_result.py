import bpy
from pathlib import Path


class VDBWatcher:
    def __init__(self):
        self.watch_dir = None
        self.file_sizes = {}
        self.running = False
        self.volume_object = None
        self.sequence_files = []

    def start(self, watch_dir):
        self.clear_loaded_sequence()
        self.watch_dir = Path(watch_dir).resolve()
        self.file_sizes.clear()
        self.sequence_files.clear()
        self.running = True

        bpy.app.timers.register(self.timer, first_interval=0.5)

        print("VDB watcher started:", self.watch_dir)

    def stop(self):
        self.running = False
        print("VDB watcher stopping")

    def clear_loaded_sequence(self):
        volume_object = self.volume_object
        self.volume_object = None
        self.sequence_files.clear()
        self.file_sizes.clear()

        if volume_object is None:
            return

        volume_data = getattr(volume_object, "data", None)

        try:
            bpy.data.objects.remove(volume_object, do_unlink=True)
        except Exception as exc:
            print("Failed to remove VDB object:", exc)
            return

        try:
            if volume_data is not None and getattr(volume_data, "users", 1) == 0:
                bpy.data.volumes.remove(volume_data)
        except Exception as exc:
            print("Failed to remove VDB volume data:", exc)

        print("Removed loaded VDB sequence from Blender")

    def _stable_vdbs(self):
        stable = []

        for path in sorted(self.watch_dir.glob("*.vdb")):
            try:
                size = path.stat().st_size
            except OSError:
                continue

            old_size = self.file_sizes.get(path)
            self.file_sizes[path] = size

            if old_size == size:
                stable.append(path)

        return stable

    def _create_sequence(self, first_vdb, stable_vdbs):
        bpy.ops.object.volume_import(filepath=str(first_vdb))

        self.volume_object = bpy.context.object
        volume = self.volume_object.data

        volume.filepath = str(first_vdb)
        volume.is_sequence = True
        volume.frame_start = 1
        volume.frame_offset = 0
        volume.frame_duration = len(stable_vdbs)
        volume.sequence_mode = 'CLIP'

        try:
            volume.reload()
        except Exception as exc:
            print("Initial volume reload failed:", exc)

        bpy.context.scene.frame_set(len(stable_vdbs))

        print("Created VDB sequence:", first_vdb)

    def _refresh_sequence(self, stable_vdbs):
        volume = self.volume_object.data
        new_duration = len(stable_vdbs)

        if volume.frame_duration == new_duration:
            return

        volume.frame_duration = new_duration

        try:
            volume.reload()
        except Exception as exc:
            print("Volume reload failed:", exc)

        bpy.context.scene.frame_set(new_duration)

        print("Updated VDB sequence duration:", new_duration)

    def timer(self):
        if not self.running:
            return None

        fps = bpy.context.scene.render.fps or 24
        interval = 1.0 / fps

        if not self.watch_dir or not self.watch_dir.exists():
            return interval

        stable_vdbs = self._stable_vdbs()

        if not stable_vdbs:
            return interval

        if self.volume_object is None:
            self._create_sequence(stable_vdbs[0], stable_vdbs)
        else:
            self._refresh_sequence(stable_vdbs)

        self.sequence_files = stable_vdbs

        return interval
