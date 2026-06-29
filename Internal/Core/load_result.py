import bpy
from pathlib import Path


_OUTPUT_DIRECTORY_PROPERTY = "continuum_flow_output_directory"


def _normalize_directory_path(path_value):
    if not path_value:
        return None

    try:
        return str(Path(path_value).resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


class VDBWatcher:
    def __init__(self):
        self.watch_dir = None
        self.file_sizes = {}
        self.running = False
        self.volume_object = None
        self.sequence_files = []
        self.progress_callback = None
        self.loaded_output_directory = None

    def start(self, watch_dir, progress_callback=None):
        self.watch_dir = Path(watch_dir).resolve()
        self.volume_object = None
        self.loaded_output_directory = None
        self.file_sizes.clear()
        self.sequence_files.clear()
        self.progress_callback = progress_callback
        self.running = True

        bpy.app.timers.register(self.timer, first_interval=0.5)

    def stop(self):
        self.running = False
        self.progress_callback = None

    def _volume_object_matches_directory(self, volume_object, output_directory):
        if volume_object is None or output_directory is None:
            return False

        try:
            tagged_directory = volume_object.get(_OUTPUT_DIRECTORY_PROPERTY)
        except Exception:
            tagged_directory = None

        if _normalize_directory_path(tagged_directory) == output_directory:
            return True

        try:
            volume_data = getattr(volume_object, "data", None)
        except Exception:
            volume_data = None

        if volume_data is not None:
            try:
                tagged_directory = volume_data.get(_OUTPUT_DIRECTORY_PROPERTY)
            except Exception:
                tagged_directory = None

            if _normalize_directory_path(tagged_directory) == output_directory:
                return True

            try:
                filepath = getattr(volume_data, "filepath", "")
            except Exception:
                filepath = ""

            normalized_filepath = _normalize_directory_path(filepath)
            if normalized_filepath is not None:
                try:
                    if str(Path(normalized_filepath).parent) == output_directory:
                        return True
                except (OSError, RuntimeError, TypeError, ValueError):
                    pass

        return False

    def _find_loaded_sequence_objects(self, output_directory):
        if output_directory is None:
            return []

        matches = []
        for volume_object in bpy.data.objects:
            if getattr(volume_object, "type", "") != 'VOLUME':
                continue
            if self._volume_object_matches_directory(volume_object, output_directory):
                matches.append(volume_object)

        return matches

    def _remove_volume_object(self, volume_object):
        if volume_object is None:
            return

        try:
            volume_data = getattr(volume_object, "data", None)
        except Exception:
            return

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

    def clear_loaded_sequence(self):
        volume_object = self.volume_object
        self.volume_object = None
        self.sequence_files.clear()
        self.file_sizes.clear()
        self.loaded_output_directory = None

        if volume_object is not None:
            self._remove_volume_object(volume_object)

    def clear_loaded_sequence_for_directory(self, output_directory):
        normalized_output_directory = _normalize_directory_path(output_directory)
        if normalized_output_directory is None:
            return

        matching_objects = self._find_loaded_sequence_objects(normalized_output_directory)
        for volume_object in matching_objects:
            if volume_object is self.volume_object:
                self.volume_object = None
            self._remove_volume_object(volume_object)

        if self.loaded_output_directory == normalized_output_directory:
            self.volume_object = None
            self.sequence_files.clear()
            self.file_sizes.clear()
            self.loaded_output_directory = None

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
        self.loaded_output_directory = str(self.watch_dir)
        if self.volume_object is not None:
            self.volume_object.location = (0.0, 0.0, 0.0)
            self.volume_object[_OUTPUT_DIRECTORY_PROPERTY] = self.loaded_output_directory

        volume = self.volume_object.data
        volume[_OUTPUT_DIRECTORY_PROPERTY] = self.loaded_output_directory
        volume.filepath = str(first_vdb)
        volume.is_sequence = True
        volume.frame_start = 1
        volume.frame_offset = 0
        volume.frame_duration = len(stable_vdbs)
        volume.sequence_mode = 'CLIP'

        bpy.context.scene.frame_set(len(stable_vdbs))

    def _refresh_sequence(self, stable_vdbs):
        volume = self.volume_object.data
        new_duration = len(stable_vdbs)

        if volume.frame_duration == new_duration:
            return

        volume.frame_duration = new_duration

        bpy.context.scene.frame_set(new_duration)

    def timer(self):
        if not self.running:
            return None

        fps = bpy.context.scene.render.fps or 24
        interval = 1.0 / fps

        if not self.watch_dir or not self.watch_dir.exists():
            return interval

        stable_vdbs = self._stable_vdbs()

        if not stable_vdbs:
            if self.progress_callback is not None:
                self.progress_callback(0)
            return interval

        if self.volume_object is None:
            self._create_sequence(stable_vdbs[0], stable_vdbs)
        else:
            self._refresh_sequence(stable_vdbs)

        self.sequence_files = stable_vdbs

        if self.progress_callback is not None:
            self.progress_callback(len(self.sequence_files))

        return interval
