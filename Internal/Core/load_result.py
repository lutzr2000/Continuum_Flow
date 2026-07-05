import re
from pathlib import Path

import bpy


_OUTPUT_DIRECTORY_PROPERTY = "continuum_flow_output_directory"
_FRAME_FILE_PATTERN = re.compile(r"^frame_(\d+)\.vdb$")


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
        self.running = False
        self.volume_object = None
        self.sequence_files = []
        self.progress_callback = None
        self.loaded_output_directory = None
        self.live_preview_enabled = False
        self.start_frame_index = 1

    def start(self, watch_dir, start_frame_index=1, live_preview_enabled=False, progress_callback=None):
        self.watch_dir = Path(watch_dir).resolve()
        self.volume_object = None
        self.loaded_output_directory = None
        self.sequence_files.clear()
        self.progress_callback = progress_callback
        self.live_preview_enabled = bool(live_preview_enabled)
        self.start_frame_index = int(start_frame_index)
        self.running = True

        bpy.app.timers.register(self.timer, first_interval=0.5)

    def stop(self):
        self.running = False
        self.progress_callback = None

    def finish_bake(self):
        ordered_vdbs = self._ordered_vdbs()
        self.sequence_files = ordered_vdbs

        if ordered_vdbs:
            self._load_full_sequence(ordered_vdbs)

        if self.progress_callback is not None:
            self.progress_callback(len(ordered_vdbs))

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
            self.loaded_output_directory = None

    def _volume_object_is_valid(self):
        volume_object = self.volume_object
        if volume_object is None:
            return False

        try:
            volume_data = getattr(volume_object, "data", None)
            return volume_data is not None
        except Exception:
            return False

    def _frame_index_from_path(self, path):
        match = _FRAME_FILE_PATTERN.match(path.name)
        if match is None:
            return None
        return int(match.group(1))

    def _ordered_vdbs(self):
        ordered = []

        for path in self.watch_dir.glob("*.vdb"):
            frame_index = self._frame_index_from_path(path)
            if frame_index is None:
                continue
            ordered.append((frame_index, path))

        ordered.sort(key=lambda item: item[0])
        return [path for _frame_index, path in ordered]

    def _contiguous_vdbs(self, ordered_vdbs):
        contiguous = []
        expected_index = self.start_frame_index

        for path in ordered_vdbs:
            frame_index = self._frame_index_from_path(path)
            if frame_index is None:
                continue
            if frame_index < expected_index:
                continue
            if frame_index != expected_index:
                break
            contiguous.append(path)
            expected_index += 1

        return contiguous

    def _tag_loaded_volume(self, volume_object):
        if volume_object is None:
            return

        self.loaded_output_directory = str(self.watch_dir)
        volume_object.location = (0.0, 0.0, 0.0)
        volume_object[_OUTPUT_DIRECTORY_PROPERTY] = self.loaded_output_directory

        volume_data = getattr(volume_object, "data", None)
        if volume_data is not None:
            volume_data[_OUTPUT_DIRECTORY_PROPERTY] = self.loaded_output_directory

    def _import_volume_object(self, filepath):
        bpy.ops.object.volume_import(filepath=str(filepath))
        self.volume_object = bpy.context.object
        self._tag_loaded_volume(self.volume_object)
        return getattr(self.volume_object, "data", None)

    def _ensure_volume_data(self, filepath):
        if not self._volume_object_is_valid():
            return self._import_volume_object(filepath)

        self._tag_loaded_volume(self.volume_object)
        return getattr(self.volume_object, "data", None)

    def _load_full_sequence(self, ordered_vdbs):
        first_vdb = ordered_vdbs[0]
        volume = self._ensure_volume_data(first_vdb)
        if volume is None:
            return

        volume.filepath = str(first_vdb)
        volume.is_sequence = True
        volume.frame_start = 1
        volume.frame_offset = 0
        volume.frame_duration = self.start_frame_index + len(ordered_vdbs)
        volume.sequence_mode = 'CLIP'

        last_frame_index = self._frame_index_from_path(ordered_vdbs[-1])
        if last_frame_index is not None:
            bpy.context.scene.frame_set(last_frame_index)

    def timer(self):
        if not self.running:
            return None

        fps = bpy.context.scene.render.fps or 24
        interval = 1.0 / fps

        if not self.watch_dir or not self.watch_dir.exists():
            return interval

        ordered_vdbs = self._ordered_vdbs()
        contiguous_vdbs = self._contiguous_vdbs(ordered_vdbs)
        self.sequence_files = contiguous_vdbs

        if contiguous_vdbs:
            latest_vdb = contiguous_vdbs[-1]
            if self.live_preview_enabled:
                needs_reload = (
                    not self._volume_object_is_valid()
                    or getattr(self.volume_object.data, "filepath", "") != str(contiguous_vdbs[0])
                    or not bool(getattr(self.volume_object.data, "is_sequence", False))
                    or int(getattr(self.volume_object.data, "frame_duration", 0)) != len(contiguous_vdbs)
                )
                if needs_reload:
                    self._load_full_sequence(contiguous_vdbs)
                else:
                    bpy.context.scene.frame_set(self._frame_index_from_path(latest_vdb) or self.start_frame_index)
            else:
                bpy.context.scene.frame_set(self._frame_index_from_path(latest_vdb) or self.start_frame_index)

        if self.progress_callback is not None:
            self.progress_callback(len(contiguous_vdbs))

        return interval
