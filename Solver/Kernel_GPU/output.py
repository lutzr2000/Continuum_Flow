import json
import queue
import socket
import threading
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
from numba import cuda


FRAME_SLOT_COUNT = 3
SPARSE_THRESHOLD = 0.0


@cuda.jit(cache=True)
def _cast_field_kernel(source, destination):
    i, j, k = cuda.grid(3)
    if i < source.shape[0] and j < source.shape[1] and k < source.shape[2]:
        destination[i, j, k] = source[i, j, k]


def _output_directory(output_cfg):
    return Path((output_cfg.get("output_path") or "").strip()).resolve()


def _output_dtype(output_cfg):
    precision = str(output_cfg.get("precision", "float16")).strip().lower()
    return np.float32 if precision == "float32" else np.float16


def _enabled_field_specs(output_cfg):
    field_cfg = output_cfg.get("fields") or {}
    specs = []

    if bool((field_cfg.get("velocity") or {}).get("enabled", False)):
        specs.append(
            {
                "name": "velocity",
                "grid_type": "vector",
                "sparse": bool((field_cfg.get("velocity") or {}).get("sparse", False)),
                "components": ("u", "v", "w"),
            }
        )

    for field_name in ("pressure", "temperature", "smoke", "fuel", "flame"):
        cfg = field_cfg.get(field_name) or {}
        if not bool(cfg.get("enabled", False)):
            continue
        specs.append(
            {
                "name": field_name,
                "grid_type": "scalar",
                "sparse": bool(cfg.get("sparse", False)),
                "components": (field_name,),
            }
        )

    return specs


def _component_sources(field_state):
    return {
        "u": field_state["u"],
        "v": field_state["v"],
        "w": field_state["w"],
        "pressure": field_state["pressure"],
        "temperature": field_state["temperature"],
        "smoke": field_state["smoke"],
        "fuel": field_state["fuel"],
        "flame": field_state["flame"],
    }


class _FrameSlot:
    def __init__(self, exporter, slot_index):
        self.exporter = exporter
        self.slot_index = int(slot_index)
        self.shm = shared_memory.SharedMemory(create=True, size=exporter.packed_nbytes)
        self.array = np.ndarray(
            exporter.packed_shape,
            dtype=exporter.output_dtype,
            buffer=self.shm.buf,
        )

    def payload(self, frame_index):
        return {
            "output_path": str(self.exporter.output_dir / f"{frame_index:04d}.vdb"),
            "packed_shm_name": self.shm.name,
            "packed_shape": list(self.exporter.packed_shape),
            "dtype": self.exporter.output_dtype.str,
            "grid_shape": list(self.exporter.grid_shape),
            "grids": self.exporter.grid_payloads,
            "sparse_mask_component_indices": self.exporter.sparse_mask_component_indices,
            "sparse_threshold": SPARSE_THRESHOLD,
            "transform": self.exporter.transform_payload,
        }

    def close(self):
        try:
            self.shm.close()
        finally:
            self.shm.unlink()


class _WriterSubmitThread(threading.Thread):
    def __init__(self, exporter):
        super().__init__(daemon=True)
        self.exporter = exporter
        self.exception = None

    def _mark_failed(self, exc):
        self.exception = exc
        with self.exporter.slot_condition:
            self.exporter.failed = True
            self.exporter.slot_condition.notify_all()

    def run(self):
        sock = None
        reader = None
        writer = None
        try:
            sock = socket.create_connection(
                (
                    self.exporter.writer_endpoint["host"],
                    int(self.exporter.writer_endpoint["port"]),
                )
            )
            reader = sock.makefile("r", encoding="utf-8", newline="\n")
            writer = sock.makefile("w", encoding="utf-8", newline="\n")

            while True:
                item = self.exporter.submit_queue.get()
                if item is None:
                    self.exporter.submit_queue.task_done()
                    break

                slot, payload = item
                try:
                    writer.write(json.dumps(payload) + "\n")
                    writer.flush()
                    response_line = reader.readline()
                    if not response_line:
                        raise RuntimeError("Writer server closed the connection.")
                    response = json.loads(response_line)
                    if response.get("status") != "ok":
                        raise RuntimeError(response.get("message", "unknown VDB writer error"))
                except Exception as exc:
                    self._mark_failed(exc)
                    self.exporter.submit_queue.task_done()
                    break

                with self.exporter.slot_condition:
                    self.exporter.free_slots.put(slot)
                    self.exporter.slot_condition.notify_all()

                self.exporter.submit_queue.task_done()
        except Exception as exc:
            self._mark_failed(exc)
        finally:
            if writer is not None:
                try:
                    writer.write("__QUIT__\n")
                    writer.flush()
                except Exception:
                    pass
            if reader is not None:
                reader.close()
            if writer is not None:
                writer.close()
            if sock is not None:
                sock.close()


class GPUOutputExporter:
    def __init__(self, config, grid_shape, delta, origin):
        simulations = config.get("simulations") or ()
        output_cfg = ((simulations[0].get("outputs") or [None])[0]) or {}
        writer_endpoint = config.get("_host_vdb_writer") or {}

        self.field_specs = _enabled_field_specs(output_cfg)
        self.writer_endpoint = writer_endpoint
        self.output_dir = _output_directory(output_cfg)
        self.output_dtype = np.dtype(_output_dtype(output_cfg))
        self.grid_shape = tuple(int(v) for v in grid_shape)
        self.delta = float(delta)
        self.origin = tuple(float(v) for v in origin[:3])
        self.enabled = bool(self.field_specs) and bool(writer_endpoint) and bool(
            str(output_cfg.get("output_path") or "").strip()
        )
        self.failed = False
        self.component_sources = None

        self.component_names = []
        self.grid_payloads = []
        self.sparse_mask_component_indices = []
        if not self.enabled:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        component_index = 0
        smoke_index = None
        flame_index = None
        for field_spec in self.field_specs:
            component_indices = []
            for component_name in field_spec["components"]:
                self.component_names.append(component_name)
                component_indices.append(component_index)
                if component_name == "smoke":
                    smoke_index = component_index
                elif component_name == "flame":
                    flame_index = component_index
                component_index += 1
            self.grid_payloads.append(
                {
                    "name": field_spec["name"],
                    "grid_type": field_spec["grid_type"],
                    "sparse": bool(field_spec["sparse"]),
                    "storage_dtype": self.output_dtype.str,
                    "component_indices": component_indices,
                }
            )

        self.component_count = len(self.component_names)
        self.packed_shape = (self.component_count,) + self.grid_shape
        self.packed_nbytes = int(np.prod(self.packed_shape, dtype=np.int64)) * self.output_dtype.itemsize
        self.transform_payload = {
            "voxel_size": self.delta,
            "origin": list(self.origin),
        }
        for maybe_index in (smoke_index, flame_index):
            if maybe_index is not None:
                self.sparse_mask_component_indices.append(int(maybe_index))

        self.blocks_per_grid = (
            (self.grid_shape[0] + 7) // 8,
            (self.grid_shape[1] + 7) // 8,
            (self.grid_shape[2] + 7) // 8,
        )
        self.threads_per_block = (8, 8, 8)
        self.cast_stream = cuda.stream()
        self.cast_buffer = None
        if self.output_dtype == np.dtype(np.float16):
            self.cast_buffer = cuda.device_array(self.grid_shape, dtype=np.float16)

        self.free_slots = queue.Queue(maxsize=FRAME_SLOT_COUNT)
        self.slot_condition = threading.Condition()
        self.slots = [_FrameSlot(self, idx) for idx in range(FRAME_SLOT_COUNT)]
        for slot in self.slots:
            self.free_slots.put(slot)
        self.submit_queue = queue.Queue()
        self.submit_thread = _WriterSubmitThread(self)
        self.submit_thread.start()

    def bind_field_state(self, field_state):
        if self.enabled:
            self.component_sources = _component_sources(field_state)

    def _copy_component_to_host(self, source_array, host_destination):
        if self.cast_buffer is None:
            source_array.copy_to_host(host_destination)
            return

        _cast_field_kernel[self.blocks_per_grid, self.threads_per_block, self.cast_stream](
            source_array,
            self.cast_buffer,
        )
        self.cast_buffer.copy_to_host(host_destination, stream=self.cast_stream)
        self.cast_stream.synchronize()

    def submit_frame(self, frame_index):
        if not self.enabled:
            return
        if self.failed:
            raise RuntimeError("GPU output exporter is in a failed state.")
        if self.component_sources is None:
            raise RuntimeError("GPU output exporter field state was not bound.")

        with self.slot_condition:
            while self.free_slots.empty() and not self.failed:
                self.slot_condition.wait(timeout=0.1)
            if self.failed:
                raise RuntimeError("GPU output exporter is in a failed state.")
            slot = self.free_slots.get_nowait()

        for component_index, component_name in enumerate(self.component_names):
            self._copy_component_to_host(
                self.component_sources[component_name],
                slot.array[component_index],
            )

        self.submit_queue.put((slot, slot.payload(frame_index)))

    def close(self):
        if not self.enabled:
            return

        self.submit_queue.put(None)
        self.submit_queue.join()
        self.submit_thread.join()

        for slot in self.slots:
            slot.close()

        if self.submit_thread.exception is not None:
            raise RuntimeError(str(self.submit_thread.exception))
