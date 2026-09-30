import numpy as np
import pyopencl as cl


def output(
    context,
    queue,
    fields,
    tile_map,
    tile_shape,
    tile_size,
    slot_count,
    dtype=np.float32,
):
    """
    Download sparse OpenCL fields and the complete tile_map.
    """
    dtype = np.dtype(dtype)
    slot_count = int(slot_count)
    tile_size = int(tile_size)

    if fields:
        field_sizes = {name: buffer.size for name, buffer in fields.items()}
        bytes_per_tile = tile_size**3 * dtype.itemsize
        pool_nbytes = next(iter(field_sizes.values()))
        slot_count = pool_nbytes // bytes_per_tile

    result = {
        "fields": {},
        "tile_map": None,
        "host_buffers": [],
        "mapped_arrays": [],
        "events": [],
        "slot_count": slot_count,
    }

    def copy_to_host(gpu_buffer, shape, data_type):
        data_type = np.dtype(data_type)
        nbytes = int(np.prod(shape)) * data_type.itemsize

        host_buffer = cl.Buffer(
            context,
            cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR,
            size=nbytes,
        )

        copy_event = cl.enqueue_copy(
            queue,
            host_buffer,
            gpu_buffer,
            byte_count=nbytes,
        )

        array, map_event = cl.enqueue_map_buffer(
            queue,
            host_buffer,
            cl.map_flags.READ,
            offset=0,
            shape=shape,
            dtype=data_type,
            wait_for=[copy_event],
            is_blocking=False,
        )

        result["host_buffers"].append(host_buffer)
        result["mapped_arrays"].append(array)
        result["events"].append(map_event)

        return array

    if slot_count > 0:
        for name, gpu_buffer in fields.items():
            result["fields"][name] = copy_to_host(
                gpu_buffer,
                (slot_count, tile_size, tile_size, tile_size),
                dtype,
            )

    result["tile_map"] = copy_to_host(
        tile_map,
        tile_shape,
        np.int32,
    )

    return result


def release(queue, output_data):
    if output_data["events"]:
        cl.wait_for_events(output_data["events"])

    unmap_events = []

    for array in output_data["mapped_arrays"]:
        event = array.base.release(queue)
        unmap_events.append(event)

    if unmap_events:
        cl.wait_for_events(unmap_events)

    output_data["fields"].clear()
    output_data["tile_map"] = None
    output_data["mapped_arrays"].clear()
    output_data["host_buffers"].clear()
    output_data["events"].clear()
