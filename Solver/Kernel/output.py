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
    Download sparse OpenCL fields into host memory.

    fields:
        dict[str, cl.Buffer]

    Returns:
        dict containing mapped numpy arrays,
        OpenCL host buffers and transfer events.
    """

    dtype = np.dtype(dtype)

    cells_per_tile = tile_size**3
    bytes_per_tile = cells_per_tile * dtype.itemsize

    if fields:
        field_sizes = {name: buffer.size for name, buffer in fields.items()}
        unique_field_sizes = set(field_sizes.values())
        if len(unique_field_sizes) != 1:
            raise ValueError(f"Output field buffer sizes differ: {field_sizes}")

        pool_nbytes = unique_field_sizes.pop()
        if pool_nbytes % bytes_per_tile != 0:
            raise ValueError(
                f"Output pool size {pool_nbytes} is not divisible by "
                f"the tile size {bytes_per_tile}"
            )
        slot_count = pool_nbytes // bytes_per_tile

    cell_count = slot_count * cells_per_tile

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

    # Download field data
    if cell_count > 0:

        for name, gpu_buffer in fields.items():

            result["fields"][name] = copy_to_host(
                gpu_buffer,
                (slot_count, tile_size, tile_size, tile_size),
                dtype,
            )

    # Download tile map
    result["tile_map"] = copy_to_host(
        tile_map,
        tile_shape,
        np.int32,
    )

    return result


def release(queue, output_data):
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
