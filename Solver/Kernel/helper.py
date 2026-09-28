import numpy as np
import pyopencl as cl

import Solver.Kernel.kernel_config as kernel_config

mf = cl.mem_flags

FIELD_DTYPE = kernel_config.FIELD_DTYPE


def to_device(context, array):
    """NumPy-Array copy to OpenCL-Buffer."""
    return cl.Buffer(
        context,
        mf.READ_WRITE | mf.COPY_HOST_PTR,
        hostbuf=array,
    )


def device_array(context, shape, dtype):
    """Allocate uninitialized OpenCL buffer."""
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize

    return cl.Buffer(
        context,
        mf.READ_WRITE,
        size=size,
    )


def zeros_device(context, shape, dtype=FIELD_DTYPE):
    return to_device(context, np.zeros(shape, dtype=dtype))


def full_device(context, shape, value, dtype=FIELD_DTYPE):
    return to_device(context, np.full(shape, value, dtype=dtype))
