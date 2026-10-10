Installation
============

Continuum Flow is based on the programming languages Python and OpenCL. Python handles the mangament and UI, OpenCL does the actual compute. OpenCL requieres a driver to be able to run on a device (device means CPU OR GPU). For GPUs this is simply the normal GPU driver. For CPU a specific driver needs to be installed (see below). AMD CPUs do not support OpenCL officially, but there is a chance it still works by simply installing the Intel driver.

Requirements
-------------
General:

- Blender 5.0.0 or later
- for GPU: an up to date GPU driver
- Apples M-GPU support is not guranteed
- for Intel CPU: The OpenCL driver is needed: https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html
- for AMD CPU: AMDs CPU support is not guranteed. Currently there is no AMD OpenCL CPU driver. You can try installing the Intel driver above.


If you want to use the CPU, it is recommended to install the OpenCL driver first.

Steps
-----

1. Download the `.zip` matching your Blender version and operating system.
2. Open Blender.
3. Go to **Edit > Preferences > Add-ons**.
4. Click the downwards arrow in the top-right corner and select **Install from Disk**.
5. Select the downloaded Continuum Flow `.zip`.
6. Click **Install from Disk**.
