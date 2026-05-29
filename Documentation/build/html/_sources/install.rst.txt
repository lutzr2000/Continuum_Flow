Installation
============

Continuum Flow consists of two parts: the UI part within Blender and the external Python script running the actual simulation.

Requirements
-------------
General:

- Blender 5.0.0 or higher
- Python installed on your computer

Extra for GPU:

- an NVIDIA GPU
- CUDA Toolkit (https://developer.nvidia.com/cuda-downloads)

If you want to use the GPU, it is recommended to install the CUDA Toolkit first.

Steps
-----

1. Download the code from the repository (LINK).
2. Open Blender > Edit > Preferences > Add-ons.
3. Select the small downward arrow in the top right and choose "Install from Disk".
4. Navigate to the `.zip` file on your computer, select it, and press the "Install from Disk" button in the bottom right.
5. Go back to the Add-ons section in the preferences and look for Continuum Flow.
6. Display more options with the right-pointing arrow next to the name.
7. Click "Install Python Environment". This might take a moment and requires an internet connection.
8. Wait until the add-on says that CPU support, and optionally GPU support, is ready.
