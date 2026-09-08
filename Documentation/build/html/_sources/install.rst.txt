Installation
============

Continuum Flow consists of two parts: the UI part within Blender and the external Python script running the actual simulation. GPU is not supported on MacOS.

Requirements
-------------
General:

- Blender 5.0.0 (currently only supported version)
- for GPU: CUDA Toolkit (https://developer.nvidia.com/cuda-downloads)

If you want to use the GPU, it is recommended to install the CUDA Toolkit first.

Steps
-----

1. Install CUDA Toolkit
2. Download the code from the repo for your operating system and Blender version
3. Open Blender
4. Go to Edit > Preference > Add-ons
5. In the top right corner, click on the downwards arrow and select install from disk
6. Navigate to the downloaded .zip and click Install from Disk
7. When you start a bake for the first time the start may take a moment since the code needs to be compiled. 
