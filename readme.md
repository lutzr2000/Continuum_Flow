# Continuum Flow
This is Continuum Flow, a free open source addon for Blender for simulating smoke, fire and gas flows in general. The solver can run on CPU and optionally on Nvidia GPU. On CPU it is roughly 3x faster than Blender's native solver and on GPU it is roughly 20 x faster.

# Requirements
- Blender 5.0.0 or higher
- `uv` package manager  
  Install with:
  `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- Optional for GPU: CUDA Toolkit  
  Download: [CUDA Toolkit](https://developer.nvidia.com/cuda/toolkit)

# Installation
1. Install the required dependencies listed above
2. Download the code from this page as a .zip
3. Open Blender
4. Go to Edit > Preference > Add-ons
5. In the top right corner, click on the downwards arrow and select install from disk
6. Navigate to the downloaded .zip and click Install from Disk
7. In the add-ons outliner go to Continuum Flow and click "Install Solver Enviroment" (This may take a moment)

You're done!

# How to start
The code comes with example files you can open and start playing around with.

