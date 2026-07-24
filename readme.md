# Continuum Flow

Continuum Flow is a free and open-source Blender add-on for simulating smoke, fire, and gas flows.

The solver can run on the CPU or optionally on an NVIDIA GPU. On CPU it is roughly 2x faster than Blender's native solver, while GPU acceleration can provide significantly higher performance.

# Downloads

Download Continuum Flow from the [GitHub Releases](https://github.com/lutzr2000/Continuum_Flow/releases) page.

Choose the package matching your Blender version and operating system:

| Blender Version | Windows | Linux | macOS |
|---|---|---|---|
| **Blender 5.0.x** | [Download Windows x64](https://github.com/lutzr2000/Continuum_Flow/releases/download/v0.1.0/continuum_flow-0.1.0-blender5.0-windows-x64.zip) | [Download Linux x64](https://github.com/lutzr2000/Continuum_Flow/releases/download/v0.1.0/continuum_flow-0.1.0-blender5.0-linux-x64.zip) | [Download macOS ARM64](https://github.com/lutzr2000/Continuum_Flow/releases/download/v0.1.0/continuum_flow-0.1.0-blender5.0-macos-arm64.zip) |
| **Blender 5.1+** | [Download Windows x64](https://github.com/lutzr2000/Continuum_Flow/releases/download/v0.1.0/continuum_flow-0.1.0-blender5.1plus-windows-x64.zip) | [Download Linux x64](https://github.com/lutzr2000/Continuum_Flow/releases/download/v0.1.0/continuum_flow-0.1.0-blender5.1plus-linux-x64.zip) | [Download macOS ARM64](https://github.com/lutzr2000/Continuum_Flow/releases/download/v0.1.0/continuum_flow-0.1.0-blender5.1plus-macos-arm64.zip) |

**GPU support:** NVIDIA CUDA acceleration is available on Windows and Linux. macOS currently uses the CPU solver.

# Requirements

- Blender 5.0.0 or higher
- Optional for GPU acceleration: NVIDIA GPU with the required CUDA environment

CUDA Toolkit: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda/toolkit)

# Installation

1. Go to the [GitHub Releases](https://github.com/lutzr2000/Continuum_Flow/releases) page.
2. Download the `.zip` matching your Blender version and operating system.
3. **Do not extract the downloaded `.zip`.**
4. Open Blender.
5. Go to **Edit > Preferences > Add-ons**.
6. Click the downwards arrow in the top-right corner and select **Install from Disk**.
7. Select the downloaded Continuum Flow `.zip`.
8. Click **Install from Disk**.

When you start a simulation for the first time, initialization may take a moment because parts of the solver need to be compiled.

You're done!

# How to Start

Continuum Flow comes with example files that you can use to get started.

# Documentation

[Continuum Flow Documentation](https://lutzr2000.github.io/Continuum_Flow/)
