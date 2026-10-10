# Continuum Flow

Continuum Flow is a free and open-source Blender add-on for simulating smoke, fire, and gas flows.

The solver can run on the CPU or GPU. On CPU it is roughly 8x faster than Blender's native solver, while GPU acceleration can provide significantly higher performance up to 60x faster.

# Downloads

Choose the package matching your Blender version and operating system:

| Blender Version | Windows | Linux | macOS |
|---|---|---|---|
| **Blender 5.0.x** | [Download Windows x64](https://github.com/lutzr2000/Continuum_Flow/releases/download/v0.1.0/continuum_flow-0.1.0-blender5.0-windows-x64.zip) | [Download Linux x64](https://github.com/lutzr2000/Continuum_Flow/releases/download/v0.1.0/continuum_flow-0.1.0-blender5.0-linux-x64.zip) | [Download macOS ARM64](https://github.com/lutzr2000/Continuum_Flow/releases/download/v0.1.0/continuum_flow-0.1.0-blender5.0-macos-arm64.zip) |
| **Blender 5.1+** | [Download Windows x64](https://github.com/lutzr2000/Continuum_Flow/releases/download/v0.1.0/continuum_flow-0.1.0-blender5.1plus-windows-x64.zip) | [Download Linux x64](https://github.com/lutzr2000/Continuum_Flow/releases/download/v0.1.0/continuum_flow-0.1.0-blender5.1plus-linux-x64.zip) | [Download macOS ARM64](https://github.com/lutzr2000/Continuum_Flow/releases/download/v0.1.0/continuum_flow-0.1.0-blender5.1plus-macos-arm64.zip) |

Important: MacOS versions are offered, but it is not guranteed that they run.

# Requirements

- Blender 5.0.0 or higher
- for CPU: an OpenCL Driver

Intels OpenCL driver: [Intel OpenCL driver](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html)

# Installation

1. Download the `.zip` matching your Blender version and operating system.
2. Open Blender.
3. Go to **Edit > Preferences > Add-ons**.
4. Click the downwards arrow in the top-right corner and select **Install from Disk**.
5. Select the downloaded Continuum Flow `.zip`.
6. Click **Install from Disk**.

You're done!

# How to Start

Continuum Flow comes with example files that you can use to get started. After installation you can unpack the .zip and use the files from the folder "example files".

# Documentation

[Continuum Flow Documentation](https://lutzr2000.github.io/Continuum_Flow/)
