Continuum Flow
==============
Bringing the fun of flow simulation to Blender!

General
-------
This add-on allows for CPU- and NVIDIA GPU-based flow simulations within Blender. It is free and open source. The goal is to make simulating things like smoke and fire in Blender faster, more intuitive, and therefore more fun. The solver can be somewhere around 2 times faster on the CPU and be roughly 20 times faster on the GPU than Blender's native solver. The add-on is integrated into Blender and comes with its own custom node tree. The solver is based on this great tutorial (https://drzgan.github.io/Python_CFD/intro.html) by Prof. Dr. Zhengtao Gao and partially on the methods presented in the paper "Stable fluids" (https://doi.org/10.1145/311535.311548).

Features
--------
- Simulating velocity, pressure, fire, smoke and temperature
- Greatly improved performance compared to Blender's native solver
- Directly integrated into Blender
- Interaction with obstacles (stationary and movable)
- Simple combustion model
- Easy to set up

Limitations
-----------
- GPU is only supported for NVIDIA GPUs
- No simulation of multi-phase flow (e.g. water)
- No interaction with Blender's native force fields
- No interaction with Blender's particle systems
- Obstacles cannot deform (shape keys or armatures have no effect)

Disclaimers
-----------
- This is a hobby project. Continued development is not guaranteed, but others are invited to participate.
- I am not a professional developer. Large parts were developed with the support of CODEX (AI). I tried to make sure to document and organize the software well, but there will still be bugs.
- The software is currently in alpha, so there will be bugs.
- Some things in the software might change in the future.

Getting started
---------------
You can start by following the installation instructions in the corresponding section. The code from the Git reposetory also contains exaple files you can start with. If you find bugs or other issues, please report them in the GitHub issue tracker. 

Further sections
----------------

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   install
   user_documentation
   examples
   best_practice
