Continuum Flow
==============
Bringing the fun of flow simulation to Blender!

General
-------
This add-on allows for CPU- and NVIDIA GPU-based flow simulations within Blender. It is free and open source. The goal is to make simulating things like smoke and fire in Blender faster, more intuitive, and therefore more fun. The solver can be somewhere between 1.5 and 3 times faster on the CPU and between 9 and 18 times faster on the GPU than Blender's native solver. The add-on is integrated into Blender and comes with its own custom node tree. Additionally, it comes with a few geometry node setups to post-process the computed results and improve their visual quality even further. The solver is based on the great tutorial by Prof. Dr. Zhengtao Gao (https://drzgan.github.io/Python_CFD/intro.html) and partially on the methods presented in the paper "Stable fluids" (https://doi.org/10.1145/311535.311548).

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
- Obstacles cannot deform (so no shape keys or armatures)

Disclaimers
-----------
- This is a hobby project. Continued development is not guaranteed, but others are invited to participate.
- I am not a professional developer. Large parts were developed with the support of CODEX (AI). I tried to make sure to document and organize the software well, but there will still be bugs.
- The software is currently in beta, so there will be bugs.
- Some things in the software might change in the future.

Further sections
----------------

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   install
   user_documentation
   custom_geo_nodes
   solver_theory
   performance
   examples
   best_practice
