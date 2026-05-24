Continuum Flow
==============
Bringing the fun of flow simulation to Blender!

General
-------
This Addon allows for CPU and NVIDIA-GPU based flow simulations within Blender. It is free and Open-Source! The goal is to make simulating things like smoke and fire in Blender faster, more intuitive and with that more fun! The solver can be somewhere between 1.5 and 3 times faster on CPU and between 9 and 18 times faster on GPU than Blenders Native solver. The addon is integrated into Blender and comes with its own custom node tree. Additionally the Addon comes with a few geometry nodes setups to post process the computed results and raise the quality of the results even higher. The solver is based on the great tutorial by Prof. Dr. Zhengtao Gao (https://drzgan.github.io/Python_CFD/intro.html) and on the methods presented in the paper "Stable fluids" (https://doi.org/10.1145/311535.311548).

Features
--------
- Simulating velocity, pressure, fire, smoke and temperature
- Greatly improved performance compared to Blenders native solver
- Directly integrated in Blender
- Interaction with obstacles (stationary and movable)
- Simple combustion model
- Easy to set up

Limitations
-----------
- GPU is only supported for Nvidia cards
- No simulation of multi-phase flow (e.g. water)
- No interaction with Blenders native force fields
- No interaction with Blenders particle systems
- Obstacles can not deform (so no shape keys or armatures)

Disclaimers
-----------
- this is a hobby project, continued development is not guranteed, others are invited to participate
- i am not a professional developer, large parts where developed with the support of CODEX (AI); i tried to make sure to document and organize the software well, but still => there will be bugs
- currently the software is in beta => there will be bugs
- some things in the software might change in the future

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
