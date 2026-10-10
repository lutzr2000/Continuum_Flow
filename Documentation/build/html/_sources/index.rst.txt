Continuum Flow
==============
Bringing the fun of flow simulation to Blender!

General
-------
This add-on allows for CPU- and GPU-based flow simulations within Blender. It is free and open source. The goal is to make simulating things like smoke and fire in Blender faster, more intuitive, and therefore more fun. The solver can be somewhere around 8 times faster on the CPU and roughly 60 times faster on the GPU than Blender's native solver (depends from case to case and hardware). The add-on is integrated into Blender and comes with its own custom node tree. The solver started from this great tutorial (https://drzgan.github.io/Python_CFD/intro.html) by Prof. Dr. Zhengtao Gao.

Features
--------
- Simulating velocity, pressure, fire, smoke and temperature
- Greatly improved performance compared to Blender's native solver
- Directly integrated into Blender
- Interaction with obstacles (stationary and movable)
- Particle systems can be used as sources
- Combustion model
- Easy to set up

Limitations
-----------
- No interaction with Blender's native force fields
- Obstacles cannot deform (shape keys or armatures have no effect)

Disclaimers
-----------
- The software is currently in alpha, so bugs can be expected. Please report them on GitHub.
- Some things in the software might change in the future. Currently compatibility between versions is not guranteed.

Getting started
---------------
You can start by following the installation instructions in the corresponding section. The code from the Git repository also contains example files you can start with.

Further sections
----------------

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   install
   user_documentation
   examples
   best_practice
