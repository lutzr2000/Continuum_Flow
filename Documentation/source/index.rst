Continuum Flow
==============

Continuum Flow is an extension for Blender that provides a new tool chain for simulating fire, smoke and air in general. The main goal of this project was to A: build a workflow that is more intuitive to use compared to Blenders native solver and B: provides more compute performance when simulating. While the former can be subjective and the software is build to the taste of myself, the performance can be measured and is compared in section :ref:`performance-section`. Continuum Flow is somewhere between 2-20 times faster than Blenders native solver, depending on the usage of CPU or GPU and the general simulation settings. Continuum Flow can run on CPU and GPU, even though only Nvidia GPUs are supported. The simulation definition is done via a custom node tree oriented at other commercial software.

At this point i want to point out that large parts of the code was written with the support of AI. I made this solver as a hobby project and i am not an experienced developer. Bugs in this code are inevitable, please feel free to report them on GitHub. Additionally it is absolutly necessary to mention Dr. Zhengtao Gao, on whoms great learning course (https://drzgan.github.io/Python_CFD/intro.html) the basis of this solver is build. 


In :doc:`Installation <install>` the installation of Continuum Flow is explained. The :doc:`User Documentation <user_documentation>` explains the workflow and the functions of each custom node. :doc:`Custom Geometry Nodes <custom_geo_nodes>` explains the usage of Blenders native geometry node setups that can be used for post processing the simulation result. The :doc:`Solver Theory <solver_theory>` chapter explains the mathematical theory behind the solver and the :doc:`Performance <performance>` chapter contains a performance evaluation. Lastly, the :doc:`Examples <examples>` chapter contains short explanations for every example file provided.


.. toctree::
   :maxdepth: 2
   :caption: Documentation

   install
   user_documentation
   custom_geo_nodes
   solver_theory
   performance
   examples


