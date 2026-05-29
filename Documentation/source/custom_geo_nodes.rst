Custom Geometry Nodes
=====================

Continuum Flow not only provides a simulation suite for flow simulations, but also additional tools to post-process and enhance the results based on geometry nodes. These node setups are all built on Blender's default geometry nodes and can be customized to your liking. You can also freely use them on simulation results from other solvers.

It is important to mention that I did not come up with these node setups entirely on my own, so some great tutorials should be credited here:

DefaultCube (https://www.youtube.com/watch?v=U6QeP8eFWRQ)
...

Resimulate
----------
Allows you to retrace a field (like smoke) through the already computed velocity field.




Edge Fade
---------
Allows the smoke density to fade towards the edges when a field reaches the domain boundary.



Particle Advect
---------------
Can be used to advect particles through the flow.
