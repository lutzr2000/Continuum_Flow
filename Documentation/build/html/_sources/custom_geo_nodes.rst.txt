Custom Geometry Nodes
=====================

Continuum Flow not only provides a simulation suite for flow simulation but also further tools to post process and enhance the results based on geometry nodes. These node setups are all build on Blenders default geometry nodes and can be customized to your liking. You can also freely use them on any other simulation result from other solvers.

Here it is important to mention that i did not come up with these node setups on my own, so some great tutorials need mentioning here:

DefaultCube (https://www.youtube.com/watch?v=U6QeP8eFWRQ)
...

Resimulate
----------
Allows you to retrace a field (like smoke) through the already computed velocity field.




Edge Fade
---------
Allows for fade of smoke density towards the edges when a field reaches the domain boundary.



Particle Advect
---------------
Can be used to advect particles through the flow.