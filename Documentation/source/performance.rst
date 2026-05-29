Performance
===========

To assess the performance of the solver, the three example files "rising smoke", "camp fire", and "car drift" are simulated at different grid resolutions on the same machine. Additionally, for the "rising smoke" and "camp fire" examples, the setup is also matched as closely as possible within Blender's native solver to compare performance.

Machine used for testing:

- OS: Windows 11
- CPU: Intel Core i9-14900KF
- RAM: 64 GB
- Disk: NVMe SSD Viper VP4300L 2TB
- GPU: NVIDIA GeForce RTX 5080


Solve time
----------
The plots below show solve time against simulated cell size. Keep in mind that decreasing the cell size by a factor of two creates 8 times more cells, drastically increasing compute cost.

01 rising smoke

.. figure:: ../images/01_rising_smoke_scaling_time.png
   :class: block-image-left
   :width: 600px

02 camp fire

.. figure:: ../images/02_camp_fire_scaling_time.png
   :class: block-image-left
   :width: 600px

05 car drift

.. figure:: ../images/05_car_drift_scaling_time.png
   :class: block-image-left
   :width: 600px

Continuum Flow consistently outperforms Blender's native solver even on the CPU. GPU solves are significantly faster.

(V)RAM usage
------------
Here, (V)RAM usage is plotted against cell size again. Blender's native results are not included here because I could not track them properly.

01 rising smoke

.. figure:: ../images/01_rising_smoke_scaling_RAM.png
   :class: block-image-left
   :width: 600px

02 camp fire

.. figure:: ../images/02_camp_fire_scaling_RAM.png
   :class: block-image-left
   :width: 600px

05 car drift

.. figure:: ../images/05_car_drift_scaling_RAM.png
   :class: block-image-left
   :width: 600px

At lower resolutions, the GPU solver needs more VRAM than the CPU solver needs RAM. Effectively, (V)RAM usage is related to cell count, which in turn is related to resolution. At higher resolutions, the GPU solver needs slightly less VRAM than the CPU needs RAM. Usually, systems have less VRAM than RAM. If your GPU runs out of VRAM, the solver will use RAM from shared memory, but this is significantly slower, though still faster than pure CPU runs in many cases.

Realtime simulation
-------------------
Here, the ratio of solve time, meaning the time it actually took to simulate the result, to simulated time, meaning the length of the simulation, is plotted. A value of one, shown by the black dotted line, means the simulation ran in real time. Smaller values mean it simulated faster than real time, while larger values mean it simulated slower than real time. Mind the log axis.

01 rising smoke

.. figure:: ../images/01_rising_smoke_ratios.png
   :class: block-image-left
   :width: 600px

02 camp fire

.. figure:: ../images/02_camp_fire_ratios.png
   :class: block-image-left
   :width: 600px

05 car drift

.. figure:: ../images/05_car_drift_ratios.png
   :class: block-image-left
   :width: 600px

The GPU solver can even simulate in real time at very small cell sizes, meaning large cell counts. Please keep in mind that simulation speed is also related to the velocity of the flow. Faster flow takes longer to simulate.
