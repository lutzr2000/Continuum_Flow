Performance
===========

To access the performance of the solver the three example files rising smoke, camp fire and car drift are simulated at different grid resolutions on the same machine. Additionally for the examples rising smoke and camp fire the setup is also matched as closely as possible within Blenders native solver to compare performance.

Machine used for testing:

- OS: Windows 11
- CPU: Intel Core i9-14900KF
- RAM: 64 GB
- Disk: NVMe SSD Viper VP4300L 2TB
- GPU: NVIDIA GeForce RTX 5080


Solve time
----------
Plotting here solve time against simulated cell size. Keep in mind that decreasing the cell size by a factor of two creates 8 times more cells, drastically increasing compute cost.

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

Continuum flow consistently outperforms Blenders native solver even on CPU. GPU solves are significantly faster.

(V)RAM usage
------------
Here (V)RAM usage is plotted again against cell size. Here Blenders native results are not included because i could not track them properly.

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

The GPU solver needs more VRAM han the CPU solver needs RAM at lower resolutions (effectivly (V)RAM usage is related to cell count which is related to resolution). At higher resolutions the GPU solver needs slightly less VRAM than the CPU needs RAM. Usually systems have les VRAM than RAM. If your GPU runs out of VRAM the solver will use RAM from shared memory, but this is significantly slower (Still faster than pure CPU runs in many cases).

Realtime simulation
-------------------
Here the ratio of solve time (the time it actually took to simulate the result) to simulated time (the length of the simulation) is plotted. A value of one (the black dotted line) means simulated in realtime. Values smaller mean simualted faster than realtime, larger mean slower than realtime. Mind the log axis!

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

The GPU sovler can even simulate at realtime on very small cell sizes (meaning large cell count). Please keep in mind that simulation speed is also related to the velocity of the flow. Faster flow takes longer to simulate.
