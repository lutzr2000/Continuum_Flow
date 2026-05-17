Solver Theory
=============

This section describes the numerical model used by the Continuum flow solver. The
simulation is based on the incompressible Navier-Stokes equations and uses a
regular Cartesian grid with a finite-difference discretization. The goal is not
to reproduce every detail of real fluid physics, but to provide a stable and
fast solver for visually convincing smoke and fire effects.


Governing Equations
-------------------

The solver starts from the incompressible Navier-Stokes equations in vector
form.

Conservation of mass:

.. math::

   \nabla \cdot \mathbf{u} = 0

Conservation of momentum:

.. math::

   \frac{\partial \mathbf{u}}{\partial t}
   + \left( \mathbf{u} \cdot \nabla \right) \mathbf{u}
   = -\frac{1}{\rho} \nabla p
   + \nu \nabla^2 \mathbf{u}
   + \mathbf{f}

Here, :math:`\mathbf{u}` is the velocity field, :math:`p` is the pressure,
:math:`\rho` is the fluid density, :math:`\nu` is the kinematic viscosity and
:math:`\mathbf{f}` contains external forces such as buoyancy, turbulence and
user-authored force fields.


Finite-Difference Discretization
--------------------------------

The simulation domain is discretized into a uniform grid with cell spacing
:math:`\Delta`. All derivatives are approximated directly on this grid by finite
differences.

For a scalar quantity :math:`\phi`, the solver uses central differences for
diffusion and for most derivatives appearing in the pressure equation:

.. math::

   \frac{\partial \phi}{\partial x}
   \approx \frac{\phi_{i+1,j,k} - \phi_{i-1,j,k}}{2\Delta}

.. math::

   \frac{\partial^2 \phi}{\partial x^2}
   \approx \frac{\phi_{i+1,j,k} - 2\phi_{i,j,k} + \phi_{i-1,j,k}}{\Delta^2}

The three-dimensional Laplacian is built from the sum of the second
derivatives in :math:`x`, :math:`y` and :math:`z`. The viscous term in the
momentum equation therefore becomes a standard 7-point stencil on the grid.

Time integration is explicit for advection, diffusion and force application. A
stable timestep is chosen automatically from the most restrictive of three
limits:

.. math::

   \Delta t_{\mathrm{conv}}
   = \min \left(
   \frac{\mathrm{CFL}\,\Delta}{|u|_{\max}},
   \frac{\mathrm{CFL}\,\Delta}{|v|_{\max}},
   \frac{\mathrm{CFL}\,\Delta}{|w|_{\max}}
   \right)

.. math::

   \Delta t_{\mathrm{diff}} = \frac{\Delta^2}{6\nu}

.. math::

   \Delta t_{\mathrm{force}}
   = \min \left(
   \frac{\mathrm{CFL}\,\Delta\,\rho}{|F_x|_{\max}},
   \frac{\mathrm{CFL}\,\Delta\,\rho}{|F_y|_{\max}},
   \frac{\mathrm{CFL}\,\Delta\,\rho}{|F_z|_{\max}}
   \right)

The final timestep is

.. math::

   \Delta t = \min\left(
   \Delta t_{\mathrm{conv}},
   \Delta t_{\mathrm{diff}},
   \Delta t_{\mathrm{force}}
   \right)

and is additionally clamped to the current output interval.


Semi-Lagrangian and MacCormack Advection
----------------------------------------

The nonlinear transport term :math:`(\mathbf{u}\cdot\nabla)\mathbf{u}` is the
most delicate part of the update. In the current solver, both velocity and
scalar transport are handled with a semi-Lagrangian backtracing scheme followed
by a MacCormack correction step. This is more stable than a purely explicit
high-order finite-difference advection and less diffusive than a plain
first-order semi-Lagrangian pass.

Semi-Lagrangian Predictor
~~~~~~~~~~~~~~~~~~~~~~~~~

For each cell center, the solver traces a departure position backward through
the current velocity field:

.. math::

   \mathbf{x}_d = \mathbf{x}_{i,j,k} - \Delta t \, \mathbf{u}(\mathbf{x}_{i,j,k})

The traced position is evaluated in grid coordinates and sampled with trilinear
interpolation. In implementation, the backtrace is split into a few small
substeps to make the characteristic integration more robust in strongly varying
flow fields.

For a transported quantity :math:`\phi`, the predictor therefore becomes

.. math::

   \phi^\star_{i,j,k} = \phi^n(\mathbf{x}_d)

The same idea is applied component-wise to the velocity field.

MacCormack Correction
~~~~~~~~~~~~~~~~~~~~~

The pure semi-Lagrangian predictor is very robust, but it tends to smooth out
detail. To recover sharper structures, the solver adds a MacCormack-style
correction. After the backward predictor, a forward trace through the predictor
field estimates the local transport error:

.. math::

   \phi^{\mathrm{rev}}_{i,j,k} = \phi^\star(\mathbf{x}_f)

The corrected value is then

.. math::

   \phi^{n+1}_{i,j,k}
   =
   \phi^\star_{i,j,k}
   +
   \kappa
   \left(
   \phi^n_{i,j,k} - \phi^{\mathrm{rev}}_{i,j,k}
   \right)

where :math:`\kappa` is the user-controlled MacCormack factor.

To avoid overshoots and new extrema, the corrected result is clamped to the
value range of the departure cell used by the semi-Lagrangian sample. This is
important for both visual stability and bounded fields such as smoke and fuel.

Velocity Update Around the Advection Step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For velocity, the corrected advective update is not the final value yet.
Diffusion and external forces are added explicitly:

.. math::

   \mathbf{u}^{\dagger}
   =
   \mathbf{u}^{\mathrm{adv}}
   +
   \nu \Delta t \, \nabla^2 \mathbf{u}
   +
   \frac{\Delta t}{\rho}\mathbf{f}

The implementation also limits the per-step velocity increment to avoid large
single-step jumps in highly forced setups. Pressure is not applied in this
predictor stage; incompressibility is enforced afterwards with a dedicated
pressure projection.


Pressure Solve and Enforcing Incompressibility
----------------------------------------------

After the velocity predictor and boundary update, the solver computes a
pressure field that removes the remaining divergence. The pressure equation is
solved as a Poisson problem:

.. math::

   \nabla^2 p = b

The right-hand side :math:`b` is assembled from the divergence of the predicted
velocity, an authored divergence source, and the thermal-expansion term:

.. math::

   b =
   \frac{\rho}{\Delta t}
   \left(
   \nabla \cdot \mathbf{u}
   + d_{\mathrm{authored}}
   - d_{\mathrm{thermal}}
   \right)

where :math:`d_{\mathrm{authored}}` is an additional user-authored divergence
source and :math:`d_{\mathrm{thermal}}` is the thermal expansion term
discussed below.

The pressure field is solved iteratively with a red-black SOR method built on
top of Gauss-Seidel updates. Each iteration updates alternating checkerboard
cells and reapplies homogeneous Neumann boundary conditions, meaning the normal
pressure gradient at the domain boundary is zero. In discrete form, boundary
pressure values are copied from the adjacent interior cells.

Because a Poisson equation with pure Neumann boundary conditions only defines
pressure up to an arbitrary constant, the discrete right-hand side must be
compatible. The solver therefore subtracts the mean value of :math:`b` before
starting the iterations, which removes the null-space inconsistency.

The number of pressure iterations is user-controlled. Higher values improve the
enforcement of incompressibility, while lower values are faster but can make the
motion softer or less stable.


Buoyancy
--------

Buoyancy is modeled with a Boussinesq-style approximation. Instead of solving a
fully variable-density flow, the solver assumes that density differences are
small and only matter in the body-force term. The buoyancy force is applied in
the vertical direction:

.. math::

   F_z \leftarrow F_z + g \, \beta \, (T - T_{\mathrm{ref}})

Here, :math:`g = 9.81`, :math:`\beta` is the user-controlled buoyancy factor,
:math:`T` is the local temperature and :math:`T_{\mathrm{ref}}` is the
reference temperature. If a cell is hotter than the reference temperature, the
vertical force becomes positive and the fluid rises. If it is colder, the force
becomes negative and the fluid sinks.

This is a deliberate simplification. The solver does not compute a full
equation of state, but this approximation is efficient and gives intuitive
control over rising hot smoke and sinking cold fluid.


Thermal Expansion
-----------------

Thermal expansion is modeled separately from buoyancy. Instead of only pushing
hot fluid upward, the solver also allows hot regions to act as volumetric
sources in the pressure projection step. Here it is important to note that we deviate from real physics here. The implementation is done via an artificial
divergence term:

.. math::

   d_{\mathrm{thermal}} =
   \alpha \, (T - T_{\mathrm{ref}})

where :math:`\alpha` is the user-controlled expansion rate. This term appears
in the pressure right-hand side with a negative sign:

.. math::

   b \sim \frac{\rho}{\Delta t}
   \left(
   \nabla \cdot \mathbf{u}
   + d_{\mathrm{authored}}
   - d_{\mathrm{thermal}}
   \right)

As a result, temperatures above the reference temperature reduce the pressure
right-hand side locally and create outward motion after the projection. In
visual terms, hot regions do not only rise because of buoyancy, they also
expand and push neighboring fluid away.

Vorticity
---------

Vorticity is used to recover small-scale swirling motion that is often damped by
grid discretization and by the numerically diffusive transport schemes. The
solver first computes the vorticity vector

.. math::

   \boldsymbol{\omega} = \nabla \times \mathbf{u}

with components

.. math::

   \omega_x = \frac{\partial w}{\partial y} - \frac{\partial v}{\partial z},
   \qquad
   \omega_y = \frac{\partial u}{\partial z} - \frac{\partial w}{\partial x},
   \qquad
   \omega_z = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}

using central differences on the grid. From this, the vorticity magnitude

.. math::

   |\boldsymbol{\omega}| =
   \sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2}

is evaluated in each fluid cell.

The user-facing vorticity setting does not change the physical viscosity.
Instead, it adds a vorticity-confinement force that re-injects rotational
energy into regions where vortices already exist. The method computes the
normalized gradient of vorticity magnitude

.. math::

   \mathbf{N} =
   \frac{\nabla |\boldsymbol{\omega}|}{|\nabla |\boldsymbol{\omega}||}

and then applies the force

.. math::

   \mathbf{f}_{\mathrm{vc}} = \epsilon \, (\mathbf{N} \times \boldsymbol{\omega})

where :math:`\epsilon` is the vorticity strength chosen by the user.

Intuitively, :math:`\mathbf{N}` points from regions of weaker rotation toward
regions of stronger rotation. The cross product with :math:`\boldsymbol{\omega}`
creates a force that wraps around existing vortices and sharpens them. This
helps the flow keep visually pleasing curls and rolling structures, especially
in smoke simulations where first-order advection would otherwise smooth them
out.

The confinement force is only applied in fluid cells and is skipped near solid
obstacles and very close to the domain boundary, where the required stencil
would be incomplete.


Scalar Fields: Temperature, Smoke and Fuel
------------------------------------------

The scalar fields temperature, smoke and fuel are all transported by the
projected velocity field with the same semi-Lagrangian predictor plus
MacCormack correction used for velocity. After the advection stage, each field
receives simple source and dissipation terms.

Fuel burns only when the temperature in a cell exceeds the ignition threshold:

.. math::

   S_{\mathrm{fuel}} =
   \begin{cases}
   -k_{\mathrm{burn}} \, f, & T > T_{\mathrm{ignition}} \\
   0, & \text{otherwise}
   \end{cases}

The temperature source combines dissipation toward the reference temperature and
heat release from burning fuel:

.. math::

   S_T =
   -k_T (T - T_{\mathrm{ref}})
   + q_T (-S_{\mathrm{fuel}})

Smoke is produced from burning fuel and dissipates over time:

.. math::

   S_{\mathrm{smoke}} =
   q_{\mathrm{smoke}} (-S_{\mathrm{fuel}})
   - k_{\mathrm{smoke}}

These source terms are intentionally simple. They are designed for direct
artistic control instead of chemically accurate combustion.


Solution Procedure
------------------

For each simulation substep, the solver advances the system in the following
order:

1. Update dynamic masks, animated constants and time-dependent force inputs.
2. Assemble the body-force field from authored forces and turbulence.
3. Add buoyancy to the vertical force component.
4. Optionally compute vorticity diagnostics and add vorticity-confinement
   forces.
5. Advect velocity with the semi-Lagrangian predictor and apply the MacCormack
   correction, diffusion and body forces.
6. Apply velocity, obstacle and source boundary conditions to the predicted
   velocity field.
7. Build and solve the pressure Poisson equation from that predicted field.
8. Project velocity with the solved pressure to enforce incompressibility.
9. Reapply boundary conditions after projection.
10. Advect and update temperature, smoke, fuel and flame with the projected
    velocity field.
11. Write output if the simulation time reached the next output point.
12. Recompute a stable timestep from the updated state.

This ordering is important because the solver now follows a projection-method
pipeline: forces and diffusion first build an intermediate velocity, and the
pressure solve then removes its divergence. Temperature affects buoyancy
through the body-force term and affects thermal expansion through the pressure
right-hand side, so the scalar state from the previous substep directly shapes
the current velocity predictor and projection.


Boundary Conditions and Practical Notes
---------------------------------------

The solver works on interior cells and handles boundaries explicitly. Pressure
uses zero-gradient boundary conditions during the Poisson solve. Velocity and
scalar fields are overwritten where necessary by the configured domain,
obstacle and source conditions.

The semi-Lagrangian sampler clamps traced positions to the valid grid range,
and the MacCormack correction is limited with local extrema clamps. Together,
these two safeguards keep the higher-detail transport step robust even near the
domain boundary.

To improve stability further, the velocity update also limits the maximum
per-step velocity change in strongly forced regions.
