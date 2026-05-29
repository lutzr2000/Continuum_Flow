Best practive
=============

In this last section a few tips for working with Continuum Flow are collected

- The most important rule: when something looks weird or the solver diverges try to reduce the CFL number. This sadly comes with the cost of slower compute time but it can often fix issues.
- While a large CFL value is desirable for quick simulations, some cases really profit from lower CFL values, like the 04 explosion example
- When starting your simulation setup, start with low resolutions and only go higher towards the end of your itteration procedure
- Only output the data you really need, saving volume grids can produce very large amounts of data really fast, this can fill up your disk and slow the solver
- when doing your final high res simulation turn off the "live preview", this frees up RAM for the solver

With these tips i wish you fun gathering experiences with Continuum Flow!

