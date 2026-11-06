# FHN solver

`parallel_bdf_fhn.cc` solves the FitzHugh-Nagumo system with BDF time stepping (PETSc TS)
and adaptive mesh refinement (deal.II). It builds as `parallel_bdf_fhn` (2D, wave segment, MPI) and as
`radial_fhn_bdf` (1D radial, serial), and takes one `input.prm` argument, for example
`kII_disk_radial_bdf.prm`.
