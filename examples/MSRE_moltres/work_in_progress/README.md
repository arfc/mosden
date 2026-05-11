Simulates a 2D MSRE-like system.
 - uses multiapp to simulate primary loop as 1D problem

# To simulate:

1. Convert `mesh.i` into `mesh.e` (not certain how yet)

2. Run the input file `precursor_dist_calc.i` in this directory:
   `mpirun -np 8 moltres-opt -i precursor_dist_calc.i`. This will generate the 
   temperature and precursor distributions at steady-state.

3. Run the `eigen_eval.i` input file to get the eigenvalue of that solve. Steps
   2 and 3 can be repeated for various `xsdata.json` files depending on the 
   DNP group parameters selected.


## Problem(s)
- The group fluxes are exceedingly small in `precursor_dist_calc`, which leads 
   to incorrect precursor distributions and temperatures. Potential solutions 
   are to assume a constant temperature profile, which may resolve this issue.
   However, using a constant inlet temperature previously did not fix the 
   problem previously. Instead, a function should set an AuxVar temperature.


