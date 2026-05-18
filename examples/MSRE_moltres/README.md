# Moltres MSRE-like example

To run with Moltres, first edit the necessary data.

The data that will need to be changed is `msre_gentry_4g_fuel_rod0_BETA_EFF` and
`msre_gentry_4g_fuel_rod0_LAMBDA`.
If using a different number of groups, then the `moder` values will also have 
to be changed so that there are the same number of groups.
To convert MoSDeN delayed neutron yields to betas, simply divide by 2.4355 (for
thermal U235, for others see https://nds.iaea.org/sgnucdat/a6.htm).
The `data` directory provides several different possible values, simply rename 
those files to use them instead.
Once the data is configured, Moltres can be run.

If running a stationary problem, set the flow rate in the input files to zero.
Use `mpirun -np 8 ~/projects/moltres/moltres-opt -i auto_diff_rho.i` to run
the main input file.
Once it has finished, run 
`mpirun -np 8 ~/projects/moltres/moltres-opt no_dnp_sim.i` to get the keff for 
the problem with no delayed neutrons.
This can be used to calculate the effective delayed neutron fraction using the 
prompt fission method:
$$\beta_{eff} \approx 1 - \frac{k_p}{k}$$