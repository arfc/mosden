## Running PRKs

Adjust the `input.json` file.
The parameters from MoSDeN have to be scaled to the form acceptable by a PRK
solver.
An assumption is made that the importance term is one (which means the effective
delayed neutron fraction is equal to the delayed neutron fraction, so the 
yields from MoSDeN just need to be scaled by the total neutron yield).
If the parameters provided are already scaled, then set the neutrons per fission
to 1.
The only data combinations simulated will be `selections`.
The `step_relative_insertion` and `rho_relative_amplitude` are
in units of dollars.
The different `problem` options are `step_insertion`, `step_relative_insertion`,
`ramp`, `sine`, and `sine_relative`.
