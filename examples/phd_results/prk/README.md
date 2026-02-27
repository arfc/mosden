## Running PRKs

Adjust the `input.json` file.
The parameters from MoSDeN have to be scaled to the form acceptable by a PRK
solver.
An assumption is made that the importance term is one (which means the effective
delayed neutron fraction is equal to the delayed neutron fraction, so the 
yields from MoSDeN just need to be scaled by the total neutron yield).
If the parameters provided are already scaled, then set the neutrons per fission
to 1.