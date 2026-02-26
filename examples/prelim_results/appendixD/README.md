## Pulse and saturation input files

As of November 3, 2025, the independent fission yield is required to run the 
saturation simulation with custom nuclides (due to the NFY data requirement).
To get around this, simply run the saturation model, adjust the concentrations
to be 250 for A, 25 for B, 2.5 for C, and 0.25 for D.
Rerun the count rate using `mosden -cnt input_sat.json`.
Then, run `mosden -g input_sat.json` followed by
`mosden --post input_sat.json` in order to use those values.
The pulse simulation can be fully run as-is with no modifications.

## OMC input file

Run this only as `mosden -g input_omc.json`.
Make sure there is a file named `omc_fission.json` containing the fission history
of interest.
Configure the different irradiation types before starting (as described previously).
No change to the input file is necessary.

### Pulse irradiation
For a pulse irradiation, the concentrations should be equal to the yield over 
the number of fissions (sufficiently short irradiation such that there is no 
decay). This means if there is one total fission, the concentration should 
equal the yield. For a final irradiation time of 0.00001, the fission rate
should be 100000 over that time.
This is provided as `omc_fissions_p.json`, simply rename it to `omc_fission.json`.


### Saturation Irradiation
For a saturation irradiation, the concentrations should be equal to the yield over 
the decay constant (assuming one fission per second).
For a final irradiation time of 100000, the fission rate should be 100000 over that time.
This is provided as `omc_fissions_s.json`, simply rename it to `omc_fission.json`.