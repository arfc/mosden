As of November 3, 2025, the independent fission yield is required to run the 
saturation simulation with custom nuclides (due to the NFY data requirement).
To get around this, simply run the saturation model, adjust the concentrations
to be 250 for A, 25 for B, 2.5 for C, and 0.25 for D.
Rerun the count rate using `mosden -cnt input_sat.json`.
Then, run `mosden -g input_sat.json` followed by
`mosden --post input_sat.json` in order to use those values.
The pulse simulation can be fully run as-is with no modifications.