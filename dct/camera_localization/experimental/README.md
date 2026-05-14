# Experimental Prototypes

This folder contains camera-localization prototypes that are imported for
comparison, but are not used by the default runtime pipeline.

## `pnp_solver_2`

`pnp_solver_2` is an AP3P/LM prototype. It imports and runs, but validation on
the current calibration card shows unstable multi-gate poses and large outliers.
Keep it as a reference implementation until it beats the default
`dct.camera_localization.pnp_solver` on the Exp 11 metrics.

