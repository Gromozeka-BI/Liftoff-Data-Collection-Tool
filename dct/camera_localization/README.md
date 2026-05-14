# Experimental Camera Localization

This package is reserved for the experimental FPV camera localization module.

The first integration target is an offline pipeline:

```text
YOLO labels / frames
  -> gate detections
  -> gate_id association + PnP
  -> CameraObservation
  -> optional DCT fusion policy
```

The main RC localizer must not depend on this package directly. Camera
observations should reach DCT through
`OnlineLocalizer.inject_position_observation(...)` after passing a fusion policy.

## Imported Runtime Modules

Imported from `FPVCamDetectV2`:

- `yolo_gate_adapter/`
- `gate_model/`
- `pnp_solver/`
- `gate_localization/`
- `config/camera_calibration.json`
- `config/track.json`
- `calibration/calibration_frames_card.json`
- `calibration_tools/`
- `experimental/pnp_solver_2/`

Imports were rewritten to use the `dct.camera_localization` namespace. Default
PnP paths now resolve inside this package.

`experimental/pnp_solver_2` is kept for comparison only. It is not used by the
default adapter/PnP/coarse-refine path.
