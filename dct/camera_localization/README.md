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
