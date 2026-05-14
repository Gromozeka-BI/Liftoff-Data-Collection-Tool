# Experiment 11: Camera Module Import

This folder tracks the staged import of the FPV camera localization module into
DCT.

Primary plan:

- `docs/camera_module_import_plan.md`

Initial scope:

1. Import geometry/PnP code from `FPVCamDetectV2`.
2. Import YOLO label adapter from `FPVCamDetectV2`.
3. Generate offline `CameraObservation` files.
4. Test guarded fusion with `OnlineLocalizer.inject_position_observation(...)`.

