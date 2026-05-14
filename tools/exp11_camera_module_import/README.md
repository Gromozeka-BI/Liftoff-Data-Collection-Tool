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

## Smoke Test

Run the imported adapter + localization chain on an annotated calibration frame:

```bash
python tools/exp11_camera_module_import/smoke_camera_import.py
```

Multi-gate sample:

```bash
python tools/exp11_camera_module_import/smoke_camera_import.py --section multi_frames --frame-idx 0 --q-m 10 --coarse-offset-x 0
```

The script converts imported calibration keypoints into a temporary YOLO-pose
label, loads it through `yolo_gate_adapter`, and runs `CoarseRefineLocalizer`.

## Imported Extras

Also imported:

- `dct/camera_localization/calibration_tools/` for reproducing
  `camera_calibration.json`;
- `docs/camera/fpv_imported/` with reference design documents from
  `FPVCamDetectV2`;
- `dct/camera_localization/experimental/pnp_solver_2/` as a comparison-only
  AP3P/LM prototype.

`pnp_solver_2` is not part of the default runtime path because its current
validation has large multi-gate outliers.

