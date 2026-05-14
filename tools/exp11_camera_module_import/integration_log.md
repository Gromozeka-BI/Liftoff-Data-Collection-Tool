# Exp 11 Camera Replay Integration Log

This file records the implementation steps for integrating the imported camera
localization module into Replay first, then Record.

## 2026-05-14

### Step 1 - Scope

Goal for the first implementation milestone:

```text
recorded session video.mp4
  -> offline camera observations
  -> camera_observations.jsonl
  -> later Replay CamKF purple arrow
```

Important decision: do not run YOLO inside the Replay GUI loop yet. Generate a
plain observation file first so timestamps, PnP, sigma, and reject policy can be
debugged independently from GUI rendering.

### Step 2 - Timestamp Source

Use `video_timestamps.parquet` as the primary source of frame timestamps. It maps
`frame_idx -> ts_wall` and is documented in `docs/session_outputs.md`.

Fallback only when that file is missing:

```text
timestamp = first_session_ts + frame_idx / fps
```

This fallback is less reliable and should be reported in the generator summary.

### Step 3 - Observation Schema

Added `dct.camera_localization.observation.CameraObservation` and JSONL helpers.
The first Replay artifact will be:

```text
session/camera_observations.jsonl
```

Each line is one observation, including rejected observations with `status` and
`reason` so the full camera pipeline is auditable.

### Step 4 - Offline Video Generator

Added:

```text
tools/exp11_camera_module_import/generate_camera_observations.py
```

Current behavior:

```text
session/video.mp4
  -> frame timestamps from session/video_timestamps.parquet
  -> YOLO weights from models/yolo_gate_pose/testgate/weights/best.pt
  -> imported gate adapter / CoarseRefineLocalizer
  -> session/camera_observations.jsonl
```

The generator uses telemetry `pos_x/pos_y/pos_z` as the first coarse prior. This
is acceptable for the first offline artifact check, but the Replay integration
must later replace that with the live `OnlineLocalizer`/`KFLayer2` prior.

### Step 5 - Replay HUD/Preview Infrastructure

Added GUI infrastructure for the future camera-fused Replay contour:

- `CamKF` HUD row and map marker layer;
- purple map arrow/trail colors;
- `VideoPreviewWidget` gate overlay API for YOLO bbox/keypoints;
- HUD `CamKF` checkbox toggles the video overlay.

Behavior decision:

```text
CamKF checked   -> Video Preview may draw YOLO gate labels/keypoints.
CamKF unchecked -> Video Preview shows the original video exactly as before.
```

The actual Replay camera observation reader and CamKF computation are still the
next step. This step only prepares the visual layer and the on/off control.

### Step 6 - Replay Observation Overlay

Added Replay-side loading of:

```text
session/camera_observations.jsonl
```

The file is read when a Replay session is selected. For each telemetry/video
update, the nearest observation by `ts_wall` is converted to a `VideoPreview`
overlay if it is not older than the current tolerance.

The generator and schema now carry:

```text
bbox_xyxy
keypoints
```

so the Video Preview can draw YOLO gate boxes and corners. The `CamKF` HUD row
still controls whether the overlay is visible. When `CamKF` is unchecked, replay
video remains unchanged.

Also fixed the HUD height calculation so the always-present `CamKF` row no
longer overflows outside the panel.

## 2026-05-15 - Replay CamKF overlay debug

The first Replay test showed no YOLO overlay with `CamKF` enabled because the
selected session did not yet have:

```text
camera_observations.jsonl
```

Generated a sampled observation file for the session. The GUI-side Replay logic
now also reloads `camera_observations.jsonl` when `CamKF` is enabled and no
observations are loaded yet. Overlay lookup now uses the nearest camera
observation timestamp around the current Replay video time instead of only the
previous telemetry timestamp.

