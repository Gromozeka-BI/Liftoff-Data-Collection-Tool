# YOLO Gate Adapter

`yolo_gate_adapter` is the input layer between YOLO keypoint labels and
`gate_localization`.

It does not choose `gate_id`. Its job is only to turn noisy YOLO detections into
a small, reliable list of `GateDetection` objects:

```text
YOLO labels
    -> parse normalized bbox/keypoints
    -> filter by keypoint confidence
    -> remove near-duplicate detections
    -> keep top-N detections
    -> GateDetection[]
    -> CoarseRefineLocalizer
```

## Input Format

Expected YOLO line format:

```text
class cx cy w h x1 y1 c1 x2 y2 c2 x3 y3 c3 x4 y4 c4
```

Coordinates are normalized to `[0..1]`.

The adapter converts keypoints to pixel coordinates using the configured image
size. Current default:

```text
image_width_px = 1920
image_height_px = 1080
```

The expected keypoint order is the same as in `GateDetection`:

```text
TL -> TR -> BR -> BL
```

## Current Defaults

```text
min_keypoint_confidence = 0.7
max_detections = 6
deduplicate_iou_threshold = 0.75
deduplicate_center_distance_px = 12
```

These defaults are conservative. They are based on the first analysis of:

```text
reference_lap_dataset_30fps_frames0_862/
yolo_auto_labels/best_conf025_imgsz640/labels
```

## Why Confidence 0.7

On the 30 FPS auto-labeled dataset:

```text
label files: 775
total YOLO detections: 3281
detections per labeled frame: 1..11
mean detections per labeled frame: 4.23
keypoint confidence median: 0.777
keypoint confidence p10: 0.560
keypoint confidence p90: 0.918
detections with at least one keypoint < 0.5: 620
```

Sample localization run on every 6th labeled frame:

```text
min_keypoint_confidence = 0.0
    accepted: 10
    improved KFLayer2: 0
    worsened KFLayer2: 10
    mean gain: -2.28 m

min_keypoint_confidence = 0.5
    accepted: 8
    improved KFLayer2: 0
    worsened KFLayer2: 8
    mean gain: -1.63 m

min_keypoint_confidence = 0.7
    accepted: 5
    improved KFLayer2: 4
    worsened KFLayer2: 1
    mean gain: +0.35 m
    median gain: +1.10 m
```

Conclusion: low-confidence YOLO keypoints are dangerous for localization. They
may pass geometric checks but still degrade `KFLayer2`. For the current model,
`0.7` is the first practical threshold.

Full threshold comparison on all `863` exported frames:

```text
min_keypoint_confidence = 0.5
    frames with detections after adapter: 729
    detections after adapter: 2441
    accepted by localization: 49
    improved KFLayer2: 13
    worsened KFLayer2: 36
    mean gain: -0.80 m
    median gain: -0.95 m

min_keypoint_confidence = 0.6
    frames with detections after adapter: 617
    detections after adapter: 1467
    accepted by localization: 55
    improved KFLayer2: 18
    worsened KFLayer2: 37
    mean gain: -0.45 m
    median gain: -0.76 m

min_keypoint_confidence = 0.7
    frames with detections after adapter: 221
    detections after adapter: 273
    accepted by localization: 24
    improved KFLayer2: 13
    worsened KFLayer2: 11
    mean gain: -0.25 m
    median gain: +0.34 m

min_keypoint_confidence = 0.8
    frames with detections after adapter: 2
    detections after adapter: 2
    accepted by localization: 0

min_keypoint_confidence = 0.9
    frames with detections after adapter: 0
    detections after adapter: 0
    accepted by localization: 0
```

Current decision: keep `min_keypoint_confidence = 0.7`.

Lower thresholds produce many more detections, but most accepted observations
hurt `KFLayer2`. Higher thresholds remove almost all usable data.

## Why Top-N Filtering

The current `CoarseRefineLocalizer` generates ID hypotheses for the visible
gates. Auto-labels can produce many detections per frame, including duplicates
and weak gates. Frames with `8..11` detections cause a combinatorial blow-up and
make the full 30 FPS pass too slow.

Therefore the adapter keeps only the most reliable detections before
localization. Current default is:

```text
max_detections = 6
```

The ranking is:

```text
min keypoint confidence
mean keypoint confidence
bbox area
```

This favors gates with all four corners confidently detected.

## Deduplication

YOLO can detect the same physical gate more than once. The adapter removes a
detection if it is too close to a stronger detection:

```text
duplicate if center distance <= 12 px
duplicate if bbox IoU >= 0.75
```

This is intentionally simple. If duplicate detections remain a major issue, the
next version should use polygon/keypoint IoU or temporal tracking.

## Runtime Contract

The adapter returns `GateDetection[]`. The normal localization pipeline remains:

```text
load_gate_detections_from_yolo(label_path)
    -> CoarseRefineLocalizer.refine(detections, kf_xyz, q_m)
    -> reject/injection policy
    -> OnlineLocalizer.inject_position_observation(...)
    -> KFLayer2
```

Rejected visual observations must not be injected into `OnlineLocalizer`.

## Current 30 FPS Localization Result

With current defaults:

```text
min_keypoint_confidence = 0.7
max_detections = 6
```

Processing result:

```text
total exported frames: 863
frames without YOLO label: 88
frames without usable detections after adapter: 554
frames passed to CoarseRefineLocalizer: 221
detections after adapter: 273

accepted by localization: 24
rejected by localization: 197

accepted impact against KFLayer2:
    improved: 13
    worsened: 11
    mean gain: -0.25 m
    median gain: +0.34 m
```

Rejected reason breakdown:

```text
too far from coarse prior: 185
not useful versus coarse prior: 8
q_out too high for injection: 4
```

Interpretation:

- the adapter/localizer already finds useful automatic observations;
- the accepted set is not yet safe enough for blind fusion;
- most accepted frames contain only one gate, where PnP is geometrically
  ambiguous;
- the next safety layer should focus on single-gate observations, temporal
  consistency, and jump rejection.

## Rejected Frames Analysis

Rejected frames are not all bad, but most of them are dangerous.

For the current `0.7` threshold:

```text
rejected frames with a computed pose: 197
would improve KFLayer2: 17
would worsen KFLayer2: 180

mean gain: -7.73 m
median gain: -6.99 m
best rejected helper: +5.74 m
worst rejected outlier: -35.12 m
```

Best rejected helpers:

```text
frame 127: 7.94 m -> 2.20 m, gain +5.74 m
frame 88: 10.00 m -> 4.85 m, gain +5.15 m
frame 85: 9.43 m -> 5.82 m, gain +3.61 m
frame 139: 4.23 m -> 0.69 m, gain +3.54 m
frame 84: 9.08 m -> 6.14 m, gain +2.94 m
```

Worst rejected outliers:

```text
frame 154: 1.26 m -> 36.38 m, loss -35.12 m
frame 649: 1.47 m -> 33.65 m, loss -32.18 m
frame 156: 1.30 m -> 33.00 m, loss -31.70 m
```

Conclusion: simply relaxing the reject gate is unsafe. If `KFLayer2` is wrong,
the camera may help, but this should require agreement across several adjacent
frames instead of trusting one rejected single-frame pose.

## Processing Speed

Benchmark scope:

```text
ready YOLO label -> yolo_gate_adapter -> CoarseRefineLocalizer
```

Not included:

```text
video frame extraction
YOLO inference on images
```

Full `863` frame dataset:

```text
total elapsed: 9.55 s
throughput over all exported frames: 90 FPS
throughput over frames sent to localizer: 23 FPS
```

Adapter speed:

```text
mean: 0.20 ms
median: 0.18 ms
p90: 0.30 ms
max: 0.72 ms
```

`CoarseRefineLocalizer` speed:

```text
mean: 42.23 ms
median: 15.91 ms
p90: 139.26 ms
p95: 154.79 ms
p99: 203.72 ms
max: 221.80 ms
```

By number of detections after adapter:

```text
1 gate:
    mean: 15.50 ms
    median: 15.41 ms
    max: 25.96 ms

2 gates:
    mean: 138.71 ms
    median: 136.00 ms
    max: 201.88 ms

3 gates:
    mean: 190.49 ms
    median: 201.96 ms
    max: 221.80 ms
```

Conclusion: `yolo_gate_adapter` is not the bottleneck. The expensive part is
`CoarseRefineLocalizer`, especially for `2-3` gates. Average throughput over the
whole exported set is above 30 FPS because many frames are skipped quickly, but
per-frame latency is not bounded enough for strict real-time use.

## Known Risk Cases

Even with `min_keypoint_confidence = 0.7`, the sample run still had one harmful
accepted frame:

```text
frame 242: KFLayer2 XZ error 3.18 m -> visual XZ error 6.10 m
```

This means confidence filtering is necessary but not sufficient. The next
filters should focus on:

- temporal consistency of selected `gate_id`;
- stronger ambiguity checks using `TopKHypothesis.confidence_to_next`;
- stricter handling of single-gate observations;
- rejecting observations that produce sudden jumps over adjacent frames.

## Example

```python
from gate_localization.coarse_refine import CoarseRefineLocalizer
from yolo_gate_adapter import YoloAdapterConfig, load_gate_detections_from_yolo

config = YoloAdapterConfig(min_keypoint_confidence=0.7, max_detections=6)
detections = load_gate_detections_from_yolo("frame_000242.txt", config)

localizer = CoarseRefineLocalizer(track_path="track.json")
result = localizer.refine(detections, coarse_position_world=kf_xyz, q_m=kf_q_m)
```
