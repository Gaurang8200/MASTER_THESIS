# Pipeline

Runs the trained model on a camera frame, decides which object the operator is
pointing at, and hands the result to the existing pick and drop system. It never
moves the robot.

```
pipeline/
├── configs/
│   └── gesture_config.yaml             thresholds, stability windows, workspace, camera
├── run/                                the two entry points
│   ├── main_pipeline.py                full chain, JSON output, robot requests
│   └── webcam_gesture_demo.py          gesture layer only, for checking the model
├── detection/
│   └── gesture_detector.py             ultralytics YOLO adapter, image coordinates only
├── logic/                              the interaction rules
│   ├── gesture_state_machine.py        selection mode policy
│   ├── fingertip_selection.py          fingertip geometry, mask touch, object selection
│   └── place_point_selector.py         free table spot detection and pose conversion
├── integration/                        the three boundaries you will replace
│   ├── object_interface_mock.py        mock segmentation source, swap for the YOLOv5 repo
│   ├── existing_calibration_adapter.py wrapper for the existing pixel to robot function
│   └── robot_handoff.py                pick and place requests, robot mocked
└── support/
    ├── schemas.py                      data contracts and the JSON output
    ├── config.py                       typed configuration loader
    ├── camera.py                       webcam resource
    └── visualization.py                OpenCV overlays
```

Only the three files in `integration/` change when the real segmentation,
calibration and robot systems are connected. Nothing in `logic/` is touched.

## Interaction rules

Every confirmation is a five second hold. Seconds rather than frame counts, so a
slow machine behaves like a fast one.

1. `open_palm_start` held for five seconds turns selection mode on.
2. `closed_palm_stop` held for five seconds turns it off. The stop gesture wins
   whenever both palms appear in the same frame.
3. With selection mode on, `pointing_finger` and `index_fingertip` together
   produce a fingertip point at the center of the fingertip box.
4. A small probe circle around that point must overlap an object segmentation
   mask. A fingertip near an object but outside its mask selects nothing.
5. The same object must be touched for five seconds before it counts as
   selected.
6. A fingertip anywhere inside the table workspace becomes a place point after a
   five second hold. An object lying on that spot does not block it, because the
   operator is naming a destination and not picking something up. The pixel to
   robot conversion runs once, when the hold completes.
7. Selection mode off clears the selection and the place point.

## Run

Both need `models/best.pt` and a webcam.

Gesture layer only. The quickest way to check detection quality on a bare hand
and on a gloved hand, and to watch the activation streaks.

```bash
python run/webcam_gesture_demo.py
```

Full pipeline with the mock object source.

```bash
python run/main_pipeline.py
```

Show the pick and place requests that would go to the robot.

```bash
python run/main_pipeline.py --show-handoff
```

Keys in the windows: `q` quits, `r` resets.

## Output

One JSON object per frame. `mode` reports what the operator achieved and
`calibration_used` reports whether a robot coordinate is part of the result.

```json
{
  "mode": "object_selection",
  "selected_object_id": "obj_001",
  "fingertip_pixel": [332.8, 374.4],
  "calibration_used": false
}
```

```json
{
  "mode": "place_selection",
  "place_pixel": [537.6, 619.2],
  "place_robot_pose": [0.78384, 0.30728, 0.05, 180.0, 0.0, 0.0],
  "calibration_used": true
}
```

The full result adds the fingertip details, the selected object with its mask
overlap and pick pose, the latency, and `safe_to_execute`, which is the only
field a caller should gate a robot action on. It is false whenever inference
failed, the frame took longer than `model.latency_budget_ms`, or the place
conversion failed. `robot_command_dispatched` is always false in this module.

## Configuration

| key                                     | default | meaning                                    |
| --------------------------------------- | ------- | ------------------------------------------ |
| `model.weights`                         | ../models/best.pt | checkpoint from Ultralytics HUB  |
| `confidence.gesture`                    | 0.75    | palm and pointing detections               |
| `confidence.fingertip`                  | 0.65    | fingertip box, smaller so lower score      |
| `confidence.object`                     | 0.70    | objects allowed to take part in selection  |
| `stability.activate_seconds`            | 5.0     | open palm hold to turn the mode on         |
| `stability.deactivate_seconds`          | 5.0     | closed palm hold to turn the mode off      |
| `stability.select_seconds`              | 5.0     | contact hold before a selection            |
| `stability.place_seconds`               | 5.0     | hold before a place point                  |
| `stability.lost_gesture_timeout_seconds`| 3.0     | idle time before the mode falls back to off|
| `selection.fingertip_radius_px`         | 8       | probe circle, 0 tests the center pixel     |
| `selection.min_mask_overlap_px`         | 1       | mask pixels that count as a touch          |
| `selection.ignore_objects_for_place`    | true    | a place point may sit on top of an object  |
| `workspace.normalized_polygon`          | table   | valid place area, fractions of the frame   |
| `place_calibration.mode`                | mock    | mock placeholder or existing_repo          |
| `model.latency_budget_ms`               | 120     | above this a frame is reported degraded    |

Tune `model.latency_budget_ms` to the host. A CPU only machine runs slower than
the default budget and reports every frame as degraded.

## Integration points

### Object detection and segmentation

The pipeline depends only on the `ObjectSource` protocol in `schemas.py`.

```python
class ObjectSource(Protocol):
    def start(self) -> None: ...
    def get_objects(self, frame) -> Sequence[DetectedObject]: ...
    def close(self) -> None: ...
    def health(self) -> dict: ...
```

Write an adapter around the existing YOLOv5 detection and segmentation
repository that returns `DetectedObject` values with a boolean mask in the frame
resolution, then pass it in.

```python
from config import load_config
from main_pipeline import build_pipeline

config = load_config()
pipeline = build_pipeline(config, object_source=Yolov5SegmentationSource(...))
```

`object_id` has to stay stable across frames, otherwise the seven frame rule
never confirms. Objects without a mask in the frame resolution are skipped on
purpose, because a bounding box hit would select an object the finger only
hovers next to.

### Pixel to robot conversion for the place point

`place_calibration.mode: mock` uses a placeholder linear mapping so the
interaction can be developed without the robot. Setting it to `existing_repo`
loads `function_pool.py` and the calibration files of the existing repository
and returns the place pose in the robot base frame. That mode also needs
`pip install scipy sympy`, which `function_pool.py` imports.

### Robot pickup

`robot_handoff.py` turns a confirmed result into `pick` and `place` requests. It
dispatches nothing until a real client is passed in and dispatch is switched on.

```python
handoff = RobotHandoff(robot=existing_pick_and_drop_client, dispatch=True)
for request in handoff.handle(result):
    ...
```

The client only has to offer `pick(object_id)` and `place(pose)`.

## Safety

1. Selection mode off is the safe state. The stop gesture beats the start
   gesture and a failed inference never keeps a streak alive.
2. The checkpoint is rejected on load when its class order does not match
   `common/gesture_classes.py`.
3. Detections below the configured confidence never reach the selection logic.
4. A frame over the latency budget is marked degraded and blocks
   `safe_to_execute`.
5. A failed place conversion reports `place_calibration_failed` and drops the
   place point instead of returning a guessed coordinate.
6. The same selection is handed over once, so a stable selection held for many
   frames does not queue the same motion repeatedly.
7. The camera and the model session are released on error and on interrupt.
