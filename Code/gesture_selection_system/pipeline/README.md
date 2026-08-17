# Pipeline

Runs the gesture model on a camera frame, decides which object the operator is
pointing at, and hands the result to the existing pick and drop system.

Nothing here is simulated. The objects come from the YOLOv5 detector of the
existing repository, the place coordinate comes from its calibration files, and
the robot is the real UR controller. Motion is off until it is switched on.

```
pipeline/
├── configs/
│   └── gesture_config.yaml       thresholds, holds, workspace, camera, robot
├── run/                          the two entry points
│   ├── main_pipeline.py          full chain, JSON output, robot handoff
│   └── webcam_gesture_demo.py    gesture layer only, for checking the model
├── detection/
│   ├── gesture_detector.py       YOLO11s gesture model
│   └── object_detector.py        YOLOv5 detector of the existing repository
├── logic/                        the interaction rules
│   ├── gesture_state_machine.py  selection mode policy
│   ├── fingertip_selection.py    fingertip geometry and object selection
│   └── place_point_selector.py   table spot detection and pose conversion
├── integration/                  the boundaries to the other systems
│   ├── calibration.py            pixel to robot, calls function_pool directly
│   └── robot_handoff.py          pick pixel and place pose to the UR controller
└── support/
    ├── schemas.py                data contracts and the JSON output
    ├── config.py                 typed configuration loader
    ├── gesture_classes.py        the four classes, single source of truth
    ├── camera.py                 webcam resource
    └── visualization.py          OpenCV overlays
```

## Interaction rules

Every confirmation is a five second hold. Seconds rather than frame counts, so a
slow machine behaves like a fast one.

1. `open_palm_start` held for five seconds turns selection mode on.
2. `closed_palm_stop` held for five seconds turns it off. The stop gesture wins
   whenever both palms appear in the same frame.
3. With selection mode on, `pointing_finger` and `index_fingertip` together
   produce a fingertip point at the center of the fingertip box.
4. The fingertip selects the smallest object box it sits inside. The existing
   detector reports boxes and no segmentation masks, so selection is box
   containment. Set `selection.max_center_distance_ratio` to also require the
   fingertip to be near the box center.
5. The same object has to be held for five seconds before it counts as selected.
6. A fingertip anywhere inside the table workspace becomes a place point after a
   five second hold. An object lying on that spot does not block it, because the
   operator is naming a destination. The pixel to robot conversion runs once,
   when the hold completes.
7. Selection mode off clears the selection and the place point.

## Where calibration is used

| step                            | coordinates | calibration |
| ------------------------------- | ----------- | ----------- |
| gesture classification          | image       | no          |
| fingertip inside an object box  | image       | no          |
| place pixel to robot place pose | robot base  | yes         |

## Run

Needs `models/best.pt`, the YOLOv5 weights of the existing repository, and a
webcam.

Gesture layer only, the quickest way to check detection on a bare hand and on a
gloved hand.

```bash
python run/webcam_gesture_demo.py
```

Full chain.

```bash
python run/main_pipeline.py
```

Also print the requests that would go to the robot.

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
  "place_pixel": [768.0, 576.0],
  "place_robot_pose": [0.366, -0.146, 0.05, 2.221, 2.221, 0.0],
  "calibration_used": true
}
```

The pose is metres and a rotation vector in radians, the form `movel(p[...])`
takes. Only x and y come from the transform, which is what `pixel2robot.py`
writes to `robot_coordinates.txt`. The release height and the orientation are
configured, the same way the existing `final_position` sets them.

`safe_to_execute` is the only field a caller should gate a robot action on. It is
false whenever inference failed, the frame took longer than
`model.latency_budget_ms`, or the place conversion failed.

## How the handoff works

The two actions leave by different routes, because that is how the existing
system is built.

**Pick** carries the pixel of the selected object. The existing pipeline already
turns a pixel into a grasp through `pixel2robot.py`, `pca.py` and `direction.py`,
and it reads that pixel from `txt_file/center_point.txt`. Writing the same file
replaces the detection step of that pipeline and leaves its grasp logic alone.
The pixel is scaled into the calibration resolution first, exactly as
`convert_origin_for_robot` in the existing `detection.py` does.

**Place** carries the coordinate the same calibration produced. It is written to
`txt_file/place_coordinates.txt` in the shape `pixel2robot.py` writes
`robot_coordinates.txt`, and it is sent as UR script on port 30002, the same port
and the same move `final_position` uses.

```
movel(p[0.366, -0.146, 0.05, 2.221, 2.221, 0.0], a=0.1, v=0.1)
```

Nothing is sent while `robot.dispatch` is false.

## Configuration

| key                                     | default | meaning                                    |
| --------------------------------------- | ------- | ------------------------------------------ |
| `model.weights`                         | ../models/best.pt | gesture checkpoint from Ultralytics HUB |
| `object_model.repo_path`                | the yolov5 folder | source the weights are loaded with |
| `object_model.track_min_iou`            | 0.5     | overlap that keeps an object id across frames |
| `robot.dispatch`                        | false   | motion stays off until switched on         |
| `confidence.gesture`                    | 0.75    | palm and pointing detections               |
| `confidence.fingertip`                  | 0.65    | fingertip box, smaller so lower score      |
| `confidence.object`                     | 0.70    | objects allowed to take part in selection  |
| `stability.activate_seconds`            | 5.0     | open palm hold to turn the mode on         |
| `stability.deactivate_seconds`          | 5.0     | closed palm hold to turn the mode off      |
| `stability.select_seconds`              | 5.0     | hold on an object before it is selected    |
| `stability.place_seconds`               | 5.0     | hold before a place point                  |
| `stability.lost_gesture_timeout_seconds`| 3.0     | idle time before the mode falls back to off|
| `selection.max_center_distance_ratio`   | empty   | optional tighter rule around the box center|
| `workspace.normalized_polygon`          | table   | valid place area, fractions of the frame   |
| `place_calibration.pose_index`          | 15      | same index the existing pixel2robot.py uses|
| `place_calibration.calibration_resolution` | 2560 by 1472 | resolution the calibration was recorded at |
| `place_calibration.place_z_m`           | 0.05    | release height, existing values are 0.040 to 0.067 |
| `place_calibration.place_orientation`   | 2.221, 2.221, 0.0 | tool orientation, radians, same as final_position |
| `model.latency_budget_ms`               | 120     | above this a frame is reported degraded    |

Tune `model.latency_budget_ms` to the host. Two models run per frame, so a CPU
only machine will report every frame as degraded at the default value.

## Prerequisites

1. `models/best.pt`, the gesture checkpoint.
2. The YOLOv5 source in the existing repository, the folder that holds
   `hubconf.py`, plus its trained weights.
3. `pip install scipy sympy`, which `function_pool.py` imports.

## Safety

1. Selection mode off is the safe state. The stop gesture beats the start gesture
   and a failed inference never keeps a hold alive.
2. The gesture checkpoint is rejected on load when its class order does not match
   `support/gesture_classes.py`.
3. Detections below the configured confidence never reach the selection logic.
4. A frame over the latency budget is marked degraded and blocks
   `safe_to_execute`.
5. A failed place conversion drops the place point instead of returning a guessed
   coordinate.
6. Each selection and each place point is handed over once.
7. A place is refused until the matching pick has been handed over.
8. A failed transport is logged and reported as not dispatched, never raised into
   the interaction loop.
9. The camera and both model sessions are released on error and on interrupt.
