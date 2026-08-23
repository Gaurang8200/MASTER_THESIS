# Gesture selection system

Runtime gesture pipeline for the robot cell. The operator switches a selection
mode on, points at a segmented object to select it, and points at a free spot on
the table to define a place position. The result is structured JSON.

The model is trained outside this repository on Ultralytics HUB. This code only
loads the finished checkpoint.

```
gesture_selection_system/
├── common/     gesture classes, device selection, path helpers
├── models/     best.pt goes here, downloaded from Ultralytics HUB
└── pipeline/   detection, object selection, place point, robot handoff
```

| class id | name             | role                                     |
| -------- | ---------------- | ---------------------------------------- |
| 0        | open_palm_start  | turns the standalone selection mode on   |
| 1        | pointing_finger  | confirms that the hand is pointing       |
| 2        | index_fingertip  | carries the point used for the selection |

The same model covers bare hands and gloved hands. Glove support comes from the
training data, the runtime has no separate glove path.

The standalone gesture pipeline uses five second holds. The speech integration
uses a three second pointing hold because starting audio activates its session.

`common/gesture_classes.py` holds the class order. The pipeline checks the
downloaded checkpoint against it on load and refuses a mismatch.

## Steps

1. Record videos, extract frames, label them in CVAT, export the YOLO dataset,
   train the gesture detector on Ultralytics HUB and download the checkpoint.
2. Copy `best.pt` into `models/`.
3. Run the pipeline.

```bash
cd pipeline
python run/main_pipeline.py
```

## Where calibration is used

This system does not perform and does not require its own camera calibration.

| step                            | coordinates | calibration |
| ------------------------------- | ----------- | ----------- |
| open palm detection             | image       | no          |
| closed palm detection           | image       | no          |
| pointing finger detection       | image       | no          |
| index fingertip detection       | image       | no          |
| fingertip inside object mask    | image       | no          |
| place pixel to robot place pose | robot base  | yes         |

The fingertip point and the object masks live in the same image, so object
selection is a pure image space decision. Calibration appears in exactly one
step and is delegated to the existing pick and drop repository.

## Install

```bash
pip install ultralytics opencv-python numpy pydantic pyyaml
```

See `pipeline/README.md` for the interaction rules, the JSON contract and the
integration points.
