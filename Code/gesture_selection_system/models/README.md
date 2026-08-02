# Models

Put the trained checkpoint here as `best.pt`.

```
models/best.pt
```

The file is not committed. Weights are large binaries and belong in the
experiment storage of the thesis, not in git.

## Where it comes from

The model is trained outside this repository.

1. Record videos of the gestures.
2. Extract frames.
3. Label the frames in CVAT with rectangles.
4. Export the YOLO dataset.
5. Upload it to Ultralytics HUB.
6. Train YOLO11s Detect.
7. Download `best.pt`.
8. Copy it into this folder.

## Class order

The order below is fixed. The pipeline checks it when the model loads and
refuses a checkpoint that was trained on a different order, because a silent
mismatch would read an open palm as a fingertip.

| class id | name             |
| -------- | ---------------- |
| 0        | open_palm_start  |
| 1        | closed_palm_stop |
| 2        | pointing_finger  |
| 3        | index_fingertip  |

Set the same order in CVAT and on Ultralytics HUB.

## Dataset advice

Train on bare hands and gloved hands in the same dataset. Glove support is a
property of the training data, the runtime has no separate glove path.

A frame of a pointing hand carries two boxes, `pointing_finger` around the hand
and `index_fingertip` around the tip only. The pipeline needs both.

## Using a different path

Point `model.weights` in `pipeline/configs/gesture_config.yaml` at any other
location if the checkpoint lives elsewhere.
