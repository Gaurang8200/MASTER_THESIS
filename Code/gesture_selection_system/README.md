# Gesture selection system

This folder contains only the gesture perception service used by the multimodal speech and robot application.

## Runtime connection

```text
UR_Audio_Steuerung_Using_LLM/application_multi_object.py
                    ↓
UR_Audio_Steuerung_Using_LLM/src/multimodal/gesture_client.py
                    ↓
pipeline/run/speech_selection_service.py
```

The service starts the camera, detects the pointing finger and fingertip, detects live object boxes, and returns one structured result to the voice application.

## Folder structure

```text
gesture_selection_system
│
├── models
│   └── handgestureyolov8m960100.pt
│
└── pipeline
    ├── configs
    │   └── gesture_config.yaml
    ├── detection
    │   ├── gesture_detector.py
    │   └── object_detector.py
    ├── logic
    │   └── fingertip_selection.py
    ├── run
    │   └── speech_selection_service.py
    └── support
        ├── camera.py
        ├── config.py
        ├── gesture_classes.py
        ├── schemas.py
        └── visualization.py
```

## Responsibilities

| Part | Responsibility |
| --- | --- |
| Gesture detector | Detects pointing finger and fingertip classes |
| Object detector | Loads the existing YOLOv5 object model and returns live boxes |
| Fingertip selection | Checks whether the fingertip centre is inside a detected object box |
| Camera | Opens the configured camera and maps rotated preview coordinates back to sensor coordinates |
| Speech selection service | Owns one gesture session and writes the final JSON result |

## Calibration ownership

This folder does not own robot calibration and does not convert pixels into robot coordinates.

The fingertip and object boxes are compared in image coordinates. Robot coordinate conversion remains in `UR_Audio_Steuerung_Using_LLM` so that Universal Robot and Franka continue using their existing calibration paths.

## Configuration

Runtime values are stored in `pipeline/configs/gesture_config.yaml`.

The active camera resolution is 1280 by 720. The gesture model uses an inference image size of 960. The object model uses an inference image size of 640. These model image sizes do not replace the camera coordinate system returned in the result.

## Main application

Run the complete multimodal workflow from the voice application folder.

```bash
python -m application_multi_object
```
