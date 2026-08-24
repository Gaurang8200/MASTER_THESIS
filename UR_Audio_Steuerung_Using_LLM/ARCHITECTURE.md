# UR Audio Steering Using LLM Architecture

## Purpose

This project lets an operator detect several objects, select one through speech or pointing, and move it with either Universal Robot or Franka.

The application combines seven technical areas.

1. A Tkinter desktop interface
2. YOLOv5 object detection
3. Local Whisper speech recognition
4. OpenAI command interpretation
5. Pointing finger and fingertip detection
6. Universal Robot motion and vacuum control
7. Franka motion and gripper control

## Complete system graph

```mermaid
flowchart TD
    Operator["Operator"]
    GUI["application_multi_object.py<br/>Tkinter interface and workflow state"]
    OverviewCamera["Overview camera or simulation image"]
    Detection["detection_multi.py<br/>YOLOv5 overview detection"]
    DetectionData["detected_objects.json<br/>classes, boxes, centers, confidence"]
    Microphone["Microphone"]
    Audio["data/audio/live_audio.wav"]
    Whisper["SpeechToTextLocal<br/>local Whisper base model"]
    Transcript["Transcribed German or English command"]
    OpenAI["InformationExtractionOpenAIMulti<br/>OpenAI command interpretation"]
    Command["Structured command JSON<br/>object, index, target zone, action"]
    GestureCamera["Live gesture camera"]
    GestureModel["Pointing finger and fingertip model"]
    LiveObjects["YOLOv5 live object boxes"]
    GestureResult["Fresh gesture session JSON"]
    Resolver["Speech and gesture object resolver"]
    Confirmation["Spoken yes confirmation"]
    Selector["robot_method_selector_multi.py<br/>object and workflow selection"]
    Methods["Ordered robot method list"]
    Verification["Verification object detection"]
    RobotChoice{"Selected robot"}
    URControl["robot_control.py<br/>Universal Robot execution"]
    FrankaControl["src/franka/workflow.py<br/>Franka execution"]
    LegacyVision["Code-YOLOv5-Windows_llm<br/>precision detection and calibration"]
    URRobot["Universal Robot<br/>URScript and vacuum tool"]
    FrankaRobot["Franka<br/>franky and gripper"]

    Operator --> GUI
    GUI --> OverviewCamera
    OverviewCamera --> Detection
    Detection --> DetectionData
    DetectionData --> GUI
    Operator --> Microphone
    Microphone --> Audio
    Audio --> Whisper
    Whisper --> Transcript
    Transcript --> OpenAI
    DetectionData --> OpenAI
    OpenAI --> Command
    GUI --> GestureCamera
    GestureCamera --> GestureModel
    GestureCamera --> LiveObjects
    GestureModel --> GestureResult
    LiveObjects --> GestureResult
    GestureResult --> Resolver
    DetectionData --> Resolver
    Command --> Resolver
    Resolver --> Selector
    Command --> Selector
    DetectionData --> Selector
    Selector --> Methods
    Methods --> Confirmation
    Operator --> Confirmation
    Confirmation --> Verification
    Verification --> RobotChoice
    RobotChoice -->|"Universal Robot"| URControl
    RobotChoice -->|"Franka"| FrankaControl
    URControl --> LegacyVision
    FrankaControl --> LegacyVision
    LegacyVision --> URControl
    LegacyVision --> FrankaControl
    URControl --> URRobot
    FrankaControl --> FrankaRobot
```

## Operator workflow

```mermaid
stateDiagram-v2
    [*] --> ReadyForDetection
    ReadyForDetection --> ProcessingDetection: Capture objects
    ProcessingDetection --> ReadyForCommands: Objects found
    ProcessingDetection --> ReadyForDetection: No valid objects
    ReadyForCommands --> RecordingAndPointing: Start interaction
    RecordingAndPointing --> ProcessingCommand: Stop recording or stable ten second point
    ProcessingCommand --> ReadyForCommands: Clarification required
    ProcessingCommand --> AwaitingConfirmation: Valid object and command
    AwaitingConfirmation --> ReadyForCommands: No or silence
    AwaitingConfirmation --> ReadyForExecution: Spoken yes
    ReadyForExecution --> Simulating: Execute in simulation mode
    ReadyForExecution --> ExecutingRobot: Execute in real mode
    Simulating --> AwaitingDestination: Pick complete without target
    ExecutingRobot --> AwaitingDestination: Pick complete without target
    AwaitingDestination --> DestinationConfirmation: Zone or point selected
    AwaitingDestination --> AwaitingDestination: Reminder after two minutes
    DestinationConfirmation --> AwaitingDestination: No or silence
    DestinationConfirmation --> Simulating: Spoken yes in simulation
    DestinationConfirmation --> ExecutingRobot: Spoken yes in real mode
    Simulating --> ReadyForCommands: Simulation complete
    ExecutingRobot --> ReadyForCommands: Robot workflow complete
    ExecutingRobot --> ReadyForExecution: Robot execution error
```

The interface starts in simulation mode. Real robot execution must be selected explicitly in the interface.

## Speech and language model pipeline

```mermaid
sequenceDiagram
    actor User as Operator
    participant GUI as application_multi_object.py
    participant Mic as SpeechRecognition
    participant Gesture as Gesture subprocess
    participant Whisper as SpeechToTextLocal
    participant LLM as InformationExtractionOpenAIMulti
    participant Files as detected_objects.json
    participant Selector as robot_method_selector_multi.py

    User->>GUI: Start recording
    par Speech path
        GUI->>Mic: Start background recording
        Mic-->>GUI: Captured audio phrases
    and Gesture path
        GUI->>Gesture: Start camera and both live models
        Gesture-->>GUI: Fresh pointed object result after five seconds
    end
    alt User stops recording
        User->>GUI: Stop recording
    else Gesture remains stable for ten seconds
        Gesture-->>GUI: Finish automatically
    end
    opt Speech was captured
        GUI->>GUI: Save live_audio.wav
        GUI->>Whisper: Transcribe audio file
        Whisper-->>GUI: Command text
        GUI->>LLM: Extract structured command
        LLM->>Files: Load currently available objects
        Files-->>LLM: Object classes and counts
        LLM-->>GUI: Intent object index action and target zone
    end
    GUI->>GUI: Match gesture result with overview objects when pointing was used
    GUI->>Selector: Select complete or pick only precision workflow
    Selector->>Files: Find the requested object instance
    Selector-->>GUI: Ordered robot method names
    GUI-->>User: Speak proposed movement
    User->>GUI: Say yes
```

### Concrete command example

The operator says:

```text
Bewege den zweiten Marker zu Zone 2
```

Whisper produces text. The OpenAI extractor is expected to return data shaped like this:

```json
{
  "intent": "bewegen",
  "target_location": "Zone_2",
  "action": "greife Marker, bewege zu Zone_2 und lasse los",
  "object": "Marker",
  "object_index": 1,
  "selection_mode": "speech",
  "needs_clarification": false,
  "clarification_fields": [],
  "command_type": "new",
  "reference_index": null
}
```

The object index starts at zero. Index `1` therefore selects the second detected Marker.

## Object detection and selection

```mermaid
flowchart TD
    Mode{"Execution mode"}
    TestImage["test_photo/test_photo.jpg"]
    Camera["Real camera"]
    MultiDetection["detection_multi.py"]
    YOLO["YOLOv5 model weights"]
    Results["txt_file/detected_objects.json"]
    Display["Detected object display"]
    Command["Object name and object index"]
    Filter["select_target_object"]
    Selected["Selected object<br/>id, class, box, center, confidence"]
    Compatibility["Legacy compatibility files"]

    Mode -->|Simulation| TestImage
    Mode -->|Real| Camera
    TestImage --> MultiDetection
    Camera --> MultiDetection
    YOLO --> MultiDetection
    MultiDetection --> Results
    Results --> Display
    Results --> Filter
    Command --> Filter
    Filter --> Selected
    Selected --> Compatibility
```

The detection result contains an array named `objects`. Each object is expected to provide these values.

1. `id` identifies the individual detection
2. `class` is the numerical YOLO class
3. `class_name` is the readable object name
4. `confidence` is the model confidence
5. `bbox` contains the bounding box
6. `center` contains the image center pixel

The application supports Cylinder, Box, and Marker names. German alternatives are translated before object validation.

## Precision robot workflow

```mermaid
flowchart TD
    Start["Start precision workflow"]
    Main["move_to_main_position"]
    Detect["detect_object"]
    Pixel["convert_pixel_to_robot"]
    Approach["move_to_selected_object"]
    DetectAgain["precision_detection"]
    Filter["filter_and_prepare_selected_object_after_precision_detection"]
    PCA["precision_pca_calculation"]
    Direction["precision_direction_object"]
    Pick["pick_the_object"]
    VacuumOn["suction_on"]
    Lift["pick_up_object"]
    IntermediateA["intermediate_position"]
    TargetKnown{"Target already known"}
    WaitDestination["Parallel destination speech and pointing"]
    Zone["move_to_target Zone"]
    ChosenZone["move_to_target selected Zone"]
    Point["Calibrate fingertip u and v to robot x and y"]
    DynamicTarget["move_to_point x and y"]
    ConfirmDestination{"Spoken yes"}
    Place["final_position"]
    VacuumOff["suction_off"]
    IntermediateB["intermediate_position"]
    Return["move_to_main_position"]
    Cleanup["delet_txt_file"]

    Start --> Main
    Main --> Detect
    Detect --> Pixel
    Pixel --> Approach
    Approach --> DetectAgain
    DetectAgain --> Filter
    Filter --> PCA
    PCA --> Direction
    Direction --> Pick
    Pick --> VacuumOn
    VacuumOn --> Lift
    Lift --> IntermediateA
    IntermediateA --> TargetKnown
    TargetKnown -->|"Yes"| Zone
    TargetKnown -->|"No"| WaitDestination
    WaitDestination -->|"Named zone"| ChosenZone
    WaitDestination -->|"Drop here or ten second point"| Point
    Point --> DynamicTarget
    Zone --> Place
    ChosenZone --> ConfirmDestination
    DynamicTarget --> ConfirmDestination
    ConfirmDestination -->|"Yes"| Place
    ConfirmDestination -->|"No or silence"| WaitDestination
    Place --> VacuumOff
    VacuumOff --> IntermediateB
    IntermediateB --> Return
    Return --> Cleanup
```

The second detection happens after the robot has moved closer to the selected object. It refines object identity, position, and orientation before the picking movement.

## Vision to robot coordinate communication

```mermaid
flowchart LR
    Detection["YOLO detection"]
    Center["center_point.txt<br/>image pixel"]
    Calibration["pixel2robot_multi.py<br/>camera calibration"]
    Coordinates["robot_coordinates.txt<br/>robot x and y"]
    Selection["selection_data.json<br/>selected object identity"]
    Move["move_to_selected_object"]
    Precision["final object files"]
    Orientation["robot_RPY.txt<br/>object orientation"]
    Pick["pick_the_object"]

    Detection --> Center
    Center --> Calibration
    Calibration --> Coordinates
    Selection --> Move
    Coordinates --> Move
    Move --> Precision
    Precision --> Orientation
    Coordinates --> Pick
    Orientation --> Pick
```

The `txt_file` directory acts as a compatibility boundary between the newer speech application and the older YOLO and robot scripts.

Important files include:

1. `detected_objects.json` stores all detected objects
2. `selection_data.json` stores the selected object identity
3. `center_point.txt` stores the selected image pixel
4. `label.txt` stores class, box, and confidence data
5. `crop_img_path.txt` points to the selected object crop
6. `robot_coordinates.txt` stores robot coordinates
7. `robot_RPY.txt` stores the picking orientation
8. `final_object_label.txt` stores the refined class
9. `final_object_center_point.txt` stores the refined center
10. `final_object_crop_path.txt` points to the refined crop

## Simulation and real robot paths

```mermaid
flowchart TD
    Methods["Robot method list"]
    Choice{"Execution mode"}
    Simulation["execute_robot_workflow_simulation"]
    SimulationOutput["Show intended positions and URScript in GUI"]
    Real["execute_robot_workflow"]
    Dispatcher["execute_robot_method"]
    URScript["movej, movel, digital outputs"]
    Port30002["TCP port 30002<br/>send URScript"]
    Port30003["TCP port 30003<br/>read robot state"]
    Robot["Universal Robots controller"]

    Methods --> Choice
    Choice -->|Simulate| Simulation
    Simulation --> SimulationOutput
    Choice -->|Real robot| Real
    Real --> Dispatcher
    Dispatcher --> URScript
    URScript --> Port30002
    Port30002 --> Robot
    Robot --> Port30003
    Port30003 --> Dispatcher
```

Simulation mode does not open a robot socket. It prints the movement and URScript that would have been sent.

Real mode sends URScript to port `30002`. Robot position and joint feedback are read through port `30003`.

The configured robot address is:

```text
192.168.2.180
```

## Robot zones

```mermaid
flowchart LR
    Command["Target zone from spoken command"]
    Lookup["zone_coordinates.py"]
    Zone1["Zone_1<br/>x 0.366<br/>y minus 0.146"]
    Zone2["Zone_2<br/>x 0.366<br/>y minus 0.008"]
    Zone3["Zone_3<br/>x 0.366<br/>y 0.121"]
    Height["Object class height"]
    Final["final_position"]

    Command --> Lookup
    Lookup --> Zone1
    Lookup --> Zone2
    Lookup --> Zone3
    Zone1 --> Final
    Zone2 --> Final
    Zone3 --> Final
    Height --> Final
```

The final `z` value depends on the detected object class.

1. Cylinder class `0` uses `0.067` metres
2. Box class `1` uses `0.048` metres
3. Marker class `2` uses `0.040` metres

## Core file responsibilities

### `application_multi_object.py`

This is the main entry point and desktop interface.

It controls workflow state, object detection, microphone recording, transcription, OpenAI extraction, object selection, command history, simulation selection, and real robot execution.

### `src/speech/base.py`

This defines the abstract interfaces for speech transcription and information extraction.

### `src/speech/speech_to_text_local.py`

This loads the local Whisper `base` model and converts a recorded WAV file into text. It configures FFmpeg when the command is not already available.

### `src/speech/information_extraction_openai_api_multi.py`

This sends the transcript and detected object context to the OpenAI API. It returns structured command information, validates object availability, supports German and English names, requests clarification when data is missing, and stores up to five recent commands.

### `src/robot_method_selector_multi.py`

This selects the requested object instance and creates the ordered precision workflow. A known target creates the complete pick and place list. A missing target creates the pick phase that stops at the intermediate position.

### `src/multimodal/gesture_client.py`

This owns the gesture subprocess lifecycle and reads fresh object or destination results for the active session.

### `src/multimodal/selection.py`

This recognizes pointing language, matches a live pointed object to the overview detection, and recognizes yes, no, zone, and drop here speech.

### `src/multimodal/feedback.py`

This prints interaction messages and speaks them through the operating system speech command.

### `src/robot_control.py`

This is the robot execution layer. It contains simulation behavior, URScript generation, socket communication, robot feedback reads, vacuum control, movement functions, precision detection calls, coordinate conversion, object orientation, method dispatch, and complete workflow execution.

### `src/franka/workflow.py`

This maps the same ordered robot methods to Franka motions. It keeps one Franka connection open while the object is held at the intermediate position and uses the original Franka calibration for a pointed destination.

### `src/zone_coordinates.py`

This maps named zones to robot coordinates and maps object classes to placement heights.

### `Code-YOLOv5-Windows_llm`

This is the legacy vision and calibration subproject. The main application starts its detection scripts and exchanges data through its `txt_file` directory.

Its most important integration scripts are:

1. `detection_multi.py` detects all visible objects
2. `detection_multi_precision_run.py` performs the closer second detection
3. `pixel2robot_multi.py` converts selected pixels into robot coordinates
4. `pca_multi.py` estimates object geometry
5. `direction_multi.py` calculates the picking orientation
6. `function_pool.py` contains camera and robot calibration mathematics
7. `yolov5/detect_multi_objects.py` performs the modified YOLOv5 inference

### `requirements-mac.txt`

This declares the complete macOS environment.

### `requirements-linux-x86_64.txt`

This declares the complete Ubuntu x86_64 environment.

### `.env`

This supplies the `OPENAI_API_KEY`. The extractor stops during import when this value is missing.

### `ip_roboter.txt`

This is a legacy Universal Robot address file. The current interface uses `192.168.2.180` for Universal Robot and `172.16.0.2` for Franka by default.

### `run_guide.txt`

This records the Windows startup procedure, camera permission requirement, robot address check, and local FFmpeg path command.

## Folder ownership

```mermaid
flowchart TD
    Root["UR_Audio_Steuerung_Using_LLM"]
    App["application_multi_object.py<br/>application orchestration"]
    Src["src<br/>speech, selection, zones, robot control"]
    Legacy["Code-YOLOv5-Windows_llm<br/>YOLOv5, calibration, precision vision"]
    Data["data<br/>audio and command history"]
    Models["models<br/>downloaded Whisper model"]
    Tools["tools<br/>local FFmpeg and FLAC binaries"]
    Environment["venv_wrapper<br/>local Python environment"]
    TestData["simulation_test_data<br/>simulation inputs"]

    Root --> App
    Root --> Src
    Root --> Legacy
    Root --> Data
    Root --> Models
    Root --> Tools
    Root --> Environment
    Root --> TestData
```

The source of truth for application behavior is the top level application plus `src`. The legacy folder owns object detection and camera to robot calculations. Model files, generated audio, environments, logs, and tool binaries are runtime support rather than application source.

## Main integration boundaries

1. The microphone boundary produces a WAV file
2. The Whisper boundary produces plain text
3. The OpenAI boundary produces structured command data
4. The YOLO boundary produces detected object JSON
5. The selector boundary produces method names
6. The calibration boundary produces robot coordinates
7. The robot boundary accepts URScript and returns robot state

Keeping these boundaries explicit makes it possible to replace Whisper, OpenAI, YOLOv5, or the Universal Robots interface independently.
