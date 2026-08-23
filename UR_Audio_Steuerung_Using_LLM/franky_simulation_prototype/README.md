# Franky simulation prototype

## Purpose

This folder runs a visible Franka FR3 and gripper through Franky without ROS 2.
It does not change the current audio, vision, calibration, or robot application.

The prototype reuses the existing Franka configuration, pose model, robot
interface, and `FrankaRobotArm` movement methods.

## What appears in the viewer

1. A Franka FR3 with seven moving joints

2. A working two finger gripper

3. A red cylinder at X `0.400`, Y `0.200`

4. A blue target marker at the current `Zone_1` coordinates

5. A table with MuJoCo contact physics

## Movement sequence

1. Move to the configured Franka home joints

2. Move to the current camera approach pose

3. Remove the camera offset and descend to the cylinder

4. Close the Franka gripper

5. Lift the cylinder

6. Move through the configured intermediate joints

7. Move to `Zone_1`

8. Release the cylinder

9. Return to the home joints

## Installation

Run this command from `UR_Audio_Steuerung_Using_LLM` with the project virtual
environment active.

```bash
python -m pip install -r franky_simulation_prototype/requirements.txt
```

## Preview the three dimensional workcell on macOS

```bash
mjpython -m franky_simulation_prototype.scene_preview
```

This command displays the FR3, hand, fingers, table, cylinder, and target zone.
It does not move the robot without Franky.

The official `franky_control` package currently provides Linux wheels for Intel
processors. On Apple Silicon, the MuJoCo scene and robot body can be validated,
but the complete Franky command connection requires a supported Linux machine
or a separately compiled Apple Silicon Franky package.

The prototype does not replace Franky with direct MuJoCo movement commands when
Franky is unavailable. This keeps the simulation and physical robot command
path the same.

## Run the complete Franky motion on supported Linux

```bash
python -m franky_simulation_prototype
```

## Run without the viewer

```bash
python -m franky_simulation_prototype --headless
```

The simulated robot address is created locally. Franky sends the same motion
and gripper commands to this local address that it sends to a physical robot.
