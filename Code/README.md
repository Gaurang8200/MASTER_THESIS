# Multimodal Gesture and Voice Control for Humanoid-Industrial Robot Collaboration

## Overview

This project enables a Unitree humanoid robot to act as an intelligent, multimodal interface for controlling an industrial robot (e.g., UR cobot) using human gestures and voice commands.

- **Gesture recognition**: Detects pointing/stop gestures using MediaPipe and camera.
- **Voice recognition**: Converts speech to text using Whisper.
- **NLP**: Extracts intent and target from voice.
- **Fusion**: Combines gesture and voice into structured robot commands.
- **Safety**: Ensures safe operation with stop gestures/commands and confidence checks.
- **Robot interface**: Sends commands to industrial robot (UR, via ROS2/socket/REST).

## Folder Structure

See code for details.

## Quick Start

1. Install macOS dependencies with `pip install -r ../requirements-mac.txt`
   or Ubuntu x86 64 dependencies with
   `pip install -r ../requirements-linux-x86_64.txt`
2. Configure `config.yaml` and `.env`
3. Run main app:  
   `python -m app.main`

## Example Commands

- "Pick this"
- "Pick this and place it on station A"
- "Move this to the left"
- "Stop"

## Safety

- Stop gesture/command has highest priority.
- Low-confidence commands require confirmation.
- No command sent if confidence is below threshold.

## Extending

- Add new gestures or voice commands in `gesture/` and `nlp/`.
- Integrate with real robot in `app/robot_interface.py`.

## License

MIT
