# MO_Changes
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import speech_recognition as sr


@dataclass(frozen=True)
class MicrophoneOption:
    device_index: int
    display_name: str


def discover_input_microphones() -> tuple[list[MicrophoneOption], int | None]:
    pyaudio = sr.Microphone.get_pyaudio()
    audio = pyaudio.PyAudio()

    try:
        default_index = _get_default_input_index(audio)
        options = []

        for device_index in range(audio.get_device_count()):
            device = audio.get_device_info_by_index(device_index)
            if int(device.get("maxInputChannels", 0)) < 1:
                continue

            device_name = str(device.get("name", f"Microphone {device_index}"))
            options.append(
                MicrophoneOption(
                    device_index=device_index,
                    display_name=f"{device_name} (Input {device_index})",
                )
            )

        available_indices = {option.device_index for option in options}
        if default_index not in available_indices:
            default_index = options[0].device_index if options else None

        return options, default_index
    finally:
        audio.terminate()


def _get_default_input_index(audio: Any) -> int | None:
    try:
        return int(audio.get_default_input_device_info()["index"])
    except (KeyError, OSError):
        return None
