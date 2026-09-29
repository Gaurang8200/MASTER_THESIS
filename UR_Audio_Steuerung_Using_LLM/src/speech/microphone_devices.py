# MO_Changes
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import speech_recognition as sr


GESTURE_ONLY_MICROPHONE = "No microphone, gesture only"


@dataclass(frozen=True)
class MicrophoneOption:
    device_index: int
    display_name: str


def build_microphone_mapping(
    options: Sequence[MicrophoneOption],
) -> dict[str, int | None]:
    mapping: dict[str, int | None] = {GESTURE_ONLY_MICROPHONE: None}
    mapping.update({option.display_name: option.device_index for option in options})
    return mapping


def start_optional_background_listener(
    device_index: int | None,
    callback: Callable[[sr.Recognizer, sr.AudioData], None],
    ambient_noise_seconds: float,
    phrase_time_limit: float,
) -> Callable[[bool], None] | None:
    if device_index is None:
        return None

    recognizer = sr.Recognizer()
    microphone = sr.Microphone(device_index=device_index)
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=ambient_noise_seconds)
    return recognizer.listen_in_background(
        microphone,
        callback,
        phrase_time_limit=phrase_time_limit,
    )


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
