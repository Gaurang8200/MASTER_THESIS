from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = REPOSITORY_ROOT / "UR_Audio_Steuerung_Using_LLM"
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))

try:
    import speech_recognition  # noqa: F401
except ModuleNotFoundError:
    speech_recognition = types.ModuleType("speech_recognition")
    speech_recognition.AudioData = object
    speech_recognition.Microphone = object
    speech_recognition.Recognizer = object
    sys.modules["speech_recognition"] = speech_recognition

from src.speech.microphone_devices import (
    GESTURE_ONLY_MICROPHONE,
    MicrophoneOption,
    build_microphone_mapping,
    start_optional_background_listener,
)


class OptionalMicrophoneTests(unittest.TestCase):
    def test_gesture_only_is_always_the_safe_default_choice(self) -> None:
        mapping = build_microphone_mapping(
            [MicrophoneOption(device_index=4, display_name="USB microphone")]
        )

        self.assertIsNone(mapping[GESTURE_ONLY_MICROPHONE])
        self.assertEqual(mapping["USB microphone"], 4)

    def test_gesture_only_does_not_create_a_pyaudio_stream(self) -> None:
        with patch(
            "src.speech.microphone_devices.sr.Microphone",
            side_effect=AssertionError("microphone must not be opened"),
        ):
            stop_listener = start_optional_background_listener(
                None,
                lambda recognizer, audio: None,
                ambient_noise_seconds=0.5,
                phrase_time_limit=5.0,
            )

        self.assertIsNone(stop_listener)

    def test_selected_microphone_still_starts_background_capture(self) -> None:
        events: list[object] = []

        class FakeMicrophone:
            def __init__(self, device_index: int) -> None:
                events.append(("device", device_index))

            def __enter__(self) -> FakeMicrophone:
                return self

            def __exit__(self, *args: object) -> None:
                return None

        class FakeRecognizer:
            def adjust_for_ambient_noise(
                self,
                source: object,
                duration: float,
            ) -> None:
                events.append(("ambient", duration))

            def listen_in_background(
                self,
                source: object,
                callback: object,
                phrase_time_limit: float,
            ) -> object:
                events.append(("listen", phrase_time_limit))
                return lambda wait_for_stop: None

        with (
            patch(
                "src.speech.microphone_devices.sr.Microphone",
                FakeMicrophone,
            ),
            patch(
                "src.speech.microphone_devices.sr.Recognizer",
                FakeRecognizer,
            ),
        ):
            stop_listener = start_optional_background_listener(
                4,
                lambda recognizer, audio: None,
                ambient_noise_seconds=0.5,
                phrase_time_limit=5.0,
            )

        self.assertIsNotNone(stop_listener)
        self.assertEqual(events, [("device", 4), ("ambient", 0.5), ("listen", 5.0)])


if __name__ == "__main__":
    unittest.main()
