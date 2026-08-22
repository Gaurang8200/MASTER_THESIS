# MO_Changes
from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from src.speech.microphone_devices import discover_input_microphones


class MicrophoneDeviceTest(unittest.TestCase):
    @patch("src.speech.microphone_devices.sr.Microphone.get_pyaudio")
    def test_only_input_devices_are_returned_and_default_is_selected(
        self,
        get_pyaudio: Mock,
    ) -> None:
        audio = self._configure_audio(
            get_pyaudio,
            devices=[
                {"name": "External Display", "maxInputChannels": 0},
                {"name": "USB Microphone", "maxInputChannels": 1},
                {"name": "MacBook Microphone", "maxInputChannels": 1},
            ],
            default_index=2,
        )

        options, default_index = discover_input_microphones()

        self.assertEqual([option.device_index for option in options], [1, 2])
        self.assertEqual(options[1].display_name, "MacBook Microphone (Input 2)")
        self.assertEqual(default_index, 2)
        audio.terminate.assert_called_once_with()

    @patch("src.speech.microphone_devices.sr.Microphone.get_pyaudio")
    def test_first_input_is_used_when_default_input_is_unavailable(
        self,
        get_pyaudio: Mock,
    ) -> None:
        self._configure_audio(
            get_pyaudio,
            devices=[
                {"name": "Display", "maxInputChannels": 0},
                {"name": "Microphone", "maxInputChannels": 1},
            ],
            default_index=0,
        )

        options, default_index = discover_input_microphones()

        self.assertEqual(len(options), 1)
        self.assertEqual(default_index, 1)

    @staticmethod
    def _configure_audio(
        get_pyaudio: Mock,
        devices: list[dict[str, object]],
        default_index: int,
    ) -> Mock:
        audio = Mock()
        audio.get_device_count.return_value = len(devices)
        audio.get_device_info_by_index.side_effect = devices.__getitem__
        audio.get_default_input_device_info.return_value = {"index": default_index}
        get_pyaudio.return_value.PyAudio.return_value = audio
        return audio


if __name__ == "__main__":
    unittest.main()
