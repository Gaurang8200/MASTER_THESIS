# MO_Changes
from __future__ import annotations

import unittest
from unittest.mock import patch

from src.speech.speech_to_text_local import SpeechToTextLocal


class SpeechToTextLocalTest(unittest.TestCase):
    @patch("src.speech.speech_to_text_local.subprocess.run")
    @patch(
        "src.speech.speech_to_text_local.shutil.which",
        return_value="/opt/homebrew/bin/ffmpeg",
    )
    def test_system_ffmpeg_is_verified_and_used(self, which, run) -> None:
        speech_to_text = SpeechToTextLocal()

        executable = speech_to_text.setup_ffmpeg_path()

        self.assertEqual(executable, "/opt/homebrew/bin/ffmpeg")
        run.assert_called_with(
            ["/opt/homebrew/bin/ffmpeg", "-version"],
            capture_output=True,
            check=True,
            timeout=5,
        )
        self.assertEqual(which.call_count, 2)

    @patch("src.speech.speech_to_text_local.os.name", "posix")
    @patch("src.speech.speech_to_text_local.shutil.which", return_value=None)
    def test_missing_ffmpeg_is_reported_before_whisper_runs(self, which) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "ffmpeg is required"):
            SpeechToTextLocal()

        which.assert_called_once_with("ffmpeg")


if __name__ == "__main__":
    unittest.main()
