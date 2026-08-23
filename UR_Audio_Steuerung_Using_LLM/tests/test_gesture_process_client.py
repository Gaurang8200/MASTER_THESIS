# MO_Changes
from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from src.multimodal.gesture_client import GestureProcessClient


FAKE_SERVICE = """
import argparse
import json
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--session-id")
parser.add_argument("--result-file", type=Path)
parser.add_argument("--request-file", type=Path)
parser.add_argument("--ready-file", type=Path)
parser.add_argument("--no-display", action="store_true")
arguments = parser.parse_args()
arguments.ready_file.write_text(
    json.dumps({"session_id": arguments.session_id, "status": "ready"}),
    encoding="utf8",
)
for attempt in range(200):
    if arguments.request_file.exists():
        break
    time.sleep(0.01)
arguments.result_file.write_text(
    json.dumps(
        {
            "schema_version": "1.0",
            "session_id": arguments.session_id,
            "status": "rejected",
            "reason": "fingertip_outside_object_boxes",
            "safe_to_use": False,
        }
    ),
    encoding="utf8",
)
"""


class GestureProcessClientTest(unittest.TestCase):
    def test_ready_and_result_use_the_same_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            audio_root = repository / "UR_Audio_Steuerung_Using_LLM"
            service = (
                repository
                / "Code"
                / "gesture_selection_system"
                / "pipeline"
                / "run"
                / "speech_selection_service.py"
            )
            service.parent.mkdir(parents=True)
            audio_root.mkdir()
            service.write_text(textwrap.dedent(FAKE_SERVICE), encoding="utf8")

            client = GestureProcessClient(audio_root, display=False)
            session = client.start()
            result = client.finish()

        self.assertEqual(result["session_id"], session.session_id)
        self.assertEqual(result["schema_version"], "1.0")
        self.assertEqual(result["reason"], "fingertip_outside_object_boxes")
        self.assertFalse(result["safe_to_use"])


if __name__ == "__main__":
    unittest.main()
