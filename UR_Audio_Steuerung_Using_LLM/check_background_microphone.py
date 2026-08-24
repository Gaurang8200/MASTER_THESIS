# MO_Changes
"""Check SpeechRecognition background capture with one physical microphone."""

from __future__ import annotations

import argparse
import faulthandler
import threading
from collections.abc import Callable, Sequence

import speech_recognition as sr


def run_diagnostic(
    device_index: int,
    calibration_seconds: float,
    timeout_seconds: float,
    phrase_time_limit: float,
) -> int:
    """Run the same microphone lifecycle used by the application."""
    recognizer = sr.Recognizer()
    microphone = sr.Microphone(device_index=device_index)
    captured = threading.Event()
    captured_bytes: list[int] = []
    stop_listening: Callable[..., None] | None = None

    def receive_audio(
        callback_recognizer: sr.Recognizer,
        audio: sr.AudioData,
    ) -> None:
        del callback_recognizer
        frame_count = len(audio.frame_data)
        captured_bytes.append(frame_count)
        print(f"BACKGROUND AUDIO CAPTURED: {frame_count} bytes", flush=True)
        captured.set()

    print(f"MICROPHONE: Using physical input {device_index}", flush=True)
    print("MICROPHONE: Opening for ambient calibration", flush=True)

    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(
                source,
                duration=calibration_seconds,
            )

        print("MICROPHONE: Starting background listener", flush=True)
        stop_listening = recognizer.listen_in_background(
            microphone,
            receive_audio,
            phrase_time_limit=phrase_time_limit,
        )
        print("MICROPHONE: Say something now", flush=True)
        captured.wait(timeout=timeout_seconds)
    except Exception as error:
        print(
            f"MICROPHONE ERROR: {type(error).__name__}: {error}",
            flush=True,
        )
        return 1
    finally:
        if stop_listening is not None:
            stop_listening(wait_for_stop=True)

    if not captured_bytes:
        print("MICROPHONE RESULT: No speech was captured", flush=True)
        return 2

    print("MICROPHONE RESULT: Background capture passed", flush=True)
    return 0


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Linux background microphone recording",
    )
    parser.add_argument("--device-index", type=int, default=3)
    parser.add_argument("--calibration-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--phrase-time-limit", type=float, default=5.0)
    parsed = parser.parse_args(arguments)
    if parsed.device_index < 0:
        parser.error("device index must not be negative")
    for name in (
        "calibration_seconds",
        "timeout_seconds",
        "phrase_time_limit",
    ):
        if getattr(parsed, name) <= 0.0:
            parser.error(f"{name.replace('_', ' ')} must be positive")
    return parsed


def main(arguments: Sequence[str] | None = None) -> int:
    faulthandler.enable()
    parsed = parse_arguments(arguments)
    return run_diagnostic(
        device_index=parsed.device_index,
        calibration_seconds=parsed.calibration_seconds,
        timeout_seconds=parsed.timeout_seconds,
        phrase_time_limit=parsed.phrase_time_limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
