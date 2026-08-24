# MO_Changes
"""Lifecycle manager for the gesture selection subprocess."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import IO


@dataclass(frozen=True)
class GestureSession:
    session_id: str
    started_at_unix_s: float
    result_file: Path
    request_file: Path
    ready_file: Path
    log_file: Path


class GestureProcessClient:
    """Starts gesture perception beside audio recording and stops it safely."""

    def __init__(self, audio_project_root: str | Path, display: bool = True) -> None:
        self._audio_root = Path(audio_project_root).resolve()
        repository_root = self._audio_root.parent
        self._service = (
            repository_root
            / "Code"
            / "gesture_selection_system"
            / "pipeline"
            / "run"
            / "speech_selection_service.py"
        )
        self._pipeline_root = self._service.parents[1]
        self._runtime_dir = self._audio_root / "data" / "multimodal"
        self._display = display
        self._process: subprocess.Popen[str] | None = None
        self._session: GestureSession | None = None
        self._log_handle: IO[str] | None = None

    @property
    def active(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def session(self) -> GestureSession | None:
        return self._session

    def start(
        self,
        selection_kind: str = "object",
        hold_seconds: float = 3.0,
    ) -> GestureSession:
        self.cancel()
        if selection_kind not in {"object", "location"}:
            raise ValueError(f"unsupported gesture selection kind {selection_kind}")
        if hold_seconds <= 0.0:
            raise ValueError("gesture hold time must be positive")
        if not self._service.is_file():
            raise FileNotFoundError(f"gesture service not found at {self._service}")

        self._runtime_dir.mkdir(parents=True, exist_ok=True)
        session_id = uuid.uuid4().hex
        session = GestureSession(
            session_id=session_id,
            started_at_unix_s=time.time(),
            result_file=self._runtime_dir / f"{session_id}_result.json",
            request_file=self._runtime_dir / f"{session_id}_request.json",
            ready_file=self._runtime_dir / f"{session_id}_ready.json",
            log_file=self._runtime_dir / f"{session_id}.log",
        )
        self._log_handle = session.log_file.open("w", encoding="utf8")
        command = [
            sys.executable,
            str(self._service),
            "--session-id",
            session.session_id,
            "--result-file",
            str(session.result_file),
            "--request-file",
            str(session.request_file),
            "--ready-file",
            str(session.ready_file),
            "--selection-kind",
            selection_kind,
            "--hold-seconds",
            str(hold_seconds),
        ]
        if not self._display:
            command.append("--no-display")
        self._process = subprocess.Popen(
            command,
            cwd=self._pipeline_root,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._session = session
        self._wait_until_ready(session, self._process)
        print(f"MULTIMODAL: Gesture session started {session.session_id}")
        return session

    def latest_result(self) -> dict[str, object] | None:
        session = self._session
        if session is None or not session.result_file.is_file():
            return None
        payload = self._read_result(session)
        if payload.get("status") != "selected" or not payload.get("safe_to_use"):
            return None
        return payload

    def finish(self, wait_seconds: float = 8.0) -> dict[str, object]:
        session = self._session
        process = self._process
        if session is None or process is None:
            return {
                "status": "error",
                "reason": "gesture_process_not_started",
                "safe_to_use": False,
            }

        self._write_request(session.request_file)
        try:
            process.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)

        self._close_log()
        payload = self._read_result(session)
        self._process = None
        self._session = None
        return payload

    def cancel(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
        self._close_log()
        self._process = None
        self._session = None

    @staticmethod
    def _write_request(path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps({"command": "stop"}), encoding="utf8")
        temporary.replace(path)

    @staticmethod
    def _read_result(session: GestureSession) -> dict[str, object]:
        if not session.result_file.is_file():
            return {
                "schema_version": "1.0",
                "session_id": session.session_id,
                "status": "error",
                "reason": "gesture_result_missing",
                "safe_to_use": False,
                "log_file": str(session.log_file),
            }
        try:
            payload = json.loads(session.result_file.read_text(encoding="utf8"))
        except (OSError, json.JSONDecodeError) as error:
            return {
                "schema_version": "1.0",
                "session_id": session.session_id,
                "status": "error",
                "reason": "gesture_result_invalid",
                "safe_to_use": False,
                "error": str(error),
                "log_file": str(session.log_file),
            }
        if payload.get("session_id") != session.session_id:
            return {
                "schema_version": "1.0",
                "session_id": session.session_id,
                "status": "error",
                "reason": "gesture_session_mismatch",
                "safe_to_use": False,
                "log_file": str(session.log_file),
            }
        if payload.get("schema_version") != "1.0":
            return {
                "schema_version": "1.0",
                "session_id": session.session_id,
                "status": "error",
                "reason": "gesture_contract_mismatch",
                "safe_to_use": False,
                "log_file": str(session.log_file),
            }
        payload["log_file"] = str(session.log_file)
        payload["session_started_at_unix_s"] = session.started_at_unix_s
        return payload

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _wait_until_ready(
        self,
        session: GestureSession,
        process: subprocess.Popen[str],
        timeout_seconds: float = 20.0,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if session.ready_file.is_file():
                return
            if process.poll() is not None:
                self._close_log()
                error = self._read_result(session)
                self._process = None
                self._session = None
                raise RuntimeError(
                    f"gesture process stopped during startup: {error.get('reason')}"
                )
            time.sleep(0.05)
        self.cancel()
        raise TimeoutError("gesture process did not become ready within 20 seconds")
