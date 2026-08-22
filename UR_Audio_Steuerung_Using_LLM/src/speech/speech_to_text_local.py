# MO_Changes
import os
import shutil
import subprocess
from pathlib import Path

import whisper

from .base import SpeechToText
import speech_recognition as sr

class SpeechToTextLocal(SpeechToText):
    def __init__(self):
        """Initialize with ffmpeg path configuration"""
        self.setup_ffmpeg_path()
        
    def setup_ffmpeg_path(self) -> str:
        """Setup ffmpeg path for Whisper to work properly"""
        executable = shutil.which("ffmpeg")
        if executable is None and os.name == "nt":
            executable = self._enable_bundled_windows_ffmpeg()

        if executable is None:
            raise FileNotFoundError(
                "ffmpeg is required for Whisper transcription but was not found"
            )

        subprocess.run(
            [executable, "-version"],
            capture_output=True,
            check=True,
            timeout=5,
        )
        print(f"INFO: Using ffmpeg executable: {executable}")
        return executable

    @staticmethod
    def _enable_bundled_windows_ffmpeg() -> str | None:
        project_root = Path(__file__).resolve().parents[2]
        executable = (
            project_root
            / "tools"
            / "ffmpeg"
            / "ffmpeg-2025-05-05-git-f4e72eb5a3-full_build"
            / "bin"
            / "ffmpeg.exe"
        )
        if not executable.is_file():
            return None

        executable_directory = str(executable.parent)
        current_path = os.environ.get("PATH", "")
        path_entries = current_path.split(os.pathsep)
        if executable_directory not in path_entries:
            os.environ["PATH"] = os.pathsep.join(
                [executable_directory, current_path]
            )
        return str(executable)

    def transcribe(self, file_path: str) -> str:
        """ transkribiert lokal mittels Whisper
        Das Modell wird im Verzeichnis ./models/whisper gespeichert.
        """
        try:
            # Ensure ffmpeg is available
            self.setup_ffmpeg_path()
            
            # Überprüfe ob Audio-Datei existiert
            if not os.path.exists(file_path):
                print(f"ERROR: Audio file not found: {file_path}")
                return ""
            
            print(f"INFO: Transcribing audio file: {file_path}")
            
            # Erstelle Models-Verzeichnis falls es nicht existiert
            models_dir = "./models/whisper"
            os.makedirs(models_dir, exist_ok=True)
            
            # Lade das Modell "base" und speichere es im Ordner "./models/whisper"
            print("INFO: Loading Whisper model...")
            model = whisper.load_model("base", download_root=models_dir)
            
            print("INFO: Starting transcription...")
            result = model.transcribe(file_path, fp16=False)  # Disable FP16 for CPU
            
            transcribed_text = result.get("text", "").strip()
            print(f"SUCCESS: Transcription completed: '{transcribed_text}'")
            return transcribed_text
            
        except Exception as e:
            print(f"ERROR: Transcription failed: {e}")
            print(f"ERROR: Exception type: {type(e).__name__}")
            import traceback
            print(f"ERROR: Traceback: {traceback.format_exc()}")
            return ""

    def test_microphone(self, microphone_index=0):
        """
        Test microphone functionality
        """
        try:
            with sr.Microphone(device_index=microphone_index) as source:
                print(f"MICROPHONE: Testing microphone {microphone_index}")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("SUCCESS: Microphone test completed")
                return True
        except Exception as e:
            print(f"ERROR: Microphone test failed: {e}")
            return False
    
    def list_microphones(self):
        """
        List available microphones
        """
        try:
            mics = sr.Microphone.list_microphone_names()
            print("INFO: Available microphones:")
            for i, name in enumerate(mics):
                print(f"   {i}: {name}")
            return mics
        except Exception as e:
            print(f"ERROR: Could not list microphones: {e}")
            return []

    if __name__ == "__main__":
        # Beispiel-Code zur Transkription
        sample_audio = "data/audio/Abschlussarbeit_Sprachbefehle 1-Aufnahme 12.wav"
        text = transcribe(sample_audio)
        print("Transkribierter Text:", text)

