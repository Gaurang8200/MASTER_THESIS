from abc import ABC, abstractmethod

class SpeechToText(ABC):
    @abstractmethod
    def transcribe(self, file_path: str) -> str:
        """Gibt den erkannten Text zurück"""
        pass

class InformationExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> dict:
        """Gibt Informationen zurück"""
        pass

