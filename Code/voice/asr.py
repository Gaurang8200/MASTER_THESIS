import whisper
from app.schemas import ASRResult
from loguru import logger

class ASR:
    def __init__(self, config):
        self.model = whisper.load_model(config.whisper_model_path)

    async def listen(self) -> ASRResult:
        # Placeholder: record audio, run ASR
        text = "Pick this and place it on station A"
        confidence = 0.95
        logger.info(f"ASR result: {text}")
        return ASRResult(text=text, confidence=confidence)