from app.schemas import NLPResult
from loguru import logger

class IntentParser:
    def __init__(self, config):
        # Load spaCy/transformers or rules
        pass

    def parse(self, text: str) -> NLPResult:
        # Placeholder: simple rule-based parsing
        if "stop" in text.lower():
            return NLPResult(action="stop", object_phrase=None, target_location=None, confidence=1.0)
        elif "pick" in text.lower() and "place" in text.lower():
            return NLPResult(action="pick_and_place", object_phrase="this", target_location="station_A", confidence=0.95)
        elif "pick" in text.lower():
            return NLPResult(action="pick", object_phrase="this", target_location=None, confidence=0.9)
        else:
            return NLPResult(action="unknown", object_phrase=None, target_location=None, confidence=0.0)