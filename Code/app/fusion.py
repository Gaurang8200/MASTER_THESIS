from .schemas import GestureResult, ASRResult, NLPResult, RobotCommand
from loguru import logger

def fuse_modalities(gesture: GestureResult, nlp: NLPResult) -> RobotCommand:
    # Example fusion logic
    if gesture.confidence < 0.5 or nlp.confidence < 0.5:
        logger.warning("Low confidence in gesture or NLP")
        return RobotCommand(action="unknown", object_id=None, target_location=None, confidence=0.0)
    return RobotCommand(
        action=nlp.action,
        object_id=gesture.object_id,
        target_location=nlp.target_location,
        confidence=(gesture.confidence + nlp.confidence) / 2
    )