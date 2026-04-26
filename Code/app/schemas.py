from pydantic import BaseModel
from typing import Optional

class GestureResult(BaseModel):
    object_id: Optional[str]
    pointing_ray: Optional[list]
    confidence: float

class ASRResult(BaseModel):
    text: str
    confidence: float

class NLPResult(BaseModel):
    action: str
    object_phrase: Optional[str]
    target_location: Optional[str]
    confidence: float

class RobotCommand(BaseModel):
    action: str
    object_id: Optional[str]
    target_location: Optional[str]
    confidence: float

class SafetyStatus(BaseModel):
    safe: bool
    reason: Optional[str]