import yaml
from pydantic import BaseModel, Field
from typing import Any, Dict
import os

class AppConfig(BaseModel):
    camera_id: int = 0
    mediapipe_model_path: str
    whisper_model_path: str
    nlp_model: str
    robot_ip: str
    confidence_threshold: float = 0.85
    log_level: str = "INFO"
    safety: Dict[str, Any] = Field(default_factory=dict)

def load_config(config_path: str = "config.yaml") -> AppConfig:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)