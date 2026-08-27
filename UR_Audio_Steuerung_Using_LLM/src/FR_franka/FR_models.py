# MO_Changes
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence


def _finite_tuple(values: Sequence[float], expected_size: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != expected_size:
        raise ValueError(f"{name} requires {expected_size} values")
    if not all(isfinite(value) for value in result):
        raise ValueError(f"{name} contains a nonfinite value")
    return result


@dataclass(frozen=True)
class CartesianPose:
    translation: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]

    @classmethod
    def create(
        cls,
        translation: Sequence[float],
        quaternion: Sequence[float],
    ) -> "CartesianPose":
        checked_translation = _finite_tuple(translation, 3, "translation")
        checked_quaternion = _finite_tuple(quaternion, 4, "quaternion")
        norm = sum(value * value for value in checked_quaternion) ** 0.5
        if norm < 1e-9:
            raise ValueError("quaternion must have a nonzero norm")
        normalized = tuple(value / norm for value in checked_quaternion)
        return cls(checked_translation, normalized)


@dataclass(frozen=True)
class PixelPoint:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not isfinite(self.x) or not isfinite(self.y):
            raise ValueError("pixel coordinates must be finite")


@dataclass(frozen=True)
class RobotPoint:
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.x, self.y, self.z)):
            raise ValueError("robot coordinates must be finite")
