# MO_Changes
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class SceneSpec:
    object_x: float = 0.4
    object_y: float = 0.2
    object_radius: float = 0.025
    object_half_height: float = 0.025
    object_mass: float = 0.1
    object_class: int = 0
    target_zone: str = "Zone_1"

    def __post_init__(self) -> None:
        values = (
            self.object_x,
            self.object_y,
            self.object_radius,
            self.object_half_height,
            self.object_mass,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("Scene values must be finite")
        if self.object_radius <= 0.0 or self.object_half_height <= 0.0:
            raise ValueError("Object dimensions must be positive")
        if self.object_mass <= 0.0:
            raise ValueError("Object mass must be positive")
        if self.object_class < 0:
            raise ValueError("Object class must not be negative")
        if not self.target_zone.strip():
            raise ValueError("Target zone must not be empty")


def add_workcell(
    worldbody: ET.Element,
    scene: SceneSpec,
    zone_x: float,
    zone_y: float,
) -> None:
    if worldbody.tag != "worldbody":
        raise ValueError("MuJoCo scene requires a worldbody element")

    table = ET.SubElement(worldbody, "body", name="prototype_table")
    ET.SubElement(
        table,
        "geom",
        name="prototype_table_surface",
        type="box",
        pos="0.35 0.1 -0.025",
        size="0.55 0.45 0.025",
        rgba="0.36 0.25 0.16 1",
        friction="1.0 0.01 0.001",
    )

    object_body = ET.SubElement(
        worldbody,
        "body",
        name="prototype_object",
        pos=(
            f"{scene.object_x} {scene.object_y} "
            f"{scene.object_half_height}"
        ),
    )
    ET.SubElement(object_body, "freejoint", name="prototype_object_joint")
    ET.SubElement(
        object_body,
        "geom",
        name="prototype_object_geom",
        type="cylinder",
        size=f"{scene.object_radius} {scene.object_half_height}",
        mass=str(scene.object_mass),
        rgba="0.85 0.16 0.12 1",
        friction="1.2 0.02 0.002",
        condim="4",
    )

    zone = ET.SubElement(
        worldbody,
        "body",
        name="prototype_target_zone",
        pos=f"{float(zone_x)} {float(zone_y)} 0.003",
    )
    ET.SubElement(
        zone,
        "geom",
        name="prototype_target_zone_geom",
        type="cylinder",
        size="0.065 0.003",
        rgba="0.12 0.55 0.92 0.55",
        contype="0",
        conaffinity="0",
    )
