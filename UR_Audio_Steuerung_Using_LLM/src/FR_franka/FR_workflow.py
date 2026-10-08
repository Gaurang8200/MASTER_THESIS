# MO_Changes
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from src import robot_control as perception_steps
from src.robot_output import emit_method_execution

from .FR_config import FrankaConfig, load_franka_config
from .FR_depth_transformer import (
    CalibratedDepth,
    FrankaDepthTransformer,
)
from .FR_geometry import rotation_vector_to_quaternion
from .FR_models import CartesianPose, PixelPoint, RobotPoint
from .FR_original_transformer import OriginalFrankaPixelTransformer
from .FR_robot import FrankaRobotArm, RobotArm, SimulatedFrankaRobotArm
from .FR_runtime_data import (
    read_class_file,
    read_detection_image_size,
    read_numeric_values,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOLO_ROOT = PROJECT_ROOT / "Code-YOLOv5-Windows_llm"
TXT_DIR = YOLO_ROOT / "txt_file"


@dataclass
class WorkflowContext:
    selected_point: RobotPoint | None = None
    selected_class: int | None = None
    selected_orientation: tuple[float, float, float, float] | None = None
    target_zone: CartesianPose | None = None
    target_is_dynamic: bool = False
    depth_measurement: CalibratedDepth | None = None


class FrankaAudioWorkflow:
    def __init__(
        self,
        arm: RobotArm,
        config: FrankaConfig,
        transformer: OriginalFrankaPixelTransformer,
        depth_transformer: FrankaDepthTransformer,
        simulation: bool,
        output: Callable[[str], None],
    ) -> None:
        self._arm = arm
        self._config = config
        self._transformer = transformer
        self._depth_transformer = depth_transformer
        self._simulation = simulation
        self._output = output
        self._context = WorkflowContext()
        self._started = False

    def execute(self, method_list: Sequence[str]) -> None:
        if not method_list:
            raise ValueError("Franka workflow requires at least one method")
        self._preflight(method_list)
        self._load_selection_context()
        self.start()
        for method_name in method_list:
            emit_method_execution(method_name, self._output)
            try:
                self._execute_method(method_name)
            except Exception as error:
                raise RuntimeError(
                    f"Franka method {method_name} failed: {error}"
                ) from error

    def start(self) -> None:
        if self._started:
            return
        self._arm.start()
        if not self._arm.health():
            self._arm.close()
            raise RuntimeError("Franka robot health check failed")
        self._started = True

    def close(self) -> None:
        if not self._started:
            return
        self._arm.close()
        self._started = False

    def _preflight(self, method_list: Sequence[str]) -> None:
        supported = {
            "move_to_main_position",
            "detect_object",
            "convert_pixel_to_robot",
            "move_to_selected_object",
            "precision_detection",
            "filter_and_prepare_selected_object_after_precision_detection",
            "precision_pca_calculation",
            "precision_direction_object",
            "pick_the_object",
            "suction_on",
            "pick_up_object",
            "intermediate_position",
            "final_position",
            "suction_off",
            "release_object",
            "delet_txt_file",
        }
        for method_name in method_list:
            if method_name.startswith("move_to_target"):
                match = re.fullmatch(r"move_to_target(?:\(([^)]+)\)|_(.+))", method_name)
                if match is None:
                    raise ValueError(f"Invalid target method {method_name}")
                zone_name = match.group(1) or match.group(2)
                if not self._simulation:
                    self._config.zone(zone_name)
                continue
            if method_name.startswith("move_to_point("):
                if re.fullmatch(r"move_to_point\([^,]+,[^)]+\)", method_name) is None:
                    raise ValueError(f"Invalid point target method {method_name}")
                continue
            if method_name not in supported:
                raise ValueError(f"Unsupported Franka workflow method {method_name}")
        required_inputs = (
            TXT_DIR / "detected_objects.json",
            TXT_DIR / "selection_data.json",
        )
        missing = [str(path) for path in required_inputs if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing Franka workflow input files: {missing}")

    def _execute_method(self, method_name: str) -> None:
        handlers: dict[str, Callable[[], None]] = {
            "move_to_main_position": self._move_home,
            "detect_object": self._detect_object,
            "convert_pixel_to_robot": self._convert_selected_pixel,
            "move_to_selected_object": self._move_above_selected_object,
            "precision_detection": self._precision_detection,
            "filter_and_prepare_selected_object_after_precision_detection": (
                self._prepare_precision_object
            ),
            "precision_pca_calculation": self._precision_pca,
            "precision_direction_object": self._precision_direction,
            "pick_the_object": self._move_to_pick,
            "suction_on": self._grip,
            "pick_up_object": self._lift,
            "intermediate_position": self._move_intermediate,
            "final_position": self._move_to_zone,
            "suction_off": self._release,
            "release_object": self._release,
            "delet_txt_file": self._delete_runtime_files,
        }
        if method_name.startswith("move_to_target"):
            self._select_zone(method_name)
            return
        if method_name.startswith("move_to_point("):
            self._select_point(method_name)
            return
        handler = handlers.get(method_name)
        if handler is None:
            raise ValueError(f"Unsupported Franka workflow method {method_name}")
        handler()

    def _move_home(self) -> None:
        self._output(f"FRANKA MOVE HOME JOINTS: {self._config.home_joints}")
        self._arm.move_joints(self._config.home_joints)

    def _detect_object(self) -> None:
        detection_path = TXT_DIR / "detected_objects.json"
        if not detection_path.is_file():
            raise FileNotFoundError("Verified object detection data is missing")
        self._output("FRANKA DETECTION: Using the verified multi object result")

    def _convert_selected_pixel(self) -> None:
        selection_path = TXT_DIR / "selection_data.json"
        data = json.loads(selection_path.read_text(encoding="utf-8"))
        rgb_pixel = PixelPoint(
            float(data["original_center_x"]),
            float(data["original_center_y"]),
        )
        self._output(
            "FRANKA RGB PIXEL: "
            f"u={rgb_pixel.x:.2f}, v={rgb_pixel.y:.2f}"
        )
        left_center = data.get("left_mono_center")
        left_size = data.get("left_mono_frame_size")
        if (
            isinstance(left_center, list)
            and len(left_center) == 2
            and isinstance(left_size, list)
            and len(left_size) == 2
        ):
            calibration_pixel = PixelPoint(
                float(left_center[0]),
                float(left_center[1]),
            )
            calibration_size = (int(left_size[0]), int(left_size[1]))
        elif self._simulation:
            calibration_pixel = rgb_pixel
            calibration_size = _read_detection_image_size()
        else:
            raise ValueError("Selected object has no mapped left mono calibration pixel")
        self._output(
            "FRANKA LEFT MONO PIXEL: "
            f"u={calibration_pixel.x:.2f}, v={calibration_pixel.y:.2f}"
        )
        planar_point = self._transform_pixel(
            calibration_pixel,
            calibration_size,
        )
        self._context.selected_class = self._read_selected_class(data)
        object_depth_value = data.get("object_depth_mm")
        sample_count_value = data.get("object_depth_sample_count")
        if object_depth_value is not None and sample_count_value is not None:
            self._context.depth_measurement = self._depth_transformer.transform(
                calibration_pixel,
                calibration_size,
                float(object_depth_value),
                self._arm.base_to_end_effector(),
                int(sample_count_value),
            )
        elif not self._simulation:
            raise ValueError("Selected object has no stable OAK D depth measurement")
        robot_z = (
            self._context.depth_measurement.robot_base_z_mm
            if self._context.depth_measurement is not None
            else planar_point.z
        )
        self._context.selected_point = RobotPoint(
            planar_point.x,
            planar_point.y,
            robot_z,
        )
        self._validate_workspace(
            (
                self._context.selected_point.x,
                self._context.selected_point.y,
                self._context.selected_point.z,
            )
        )
        self._output(
            "FRANKA COORDINATES: "
            f"x={self._context.selected_point.x:.2f} mm, "
            f"y={self._context.selected_point.y:.2f} mm, "
            f"z={self._context.selected_point.z:.2f} mm"
        )
        if self._context.depth_measurement is not None:
            measurement = self._context.depth_measurement
            camera_x, camera_y, camera_z = measurement.camera_point_mm
            self._output(
                "OAK D DEPTH: "
                f"object={measurement.object_depth_mm:.1f} mm, "
                f"samples={measurement.sample_count}, "
                f"camera_x={camera_x:.1f} mm, "
                f"camera_y={camera_y:.1f} mm, "
                f"camera_z={camera_z:.1f} mm, "
                f"robot_base_z={measurement.robot_base_z_mm:.1f} mm"
            )

    def _move_above_selected_object(self) -> None:
        point = self._require_selected_point()
        offset_x, offset_y, offset_z = self._config.camera_offset_mm
        camera_x = point.x + offset_x
        camera_y = point.y + offset_y
        camera_z = self._config.approach_height_mm + offset_z
        self._output(
            "FRANKA CAMERA OFFSET: "
            f"x={offset_x:.2f} mm, y={offset_y:.2f} mm, z={offset_z:.2f} mm"
        )
        self._output(
            "FRANKA CAMERA APPROACH: "
            f"x={camera_x:.2f} mm, y={camera_y:.2f} mm, z={camera_z:.2f} mm"
        )
        self._move_cartesian(
            camera_x,
            camera_y,
            camera_z,
            self._config.default_orientation,
        )

    def _precision_detection(self) -> None:
        if self._simulation:
            self._output("FRANKA SIMULATION: Precision detection accepted")
            return
        previous_robot_type = os.environ.get("ROBOT_TYPE")
        os.environ["ROBOT_TYPE"] = "franka"
        try:
            perception_steps.precision_detection(self._config.robot_ip)
        finally:
            if previous_robot_type is None:
                os.environ.pop("ROBOT_TYPE", None)
            else:
                os.environ["ROBOT_TYPE"] = previous_robot_type

    def _prepare_precision_object(self) -> None:
        if self._simulation:
            self._output("FRANKA SIMULATION: Precision object data accepted")
        else:
            success = perception_steps.filter_and_prepare_selected_object_after_precision_detection(
                self._config.robot_ip
            )
            if success is False:
                raise RuntimeError("Precision object filtering failed")
            self._context.selected_class = read_class_file(
                TXT_DIR / "final_object_label.txt"
            )

    def _precision_pca(self) -> None:
        if self._simulation:
            self._output("FRANKA SIMULATION: Precision PCA accepted")
            return
        perception_steps.precision_pca_calculation(self._config.robot_ip)

    def _precision_direction(self) -> None:
        if self._simulation:
            self._context.selected_orientation = self._config.default_orientation
            self._output("FRANKA SIMULATION: Default downward orientation accepted")
            return
        perception_steps.precision_direction_object(self._config.robot_ip)
        orientation_path = TXT_DIR / "robot_RPY.txt"
        values = read_numeric_values(orientation_path)
        if len(values) != 3:
            raise ValueError("Precision direction must produce three rotation vector values")
        self._context.selected_orientation = rotation_vector_to_quaternion(values)

    def _move_to_pick(self) -> None:
        point = self._require_selected_point()
        orientation = self._context.selected_orientation or self._config.default_orientation
        measurement = self._context.depth_measurement
        if measurement is None:
            raise RuntimeError("OAK D depth measurement is required for pickup")
        grip_offset = self._config.grip_offset_below_surface_mm
        pick_height = point.z - grip_offset
        self._validate_workspace((point.x, point.y, pick_height))
        alignment_height = max(self._config.lift_height_mm, pick_height)
        self._output("FRANKA GRIPPER: Opening before pickup approach")
        self._arm.release()
        self._output(
            "FRANKA PICK ALIGNMENT: "
            f"x={point.x:.2f} mm, y={point.y:.2f} mm, "
            f"z={alignment_height:.2f} mm"
        )
        self._move_cartesian(
            point.x,
            point.y,
            alignment_height,
            orientation,
        )
        self._output(
            "OAK D GRIP TARGET: "
            f"object_depth={measurement.object_depth_mm:.1f} mm, "
            f"robot_surface_z={point.z:.1f} mm, "
            f"grip_offset={grip_offset:.1f} mm"
        )
        self._output(
            "FRANKA PICK COORDINATES: "
            f"x={point.x:.2f} mm, y={point.y:.2f} mm, "
            f"z={pick_height:.2f} mm, "
            f"grip_offset={grip_offset:.2f} mm"
        )
        self._move_cartesian(
            point.x,
            point.y,
            pick_height,
            orientation,
        )

    def _grip(self) -> None:
        self._output("FRANKA GRIPPER: Closing around selected object")
        self._arm.grip()
        self._output("FRANKA GRIPPER: Grasp confirmed")

    def _lift(self) -> None:
        point = self._require_selected_point()
        orientation = self._context.selected_orientation or self._config.default_orientation
        self._output(
            "FRANKA PICK LIFT: "
            f"x={point.x:.2f} mm, y={point.y:.2f} mm, "
            f"z={self._config.lift_height_mm:.2f} mm"
        )
        self._move_cartesian(point.x, point.y, self._config.lift_height_mm, orientation)
        self._output("FRANKA PICK LIFT: Completed")

    def _move_intermediate(self) -> None:
        self._output(
            f"FRANKA MOVE INTERMEDIATE JOINTS: {self._config.intermediate_joints}"
        )
        self._arm.move_joints(self._config.intermediate_joints)

    def _select_zone(self, method_name: str) -> None:
        match = re.fullmatch(r"move_to_target(?:\(([^)]+)\)|_(.+))", method_name)
        if match is None:
            raise ValueError(f"Invalid target method {method_name}")
        zone_name = match.group(1) or match.group(2)
        if self._simulation and zone_name not in self._config.zones:
            self._context.target_zone = CartesianPose.create(
                (400.0, 0.0, 100.0), self._config.default_orientation
            )
            self._output(f"FRANKA SIMULATION ZONE: {zone_name}")
            return
        zone = self._config.zone(zone_name)
        object_class = self._require_selected_class()
        self._context.target_zone = CartesianPose.create(
            (
                zone.translation[0],
                zone.translation[1],
                self._config.place_height_mm(object_class),
            ),
            zone.quaternion,
        )
        self._context.target_is_dynamic = False

    def _select_point(self, method_name: str) -> None:
        values = method_name[len("move_to_point("):-1].split(",")
        if len(values) != 2:
            raise ValueError(f"Invalid point target method {method_name}")
        object_class = self._require_selected_class()
        self._context.target_zone = CartesianPose.create(
            (
                float(values[0]),
                float(values[1]),
                self._config.place_height_mm(object_class),
            ),
            self._config.default_orientation,
        )
        self._context.target_is_dynamic = True
        x, y, z = self._context.target_zone.translation
        self._output(
            f"FRANKA POINT TARGET: x={x:.2f} mm, y={y:.2f} mm, z={z:.2f} mm"
        )

    def _move_to_zone(self) -> None:
        if self._context.target_zone is None:
            raise RuntimeError("Franka target zone has not been selected")
        if not self._context.target_is_dynamic:
            self._validate_workspace(self._context.target_zone.translation)
        x, y, z = self._context.target_zone.translation
        self._output(
            f"FRANKA MOVE TARGET: x={x:.2f} mm, y={y:.2f} mm, z={z:.2f} mm"
        )
        self._arm.move_pose(self._context.target_zone)

    def _release(self) -> None:
        self._arm.release()

    def _delete_runtime_files(self) -> None:
        if self._simulation:
            self._output("FRANKA SIMULATION: Runtime file cleanup accepted")
            return
        perception_steps.delet_txt_file()

    def _transform_pixel(
        self,
        pixel: PixelPoint,
        image_size: tuple[int, int],
    ) -> RobotPoint:
        point = self._transformer.transform(
            pixel,
            image_size,
        )
        self._validate_workspace((point.x, point.y, point.z))
        return point

    def _move_cartesian(
        self,
        x: float,
        y: float,
        z: float,
        quaternion: Sequence[float],
    ) -> None:
        self._validate_workspace((x, y, z))
        self._output(
            f"FRANKA MOVE CARTESIAN: x={x:.2f} mm, y={y:.2f} mm, z={z:.2f} mm"
        )
        self._arm.move_pose(CartesianPose.create((x, y, z), quaternion))

    def _validate_workspace(self, translation: Sequence[float]) -> None:
        x, y, z = (float(value) for value in translation)
        checks = (
            (self._config.workspace_x_mm, x, "x"),
            (self._config.workspace_y_mm, y, "y"),
            (self._config.workspace_z_mm, z, "z"),
        )
        for limits, value, axis in checks:
            if not limits[0] <= value <= limits[1]:
                raise ValueError(
                    f"Franka target {axis}={value:.2f} mm is outside workspace "
                    f"{limits} mm"
                )

    def _require_selected_point(self) -> RobotPoint:
        if self._context.selected_point is None:
            raise RuntimeError("Selected object has no Franka coordinates")
        return self._context.selected_point

    def _require_selected_class(self) -> int:
        if self._context.selected_class is None:
            raise RuntimeError("Selected object has no class")
        return self._context.selected_class

    def _load_selection_context(self) -> None:
        selection_path = TXT_DIR / "selection_data.json"
        if not selection_path.is_file():
            return
        data = json.loads(selection_path.read_text(encoding="utf-8"))
        self._context.selected_class = self._read_selected_class(data)

    @staticmethod
    def _read_selected_class(selection_data: dict[str, object]) -> int:
        label_path = TXT_DIR / "label.txt"
        if label_path.is_file():
            return read_class_file(label_path)
        class_name = str(selection_data["selected_object_class"]).lower()
        class_ids = {"cylinder": 0, "box": 1, "marker": 2}
        if class_name not in class_ids:
            raise ValueError(f"Unknown selected object class {class_name}")
        return class_ids[class_name]

class FrankaWorkflowSession:
    def __init__(
        self,
        workflow: FrankaAudioWorkflow,
        simulation: bool,
        output_callback: Callable[[str], None] | None,
    ) -> None:
        self._workflow = workflow
        self._simulation = simulation
        self._output_callback = output_callback

    def __enter__(self) -> "FrankaWorkflowSession":
        perception_steps.set_simulation_mode(
            self._simulation,
            self._output_callback,
        )
        try:
            self._workflow.start()
        except Exception:
            perception_steps.set_simulation_mode(False, None)
            raise
        return self

    def execute(self, method_list: Sequence[str]) -> None:
        self._workflow.execute(method_list)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self._workflow.close()
        finally:
            perception_steps.set_simulation_mode(False, None)


def create_franka_workflow_session(
    robot_ip: str | None = None,
    simulation: bool = False,
    output_callback: Callable[[str], None] | None = None,
) -> FrankaWorkflowSession:
    config = load_franka_config()
    if robot_ip and robot_ip.strip():
        config = FrankaConfig(**{**config.__dict__, "robot_ip": robot_ip.strip()})
    output = output_callback or print
    transformer = OriginalFrankaPixelTransformer(
        (config.calibration_width, config.calibration_height),
        config.mirror_x,
        config.calibration_directory,
    )
    depth_transformer = FrankaDepthTransformer(
        (config.calibration_width, config.calibration_height),
        config.mirror_x,
        config.calibration_directory,
    )
    arm: RobotArm
    if simulation:
        arm = SimulatedFrankaRobotArm(transformer.calibration_pose())
    else:
        arm = FrankaRobotArm(
            config.robot_ip,
            config.dynamics_factor,
            config.gripper_speed_mm_s,
            config.gripper_force,
        )
    workflow = FrankaAudioWorkflow(
        arm,
        config,
        transformer,
        depth_transformer,
        simulation,
        output,
    )
    return FrankaWorkflowSession(workflow, simulation, output_callback)


def execute_franka_workflow(
    method_list: Sequence[str],
    robot_ip: str | None = None,
    simulation: bool = False,
    output_callback: Callable[[str], None] | None = None,
) -> None:
    with create_franka_workflow_session(
        robot_ip=robot_ip,
        simulation=simulation,
        output_callback=output_callback,
    ) as session:
        session.execute(method_list)


def prepare_franka_for_detection(robot_ip: str | None = None) -> None:
    config = load_franka_config()
    if robot_ip and robot_ip.strip():
        config = FrankaConfig(**{**config.__dict__, "robot_ip": robot_ip.strip()})
    arm = FrankaRobotArm(
        config.robot_ip,
        config.dynamics_factor,
        config.gripper_speed_mm_s,
        config.gripper_force,
    )
    try:
        arm.start()
        if not arm.health():
            raise RuntimeError("Franka robot health check failed")
        arm.move_joints(config.home_joints)
    finally:
        arm.close()


def transform_franka_left_mono_pixel_to_robot(
    pixel_x: float,
    pixel_y: float,
    frame_width: int,
    frame_height: int,
) -> RobotPoint:
    config = load_franka_config()
    expected_size = (config.calibration_width, config.calibration_height)
    actual_size = (int(frame_width), int(frame_height))
    if actual_size[0] <= 0 or actual_size[1] <= 0:
        raise ValueError("Franka destination image dimensions must be positive")
    if actual_size[0] * expected_size[1] != actual_size[1] * expected_size[0]:
        raise ValueError(
            f"Franka destination image is {actual_size[0]} x {actual_size[1]} but "
            f"the active calibration aspect ratio is "
            f"{expected_size[0]} x {expected_size[1]}"
        )
    transformer = OriginalFrankaPixelTransformer(
        expected_size,
        config.mirror_x,
        config.calibration_directory,
    )
    return transformer.transform(
        PixelPoint(float(pixel_x), float(pixel_y)),
        actual_size,
    )


def _read_detection_image_size() -> tuple[int, int]:
    return read_detection_image_size(TXT_DIR, YOLO_ROOT)
