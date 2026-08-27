# MO_Changes
# YOLOv5 Multi-Object Detection 🚀 by Ultralytics, modified for simultaneous multi-object detection
"""
Run YOLOv5 detection inference to detect ALL objects simultaneously.
Designed for audio-controlled robot system with free object selection.

Usage:
    $ python detect_multi_objects.py --weights my_model.pt --source photo.jpg
"""

import argparse
import csv
import os
import platform
import sys
import json
import pathlib
from pathlib import Path
from collections import defaultdict

import torch

if os.name != "nt":
    # models/experimental.py unconditionally aliases PosixPath to
    # WindowsPath, assuming a Windows host. Pre empt that here so the
    # alias becomes a no op on macOS/Linux instead of an unusable class.
    pathlib.WindowsPath = pathlib.PosixPath

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

class MultiObjectTracker:
    """
    Tracks multiple objects and provides structured output for audio control system
    """
    def __init__(self):
        self.detected_objects = []
        self.detection_metadata = {}
    
    def add_detection(self, obj_class, confidence, xyxy, class_name):
        """Add a detected object to the tracker"""
        detection = {
            "id": len(self.detected_objects),
            "class": obj_class,
            "class_name": class_name,
            "confidence": float(confidence),
            "bbox": [float(x) for x in xyxy],
            "center": [
                float((xyxy[0] + xyxy[2]) / 2),
                float((xyxy[1] + xyxy[3]) / 2)
            ]
        }
        self.detected_objects.append(detection)
    
    def convert_origin_for_robot(self, origin):
        """
        Convert pixel coordinates to robot coordinate space - SAME AS UR_detection.py
        Args:
            origin: tuple/list of (x, y) pixel coordinates
        Returns:
            tuple of (x_robot, y_robot) in robot coordinate space
        """
        try:
            # Robot and image resolution constants (should match detection_multi.py)
            ROBOT_RESOLUTION = (2560, 1472)
            IMAGE_RESOLUTION = (2560, 1472)
            
            x_robot = int((origin[0] * ROBOT_RESOLUTION[0]) / IMAGE_RESOLUTION[0])
            y_robot = int((origin[1] * ROBOT_RESOLUTION[1]) / IMAGE_RESOLUTION[1])
            return (x_robot, y_robot)
        except Exception as e:
            print(f"Error converting origin for robot: {e}")
            return origin  # fallback to original coordinates
    
    def export_to_files(self, txt_dir="../txt_file"):
        """Export detection results to various file formats for compatibility"""
        os.makedirs(txt_dir, exist_ok=True)
        
        # Export detailed JSON for multi-object selection
        json_path = os.path.join(txt_dir, "detected_objects.json")
        with open(json_path, 'w') as f:
            json.dump({
                "objects": self.detected_objects,
                "count": len(self.detected_objects),
                "available_objects": list(set([obj["class_name"] for obj in self.detected_objects])),
                "metadata": self.detection_metadata
            }, f, indent=2)
        
        # Export individual object files for compatibility - NOW WITH ROBOT COORDINATES
        for i, obj in enumerate(self.detected_objects):
            # Convert pixel center to robot coordinates
            pixel_center = obj["center"]
            robot_center = self.convert_origin_for_robot(pixel_center)
            
            # Create individual label file for each object
            label_path = os.path.join(txt_dir, f"label_object_{i}.txt")
            with open(label_path, 'w') as f:
                line = f"{obj['class']} {' '.join(map(str, obj['bbox']))} {obj['confidence']}"
                f.write(line + "\n")
            
            # Create center point file for each object - CORRECTED FORMAT
            center_path = os.path.join(txt_dir, f"center_point_object_{i}.txt")
            with open(center_path, 'w') as f:
                f.write(f"{robot_center[0]} {robot_center[1]}")  # Space-separated, no newline
        
        # Export summary files
        with open(os.path.join(txt_dir, "object_count.txt"), 'w') as f:
            f.write(str(len(self.detected_objects)))
        
        with open(os.path.join(txt_dir, "available_objects.txt"), 'w') as f:
            for obj_name in set([obj["class_name"] for obj in self.detected_objects]):
                f.write(f"{obj_name}\n")
        
        # Compatibility: Export first object as default (for legacy system) - WITH ROBOT COORDINATES
        if self.detected_objects:
            first_obj = self.detected_objects[0]
            
            # Convert first object center to robot coordinates
            first_pixel_center = first_obj["center"]
            first_robot_center = self.convert_origin_for_robot(first_pixel_center)
            
            with open(os.path.join(txt_dir, "label.txt"), 'w') as f:
                line = f"{first_obj['class']} {' '.join(map(str, first_obj['bbox']))} {first_obj['confidence']}"
                f.write(line + "\n")
            
            # FIXED: Legacy center_point.txt with robot coordinates and correct format
            with open(os.path.join(txt_dir, "center_point.txt"), 'w') as f:
                f.write(f"{first_robot_center[0]} {first_robot_center[1]}")  # Space-separated, no newline

# Global object tracker
object_tracker = MultiObjectTracker()

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=True,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    global object_tracker
    object_tracker = MultiObjectTracker()  # Reset tracker
    
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # CSV path
        csv_path = save_dir / "predictions.csv"

        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions - DETECT ALL OBJECTS (no break statement!)
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Sort by confidence for consistent ordering
                detections = det.tolist()
                detections.sort(key=lambda x: x[4], reverse=True)  # Sort by confidence
                sorted_det = torch.tensor(detections)

                # Process ALL detected objects (removed break statement!)
                crop_counter = 0
                for obj_idx, (*xyxy, conf, cls) in enumerate(reversed(sorted_det)):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"
                    
                    # Add to object tracker
                    object_tracker.add_detection(c, confidence, xyxy, names[c])

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xyxy, conf) if save_conf else (cls, *xyxy)  # label format
                        
                        # Write individual object label file
                        obj_txt_path = f"{txt_path}_obj_{obj_idx}.txt"
                        with open(obj_txt_path, "w") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")
                        
                        # Write to main label file (all objects)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    
                    if save_crop:
                        crop_path = save_dir / "crops" / names[c] / f"{p.stem}_obj_{obj_idx}.jpg"
                        save_one_box(xyxy, imc, file=crop_path, BGR=True)
                        crop_counter += 1
                        
                        # Write crop path for each object
                        with open(f"../txt_file/crop_img_path_obj_{obj_idx}.txt", "w") as file:
                            file.write(str(crop_path))
                    
                    # NO BREAK STATEMENT HERE! Process all objects!

                # Update metadata
                object_tracker.detection_metadata = {
                    "image_path": str(p),
                    "total_detections": len(sorted_det),
                    "detection_timestamp": str(torch.tensor([]).tolist()),  # Simple timestamp
                    "model_confidence_threshold": conf_thres
                }
                
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                    with open('../txt_file/detect_img_path.txt', 'w') as f:
                        f.write(save_path)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}({len(det) if len(det) else 'no'} detections), {dt[1].dt * 1E3:.1f}ms")

    # Export all detected objects to files
    # Use absolute path to ensure correct export location
    export_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "txt_file")
    object_tracker.export_to_files(export_dir)
    
    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    
    # Summary of multi-object detection
    LOGGER.info(f"Multi-Object Detection Summary: {len(object_tracker.detected_objects)} objects detected")
    for obj in object_tracker.detected_objects:
        LOGGER.info(f"  - {obj['class_name']} (conf: {obj['confidence']:.2f})")
    
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    return object_tracker.detected_objects


def parse_opt():
    """Parses command-line arguments for YOLOv5 multi-object detection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="multi_exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 multi-object detection with given options."""
    detected_objects = run(**vars(opt))
    return detected_objects


if __name__ == "__main__":
    opt = parse_opt()
    main(opt) 
