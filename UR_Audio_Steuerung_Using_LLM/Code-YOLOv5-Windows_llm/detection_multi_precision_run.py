# MO_Changes
import cv2
import os
import subprocess
import time
import sys
import json
from pathlib import Path

# === Configuration Constants ===
SAVE_DIRECTORY = 'photos'
DETECTION_SCRIPT = 'yolov5/detect_multi_objects.py'  # Use new multi-object detection
WEIGHTS = 'my_model.pt'
CAMERA_INDEX = 0
ROBOT_RESOLUTION = (2560, 1472)
UNIVERSAL_CAPTURE_RESOLUTION = (2560, 1472)
FRANKA_CAPTURE_RESOLUTION = (1280, 720)
IMAGE_RESOLUTION = UNIVERSAL_CAPTURE_RESOLUTION


def get_capture_resolution():
    robot_type = os.environ.get("ROBOT_TYPE", "universal").strip().lower()
    if robot_type == "franka":
        return FRANKA_CAPTURE_RESOLUTION
    return UNIVERSAL_CAPTURE_RESOLUTION

# === Base directory configuration ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Code-YOLOv5-Windows_llm directory

# === Output Directories ===
TXT_FILE_DIR = os.path.join(BASE_DIR, 'txt_file')
PHOTOS_DIR = os.path.join(BASE_DIR, 'photos') 
TEST_PHOTOS_DIR = os.path.join(BASE_DIR, 'test_photo')

# Create directories if they don't exist
os.makedirs(TXT_FILE_DIR, exist_ok=True)
os.makedirs(PHOTOS_DIR, exist_ok=True)
os.makedirs(TEST_PHOTOS_DIR, exist_ok=True)

# === Python executable configuration ===
def get_python_executable():
    """Get the correct Python executable for YOLOv5"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different Python executables in order of preference
    python_candidates = [
        sys.executable,  # Current Python interpreter
        'python',
        'python3',
        os.path.join(current_dir, 'venv', 'Scripts', 'python.exe'),  # Windows venv
        os.path.join(current_dir, 'venv', 'bin', 'python'),  # Linux/Mac venv
    ]
    
    for python_exe in python_candidates:
        try:
            result = subprocess.run([python_exe, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"INFO: Using Python executable: {python_exe}")
                return python_exe
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    print("WARNING: No suitable Python executable found, using 'python'")
    return 'python'

PYTHON_EXE = get_python_executable()

class MultiObjectDetector:
    """
    Multi-Object Detection System for Audio-Controlled Robot
    Detects ALL objects simultaneously and provides structured output for object selection
    """
    
    def __init__(self):
        self.detected_objects = []
        self.object_selection_data = {}
        self.last_detection_result = {}
    
    def capture_image(self, filename="photo_1.jpg"):
        """Capture image from camera"""
        cap = None
        try:
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                raise Exception(f"Cannot open camera {CAMERA_INDEX}")

            capture_width, capture_height = get_capture_resolution()
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
            
            ret, frame = cap.read()
            if not ret:
                raise Exception("Failed to capture image")

            actual_height, actual_width = frame.shape[:2]
            robot_type = os.environ.get("ROBOT_TYPE", "universal").strip().lower()
            print(
                f"CAMERA: Requested {capture_width} x {capture_height}, "
                f"received {actual_width} x {actual_height}"
            )
            if robot_type == "franka" and (
                actual_width != capture_width or actual_height != capture_height
            ):
                raise RuntimeError(
                    "Franka camera must provide 1280 x 720 because the active "
                    "calibration was recorded at 1280 x 720"
                )
            
            # Use absolute path for photos directory
            image_path = os.path.join(PHOTOS_DIR, filename)
            if not cv2.imwrite(image_path, frame):
                raise RuntimeError(f"Could not save captured image {image_path}")
            
            print(f"SUCCESS: Image captured: {image_path}")
            return image_path
        except Exception as e:
            print(f"ERROR: Camera capture failed: {e}")
            return None
        finally:
            if cap is not None:
                cap.release()
    
    def detect_all_objects(self, image_path, confidence_threshold=0.25):
        """
        Run YOLOv5 detection on image and return all detected objects
        Also automatically selects first object for legacy compatibility
        """
        print(f"DETECTION: Starting object detection with image: {image_path}")
        
        # Get absolute paths to avoid path issues
        current_dir = BASE_DIR
        yolov5_dir = os.path.join(current_dir, "yolov5")
        abs_image_path = os.path.abspath(image_path)
        txt_file_abs = TXT_FILE_DIR  # Already absolute
        
        print(f"DEBUG: Current dir: {current_dir}")
        print(f"DEBUG: YOLOv5 dir: {yolov5_dir}")
        print(f"DEBUG: Image path: {abs_image_path}")
        print(f"DEBUG: TXT dir: {txt_file_abs}")
        
        if not os.path.exists(yolov5_dir):
            print(f"ERROR: YOLOv5 directory not found: {yolov5_dir}")
            return []
            
        if not os.path.exists(abs_image_path):
            print(f"ERROR: Image file not found: {abs_image_path}")
            return []
        
        # Clear old detection results first
        self._clear_old_results()
        
        # Find model
        model_path = self._find_model(yolov5_dir)
        if not model_path:
            return []
        
        try:
            # Save original working directory
            original_cwd = os.getcwd()
            print(f"DEBUG: Original working directory: {original_cwd}")
            
            # Change to yolov5 directory for detection
            os.chdir(yolov5_dir)
            print(f"DEBUG: Changed to YOLOv5 directory: {os.getcwd()}")
            
            # Create runs directory if it doesn't exist
            runs_dir = os.path.join(yolov5_dir, "runs", "detect")
            os.makedirs(runs_dir, exist_ok=True)
            
            # Build command with absolute paths
            cmd = [
                PYTHON_EXE, 
                'detect_multi_objects.py',
                '--weights', model_path,
                '--source', abs_image_path,
                '--conf-thres', str(confidence_threshold),
                '--save-txt',
                '--save-crop',
                '--project', 'runs/detect',
                '--name', 'multi_exp'
                # NOTE: Removed --exist-ok to enable automatic run incrementing (multi_exp1, multi_exp2, etc.)
            ]
            
            print(f"DEBUG: Running command: {' '.join(cmd)}")
            print("DETECTION: Running YOLOv5 multi-object detection...")
            
            # Run detection with timeout
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                                  encoding='utf-8', errors='ignore')
            
            print(f"DEBUG: Detection return code: {result.returncode}")
            print(f"DEBUG: Detection stdout: {result.stdout}")
            if result.stderr:
                print(f"DEBUG: Detection stderr: {result.stderr}")
            
            if result.returncode != 0:
                print(f"ERROR: YOLOv5 detection failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                print(f"STDOUT: {result.stdout}")
                return []
            
            print("SUCCESS: YOLOv5 detection completed")
            
            # Load detection results from JSON (with absolute path)
            json_path = os.path.join(txt_file_abs, "detected_objects.json")
            print(f"DEBUG: Looking for results at: {json_path}")
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                self.detected_objects = data.get('objects', [])
                print(f"DEBUG: Loaded {len(self.detected_objects)} objects from JSON")
                
                # Create object selection data
                self.create_object_selection_data()
                
                # **NEW: Automatically select first object for legacy compatibility**
                if self.detected_objects:
                    first_object = self.detected_objects[0]
                    class_name = first_object['class_name']
                    print(f"AUTO-SELECT: Selecting first detected object '{class_name}' for legacy compatibility")
                    
                    # Write legacy files for first object
                    self._write_object_data_for_robot(first_object)
                    
                    print(f"LEGACY: Legacy txt files created for {class_name}")
                    
                    # **NEW: Create individual crop path files for each detected object**
                    # Find latest multi_exp directory for crop paths
                    runs_detect_dir = os.path.join(yolov5_dir, "runs", "detect")
                    if os.path.exists(runs_detect_dir):
                        run_dirs = [d for d in os.listdir(runs_detect_dir) if d.startswith('multi_exp')]
                        if run_dirs:
                            run_dirs.sort()
                            latest_exp_dir = os.path.join(runs_detect_dir, run_dirs[-1])
                            create_and_save_crop_paths(self.detected_objects, latest_exp_dir)
                    
                    # **PRECISION DETECTION: Only detection, no automatic processing**
                    print("PRECISION DETECTION: Detection completed, processing will be done separately")
                    print("INFO: PCA/Direction/Border will be executed only for selected object")
                    
                else:
                    print("WARNING: No objects found in detection results")
                
                print(f"DETECTION: Found {len(self.detected_objects)} objects total")
                for i, obj in enumerate(self.detected_objects):
                    print(f"  [{i}] {obj['class_name']} (confidence: {obj['confidence']:.2f})")
                
                return self.detected_objects
            else:
                print(f"ERROR: Detection results not found at {json_path}")
                
                # Check if detection created any files in runs/detect
                runs_detect_dir = os.path.join(yolov5_dir, "runs", "detect")
                if os.path.exists(runs_detect_dir):
                    print(f"DEBUG: Contents of runs/detect: {os.listdir(runs_detect_dir)}")
                    for subdir in os.listdir(runs_detect_dir):
                        subdir_path = os.path.join(runs_detect_dir, subdir)
                        if os.path.isdir(subdir_path):
                            print(f"DEBUG: Contents of {subdir}: {os.listdir(subdir_path)}")
                
                return []
            
        except subprocess.TimeoutExpired:
            print("ERROR: Detection timeout - process took too long (>120s)")
            return []
        except Exception as e:
            print(f"ERROR: Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            # Always restore original working directory
            os.chdir(original_cwd)
            print(f"DEBUG: Restored working directory: {os.getcwd()}")
    
    def _clear_old_results(self):
        """Clear old detection results to ensure fresh detection"""
        txt_dir = TXT_FILE_DIR  # Already absolute path
        files_to_clear = [
            'detected_objects.json',
            'object_selection.json',
            'available_objects.txt',
            'object_count.txt',
            'label.txt',
            'center_point.txt'
        ]
        
        for filename in files_to_clear:
            file_path = os.path.join(txt_dir, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"DEBUG: Cleared old file: {filename}")
                except Exception as e:
                    print(f"WARNING: Could not clear {filename}: {e}")
    
    def _find_model(self, yolov5_dir):
        """Find suitable YOLOv5 model"""
        model_candidates = ["my_model.pt", "best.pt", "yolov5s.pt"]
        
        for model_name in model_candidates:
            model_path = os.path.join(yolov5_dir, model_name)
            if os.path.exists(model_path):
                print(f"INFO: Using model: {model_path}")
                return model_path
        
        print("ERROR: No YOLOv5 model found!")
        print(f"Looked for: {model_candidates}")
        print(f"In directory: {yolov5_dir}")
        if os.path.exists(yolov5_dir):
            print(f"Directory contents: {[f for f in os.listdir(yolov5_dir) if f.endswith('.pt')]}")
        return None
    
    def _execute_additional_processing(self):
        """
        Execute additional image processing steps that were missing in multi-object system:
        - Border detection (border_multi.py) → border_1.jpg
        - Direction calculation (direction_multi.py) → direction_1.jpg  
        - PCA calculation (pca_multi.py) → pca_1.jpg
        """
        try:
            print("PROCESSING: Starting additional image processing steps...")
            
            # Save original working directory
            original_cwd = os.getcwd()
            
            # Change to Code-YOLOv5-Windows_llm directory for processing scripts
            os.chdir(BASE_DIR)
            print(f"DEBUG: Changed to base directory: {os.getcwd()}")
            
            # Step 1: Border detection
            print("STEP 1/3: Border detection...")
            try:
                result = subprocess.run([PYTHON_EXE, 'border_multi.py'], 
                                      capture_output=True, text=True, timeout=60,
                                      encoding='utf-8', errors='ignore')
                if result.returncode == 0:
                    print(" Border detection completed successfully")
                else:
                    print(f" Border detection failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(" Border detection timeout")
            except Exception as e:
                print(f" Border detection error: {e}")
            
            # Step 2: Direction calculation  
            print("STEP 2/3: Direction calculation...")
            try:
                result = subprocess.run([PYTHON_EXE, 'direction_multi.py'], 
                                      capture_output=True, text=True, timeout=60,
                                      encoding='utf-8', errors='ignore')
                if result.returncode == 0:
                    print(" Direction calculation completed successfully")
                else:
                    print(f" Direction calculation failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(" Direction calculation timeout")
            except Exception as e:
                print(f" Direction calculation error: {e}")
            
            # Step 3: PCA calculation
            print("STEP 3/3: PCA calculation...")
            try:
                result = subprocess.run([PYTHON_EXE, 'pca_multi.py'], 
                                      capture_output=True, text=True, timeout=60,
                                      encoding='utf-8', errors='ignore')
                if result.returncode == 0:
                    print(" PCA calculation completed successfully")
                else:
                    print(f" PCA calculation failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(" PCA calculation timeout")
            except Exception as e:
                print(f" PCA calculation error: {e}")
            
            print("PROCESSING: Additional image processing steps completed")
            
            # Check if processing results exist in YOLOv5 runs
            self._verify_processing_results()
            
        except Exception as e:
            print(f"ERROR: Additional processing failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always restore original working directory
            os.chdir(original_cwd)
            print(f"DEBUG: Restored working directory: {os.getcwd()}")
    
    def _verify_processing_results(self):
        """Verify that additional processing created the expected output files"""
        try:
            # Check YOLOv5 runs directory for latest results
            yolov5_runs_dir = os.path.join(BASE_DIR, "yolov5", "runs", "detect")
            
            if not os.path.exists(yolov5_runs_dir):
                print("WARNING: YOLOv5 runs directory not found")
                return
            
            # Find the latest multi_exp directory
            run_dirs = [d for d in os.listdir(yolov5_runs_dir) if d.startswith('multi_exp')]
            if not run_dirs:
                print("WARNING: No multi_exp run directories found")
                return
            
            # Sort to get the latest run
            run_dirs.sort()
            latest_run = run_dirs[-1]
            latest_run_path = os.path.join(yolov5_runs_dir, latest_run)
            
            print(f"DEBUG: Checking processing results in: {latest_run_path}")
            
            # Check for expected processing output files
            expected_files = [
                "border_1.jpg",      # Border detection result
                "direction_1.jpg",   # Direction calculation result
                "pca_1.jpg"          # PCA calculation result
            ]
            
            found_files = []
            missing_files = []
            
            if os.path.exists(latest_run_path):
                run_contents = os.listdir(latest_run_path)
                for expected_file in expected_files:
                    if expected_file in run_contents:
                        found_files.append(expected_file)
                        file_path = os.path.join(latest_run_path, expected_file)
                        file_size = os.path.getsize(file_path)
                        print(f" Found: {expected_file} ({file_size} bytes)")
                    else:
                        missing_files.append(expected_file)
                        print(f" Missing: {expected_file}")
            
            if missing_files:
                print(f"WARNING: Missing processing files: {missing_files}")
                print("This may indicate that additional processing steps failed")
            else:
                print("SUCCESS: All expected processing files found")
            
            # Show all files in latest run for debugging
            if os.path.exists(latest_run_path):
                all_files = os.listdir(latest_run_path)
                print(f"DEBUG: All files in {latest_run}: {all_files}")
                
        except Exception as e:
            print(f"ERROR: Failed to verify processing results: {e}")

    def get_available_objects(self):
        """Get list of available object types"""
        return list(set([obj['class_name'] for obj in self.detected_objects]))

    def get_objects_by_class(self, class_name):
        """Get all objects of a specific class"""
        return [obj for obj in self.detected_objects if obj['class_name'].lower() == class_name.lower()]

    def select_object(self, class_name, index=0):
        """
        Select a specific object for robot manipulation
        Args:
            class_name: Name of the object class (e.g., "Zylinder", "Quader", "Marker")
            index: Index if multiple objects of same class exist (default: 0 for first)
        """
        objects_of_class = self.get_objects_by_class(class_name)
        
        if not objects_of_class:
            print(f"ERROR: No objects of type '{class_name}' found")
            available = self.get_available_objects()
            if available:
                print(f"   Available objects: {', '.join(available)}")
            return None
        
        if index >= len(objects_of_class):
            print(f"ERROR: Index {index} out of range for {class_name} (found {len(objects_of_class)} objects)")
            return None
        
        selected_obj = objects_of_class[index]
        print(f"SUCCESS: Selected object: {selected_obj['class_name']} #{index} (conf: {selected_obj['confidence']:.2f})")
        
        # Write selected object data to files for robot system compatibility
        self._write_object_data_for_robot(selected_obj)
        
        return selected_obj
    
    def convert_origin_for_robot(self, origin):
        """
        Convert pixel coordinates to robot coordinate space - EXACT SAME AS ORIGINAL detection.py
        Args:
            origin: tuple of (x, y) pixel coordinates
        Returns:
            tuple of (x_robot, y_robot) in robot coordinate space
        """
        try:
            x_robot = int((origin[0] * ROBOT_RESOLUTION[0]) / IMAGE_RESOLUTION[0])
            y_robot = int((origin[1] * ROBOT_RESOLUTION[1]) / IMAGE_RESOLUTION[1])
            print(f"DEBUG: Converted pixel ({origin[0]:.1f}, {origin[1]:.1f}) -> robot ({x_robot}, {y_robot})")
            return (x_robot, y_robot)
        except Exception as e:
            print(f"Error converting origin for robot: {e}")
            return None
    
    def _write_object_data_for_robot(self, selected_obj):
        """
        Write object data in the format expected by the robot control system
        KORRIGIERTE VERSION - now identical to original detection.py workflow
        Creates both new JSON format and legacy txt files for compatibility
        """
        try:
            # Use the absolute TXT_FILE_DIR path
            txt_dir = TXT_FILE_DIR
            
            # Ensure directory exists
            os.makedirs(txt_dir, exist_ok=True)
            
            # 1. Get pixel center coordinates from detected object
            pixel_center = selected_obj["center"]
            print(f"DEBUG: Original pixel center: ({pixel_center[0]:.1f}, {pixel_center[1]:.1f})")
            
            # 2. CRITICAL: Convert pixel coordinates to robot coordinates (like original!)
            robot_center = self.convert_origin_for_robot(pixel_center)
            if not robot_center:
                print("ERROR: Robot coordinate conversion failed")
                return
            
            center_x, center_y = robot_center
            print(f"DEBUG: Final robot center coordinates: ({center_x}, {center_y})")
            
            # 3. Write center point data in CORRECT FORMAT (space-separated, no newline)
            # Legacy single-object files (for pixel2robot.py compatibility)
            center_point_path = os.path.join(txt_dir, "center_point.txt")
            with open(center_point_path, 'w') as f:
                f.write(f"{center_x} {center_y}")  # Space-separated, no newline - EXACT SAME AS ORIGINAL
            print(f"DEBUG: Wrote legacy center_point.txt with robot coordinates")
            
            # 4. Write label data for the selected object
            label_path = os.path.join(txt_dir, "label.txt")
            bbox = selected_obj["bbox"]  # [x1, y1, x2, y2]
            confidence = selected_obj["confidence"]
            class_id = selected_obj["class"]
            
            with open(label_path, 'w') as f:
                # Format: class_id x1 y1 x2 y2 confidence
                f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {confidence}\n")
            print(f"DEBUG: Wrote legacy label.txt")
            
            # 5. Write label path reference
            label_path_file = os.path.join(txt_dir, "label_path.txt")
            with open(label_path_file, 'w') as f:
                f.write(label_path)
            print(f"DEBUG: Wrote label_path.txt")
            
            # 6. Multi-object specific files (also with ROBOT coordinates now!)
            center_point_obj_path = os.path.join(txt_dir, f"center_point_object_{selected_obj['id']}.txt")
            with open(center_point_obj_path, 'w') as f:
                f.write(f"{center_x} {center_y}")  # Consistent format - space-separated, no newline
            print(f"DEBUG: Wrote center_point_object_{selected_obj['id']}.txt with robot coordinates")
            
            label_obj_path = os.path.join(txt_dir, f"label_object_{selected_obj['id']}.txt")
            with open(label_obj_path, 'w') as f:
                f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {confidence}\n")
            print(f"DEBUG: Wrote label_object_{selected_obj['id']}.txt")
            
            # 7. Verify files were created and have content
            files_to_check = [
                center_point_path,
                label_path,
                center_point_obj_path,
                label_obj_path
            ]
            
            for file_path in files_to_check:
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    print(f"DEBUG: File {os.path.basename(file_path)} created, size: {size} bytes")
                else:
                    print(f"WARNING: File {os.path.basename(file_path)} was not created!")
            
            print(f"SUCCESS: Object data written to txt files")
            print(f"   - Pixel Center: ({pixel_center[0]:.1f}, {pixel_center[1]:.1f})")
            print(f"   - Robot Center: ({center_x}, {center_y})")
            print(f"   - Bbox: {bbox}")
            print(f"   - Class: {class_id} ({selected_obj['class_name']})")
            print(f"   - Confidence: {confidence:.3f}")
            print(f"   - Legacy files created for pixel2robot.py compatibility")
            print(f"   - FIXED: Now uses robot coordinates and correct file format!")
            
        except Exception as e:
            print(f"ERROR: Failed to write object data: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_object_selection_prompt(self):
        """Generate user-friendly object selection prompt"""
        if not self.detected_objects:
            return "No objects detected. Please capture a new image."
        
        # Group objects by class
        objects_by_class = {}
        for obj in self.detected_objects:
            class_name = obj['class_name']
            if class_name not in objects_by_class:
                objects_by_class[class_name] = []
            objects_by_class[class_name].append(obj)
        
        prompt = "Available objects for manipulation:\n"
        for class_name, objects in objects_by_class.items():
            if len(objects) == 1:
                prompt += f"  - {class_name} (confidence: {objects[0]['confidence']:.2f})\n"
            else:
                prompt += f"  - {class_name} (found {len(objects)} objects):\n"
                for i, obj in enumerate(objects):
                    prompt += f"    [{i}] confidence: {obj['confidence']:.2f}\n"
        
        return prompt
    
    def create_object_selection_data(self):
        """Create structured data for object selection in audio system"""
        if not self.detected_objects:
            return {}
        
        selection_data = {
            'total_objects': len(self.detected_objects),
            'available_classes': self.get_available_objects(),
            'objects_by_class': {},
            'selection_options': {}
        }
        
        # Group by class
        for obj in self.detected_objects:
            class_name = obj['class_name']
            if class_name not in selection_data['objects_by_class']:
                selection_data['objects_by_class'][class_name] = []
            selection_data['objects_by_class'][class_name].append(obj)
        
        # Create selection options
        option_counter = 1
        for class_name, objects in selection_data['objects_by_class'].items():
            if len(objects) == 1:
                selection_data['selection_options'][class_name] = {
                    'option_number': option_counter,
                    'object_data': objects[0],
                    'selection_key': class_name.lower()
                }
                option_counter += 1
            else:
                for i, obj in enumerate(objects):
                    key = f"{class_name}_{i+1}"
                    selection_data['selection_options'][key] = {
                        'option_number': option_counter,
                        'object_data': obj,
                        'selection_key': f"{class_name.lower()}_{i+1}"
                    }
                    option_counter += 1
        
        self.object_selection_data = selection_data
        
        # Save to file for audio system
        selection_file = os.path.join(TXT_FILE_DIR, 'object_selection.json')
        with open(selection_file, 'w') as f:
            json.dump(selection_data, f, indent=2)
        
        return selection_data

def capture_and_detect_all(mode="auto"):
    """
    Main function to capture image and detect all objects
    Args:
        mode: "auto", "simulation", or "real"
              - "auto": Detect mode automatically (simulation if test image exists, else real)
              - "simulation": Force use of test image
              - "real": Force use of camera
    Returns list of all detected objects
    """
    detector = MultiObjectDetector()
    image_path = None
    
    # Check environment variable for explicit mode override
    env_mode = os.environ.get('DETECTION_MODE', '').lower()
    if env_mode in ['real', 'simulation']:
        mode = env_mode
        print(f"DEBUG: Mode overridden by environment variable: {mode}")
    
    print(f"DEBUG: Final detection mode: {mode}")
    
    if mode == "simulation":
        # Simulation mode: Use test image
        image_path = _get_test_image_path()
        if image_path:
            print(f"SIMULATION: Using test image: {image_path}")
        else:
            print("ERROR: No test image found for simulation mode")
            return []
    elif mode == "real":
        # Real mode: Force camera capture, ignore test images
        print("CAMERA: Forcing real camera capture (ignoring test images)...")
        image_path = detector.capture_image()
        if not image_path:
            print("ERROR: Failed to capture image from camera")
            return []
    else:
        # Auto mode: Check for test image only if no explicit mode set
        if _has_test_image():
            # Simulation mode: Use test image
            image_path = _get_test_image_path()
            if image_path:
                print(f"AUTO->SIMULATION: Using test image: {image_path}")
            else:
                print("ERROR: No test image found for auto simulation mode")
                return []
        else:
            # Real mode: Capture from camera
            print("AUTO->CAMERA: No test image found, capturing from camera...")
            image_path = detector.capture_image()
            if not image_path:
                print("ERROR: Failed to capture image from camera")
                return []
    
    # Detect all objects
    detected_objects = detector.detect_all_objects(image_path)
    
    if detected_objects:
        # Create selection data
        detector.create_object_selection_data()
        print("\n" + detector.get_object_selection_prompt())
    else:
        print("ERROR: No objects detected")
    
    return detected_objects

def _has_test_image():
    """Check if test image exists for simulation mode"""
    test_image_candidates = [
        os.path.join(TEST_PHOTOS_DIR, "test_photo.jpg"),
        os.path.join(TEST_PHOTOS_DIR, "test_photo_1.jpg"),
        os.path.join(TEST_PHOTOS_DIR, "test_photo_2.jpg"),
        os.path.join(BASE_DIR, "test.jpg"),
        os.path.join(BASE_DIR, "test_photo.jpg")
    ]
    
    for path in test_image_candidates:
        if os.path.exists(path):
            return True
    return False

def _get_test_image_path():
    """Get path to test image for simulation mode"""
    test_image_candidates = [
        os.path.join(TEST_PHOTOS_DIR, "test_photo.jpg"),
        os.path.join(TEST_PHOTOS_DIR, "test_photo_1.jpg"),
        os.path.join(TEST_PHOTOS_DIR, "test_photo_2.jpg"),
        os.path.join(BASE_DIR, "test.jpg"),
        os.path.join(BASE_DIR, "test_photo.jpg")
    ]
    
    for path in test_image_candidates:
        if os.path.exists(path):
            print(f"DEBUG: Found test image: {path}")
            return path
    
    print("DEBUG: No test image found, checked paths:")
    for path in test_image_candidates:
        print(f"  - {path}")
    return None

def select_and_prepare_object(class_name, index=0):
    """
    Select a specific object and prepare data for robot manipulation
    Args:
        class_name: Name of the object class
        index: Index if multiple objects of same class exist
    """
    # Load detection results using absolute path
    json_path = os.path.join(TXT_FILE_DIR, 'detected_objects.json')
    if not os.path.exists(json_path):
        print("ERROR: No detection results found. Run capture_and_detect_all() first.")
        return None
    
    detector = MultiObjectDetector()
    with open(json_path, 'r') as f:
        detection_data = json.load(f)
    detector.detected_objects = detection_data.get('objects', [])
    
    return detector.select_object(class_name, index)

def create_and_save_crop_paths(detected_objects, exp_dir):
    """
    Create individual crop path files for each detected object.
    This creates crop_img_path_obj_X.txt files that the multi-object processing scripts can use.
    """
    try:
        os.makedirs("txt_file", exist_ok=True)
        
        for obj in detected_objects:
            obj_id = obj.get('id', 0)
            class_name = obj.get('class_name', 'Unknown')
            
            # Create crop path based on detection output structure
            # YOLOv5 saves crops as: runs/detect/multi_expX/crops/CLASS_NAME/PHOTO_NAME_objY.jpg
            crop_path = os.path.join(exp_dir, "crops", class_name, f"photo_1_obj_{obj_id}.jpg")
            
            # Create individual crop path file for this object
            crop_path_file = f"txt_file/crop_img_path_obj_{obj_id}.txt"
            with open(crop_path_file, 'w') as f:
                f.write(crop_path)
            
            print(f"CROP: Crop-Pfad fuer Objekt {obj_id} ({class_name}): {crop_path}")
            print(f"   Gespeichert in: {crop_path_file}")
        
        # Also create a legacy crop_img_path.txt for single-object compatibility
        if detected_objects:
            # Use the first object for legacy compatibility
            first_obj = detected_objects[0]
            first_class = first_obj.get('class_name', 'Unknown')
            first_id = first_obj.get('id', 0)
            legacy_crop_path = os.path.join(exp_dir, "crops", first_class, f"photo_1_obj_{first_id}.jpg")
            
            with open("txt_file/crop_img_path.txt", 'w') as f:
                f.write(legacy_crop_path)
            
            print(f"LEGACY: Legacy crop_img_path.txt erstellt fuer ersten Objekt: {legacy_crop_path}")
        
        print(f"SUCCESS: Crop-Pfad-Dateien fuer {len(detected_objects)} Objekte erstellt")
        
    except Exception as e:
        print(f"ERROR: Fehler beim Erstellen der Crop-Pfad-Dateien: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("ROBOT: Multi-Object Detection System for Audio-Controlled Robot")
    print("=" * 60)
    
    # Check for environment variable to determine mode
    detection_mode = os.environ.get('DETECTION_MODE', 'auto')
    print(f"DETECTION: Running in {detection_mode} mode")
    
    # Test the system
    detected_objects = capture_and_detect_all(mode=detection_mode)
    
    if detected_objects:
        print(f"\nDETECTION: Successfully detected {len(detected_objects)} objects")
        
        # Example: Select first Zylinder if available, otherwise select first object
        print("\nDETECTION: Selecting object for robot manipulation...")
        
        # Try to select a Zylinder first (common target)
        zylinder_objects = [obj for obj in detected_objects if obj['class_name'].lower() == 'zylinder']
        if zylinder_objects:
            select_and_prepare_object("Zylinder", 0)
            print("DETECTION: Selected Zylinder for robot manipulation")
        else:
            # Select first available object
            first_object = detected_objects[0]
            select_and_prepare_object(first_object['class_name'], 0)
            print(f"DETECTION: Selected {first_object['class_name']} for robot manipulation")
    else:
        print("\nERROR: No objects detected - robot manipulation not possible")
        print("Please check:")
        print("  1. Test image exists (for simulation mode)")
        print("  2. Camera is connected (for real mode)")
        print("  3. YOLOv5 model is available")
        print("  4. Lighting conditions are adequate") 
