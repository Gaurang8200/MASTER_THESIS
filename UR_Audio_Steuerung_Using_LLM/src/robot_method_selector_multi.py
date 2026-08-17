"""
Enhanced Robot Method Selector for Multi-Object Detection System
Handles object selection and robot workflow configuration for multiple detected objects
"""

import os
import json
import sys
from pathlib import Path

# Add Code-YOLOv5-Windows_llm to path for detection integration
ROOT = os.path.abspath(os.path.dirname(__file__))
CODE_YOLO_PATH = os.path.join(ROOT, "..", "Code-YOLOv5-Windows_llm")
if CODE_YOLO_PATH not in sys.path:
    sys.path.insert(0, CODE_YOLO_PATH)

def load_detection_data():
    """Load current detection data from multi-object detection"""
    try:
        json_path = os.path.join(CODE_YOLO_PATH, "txt_file", "detected_objects.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        else:
            print("WARNING: No detection data found. Run object detection first.")
            return None
    except Exception as e:
        print(f"ERROR: Error loading detection data: {e}")
        return None

def select_target_object(info: dict):
    """
    Select the target object based on extracted information
    Args:
        info: Extracted information containing object, object_index, etc.
    Returns:
        Selected object data or None if not found
    """
    detection_data = load_detection_data()
    if not detection_data:
        return None
    
    target_object_type = info.get('object', '').lower()
    target_index = info.get('object_index', 0)
    
    if not target_object_type:
        print("ERROR: No target object specified in command")
        return None
    
    # Find objects of the requested type
    available_objects = detection_data.get('objects', [])
    objects_of_type = [obj for obj in available_objects if obj['class_name'].lower() == target_object_type]
    
    if not objects_of_type:
        print(f"ERROR: No objects of type '{target_object_type}' found")
        available_types = list(set([obj['class_name'] for obj in available_objects]))
        if available_types:
            print(f"   Available types: {', '.join(available_types)}")
        return None
    
    if target_index >= len(objects_of_type):
        print(f"ERROR: Index {target_index} out of range for {target_object_type} (found {len(objects_of_type)} objects)")
        return None
    
    selected_object = objects_of_type[target_index]
    print(f"SUCCESS: Selected object: {selected_object['class_name']} #{target_index} (conf: {selected_object['confidence']:.2f})")
    
    return selected_object

def prepare_object_data_for_robot(selected_object):
    """
    Prepare selected object data for robot system compatibility
    Updates txt files used by the existing robot system
    """
    try:
        txt_dir = os.path.join(CODE_YOLO_PATH, "txt_file")
        os.makedirs(txt_dir, exist_ok=True)
        
        # Write main label file
        label_path = os.path.join(txt_dir, 'label.txt')
        with open(label_path, 'w') as f:
            line = f"{selected_object['class']} {' '.join(map(str, selected_object['bbox']))} {selected_object['confidence']}"
            f.write(line + "\n")
        
        # Write center point
        center_path = os.path.join(txt_dir, 'center_point.txt')
        with open(center_path, 'w') as f:
            f.write(f"{selected_object['center'][0]},{selected_object['center'][1]}\n")
        
        # Write label path
        label_path_file = os.path.join(txt_dir, 'label_path.txt')
        with open(label_path_file, 'w') as f:
            f.write(label_path)
        
        # Copy crop image path if available
        obj_id = selected_object['id']
        crop_path_file = os.path.join(txt_dir, f'crop_img_path_obj_{obj_id}.txt')
        if os.path.exists(crop_path_file):
            with open(crop_path_file, 'r') as f:
                crop_path = f.read().strip()
            
            # Copy to main crop path file
            main_crop_path = os.path.join(txt_dir, 'crop_img_path.txt')
            with open(main_crop_path, 'w') as f:
                f.write(crop_path)
        
        print(f"FILE: Object data prepared for robot system")
        return True
        
    except Exception as e:
        print(f"ERROR: Error preparing object data: {e}")
        return False

def select_robot_methods_precision(info: dict) -> list:
    """
    Precision robot method selection with second detection and final object filtering
    Uses the new precision detection workflow with final object files
    
    Args:
        info: Dictionary containing extracted information
    
    Returns:
        List of method names for precision workflow
    """
    
    print("ROBOT: Multi-Object Robot Method Selection (PRECISION MODE)")
    print("=" * 50)
    
    # Load and validate detection data
    detection_data = load_detection_data()
    if not detection_data:
        print("ERROR: No detection data available. Cannot proceed with robot commands.")
        return []
    
    print(f"INFO: Available objects: {detection_data.get('count', 0)} total")
    for obj_type in detection_data.get('available_objects', []):
        count = len([obj for obj in detection_data.get('objects', []) if obj['class_name'] == obj_type])
        print(f"   - {obj_type}: {count} object(s)")
    
    # Select target object
    selected_object = select_target_object(info)
    if not selected_object:
        return []
    
    # Prepare object data for robot system
    if not prepare_object_data_for_robot(selected_object):
        return []
    
    # Extract command information
    intent = info.get('intent', '').lower()
    target_location = info.get('target_location', '')
    action = info.get('action', '').lower()
    
    print(f"TARGET: Command Details:")
    print(f"   Intent: {intent}")
    print(f"   Target: {target_location}")
    print(f"   Object: {selected_object['class_name']} #{info.get('object_index', 0)}")
    
    # Basic safety checks
    if intent in ['danger', 'error']:
        print("WARNING: Dangerous command detected. Aborting.")
        return []
    
    if not target_location or target_location not in ['Zone_1', 'Zone_2', 'Zone_3']:
        print(f"ERROR: Invalid target location: {target_location}")
        return []
    
    # Build PRECISION WORKFLOW sequence with final object filtering
    methods = []
    
    print(f"INFO: Building PRECISION WORKFLOW for {selected_object['class_name']} → {target_location}")
    
    # Standard beginning
    methods.append("move_to_main_position")
    methods.append("detect_object")
    methods.append("convert_pixel_to_robot")
    
    # KORRIGIERTE PRECISION SEQUENCE (entspricht Vorgänger-Workflow)
    methods.append("move_to_selected_object")                                    # Heranfahren an gewähltes Objekt
    methods.append("precision_detection")                                        # Zweite Detection (NUR Detection)
    methods.append("filter_and_prepare_selected_object_after_precision_detection")  # Objektfilterung
    methods.append("precision_pca_calculation")                                  # PCA nur für gewähltes Objekt
    methods.append("precision_direction_object")                                 # Direction nur für gewähltes Objekt
    
    # Continue with standard picking and placing
    methods.append("pick_the_object")               # Greifen mit finalen Daten
    methods.append("suction_on")                    # Ansaugen
    methods.append("pick_up_object")                # Anheben
    methods.append("intermediate_position")         # Zwischenposition
    
    # Target-specific movement
    methods.append(f"move_to_target({target_location})")  # REAKTIVIERT: Setzt Zone-Koordinaten (ohne Bewegung)
    methods.append("final_position")                      # Ablegen (verwendet bereits gesetzte Zone-Koordinaten)
    methods.append("suction_off")                         # Loslassen
    
    # Cleanup
    methods.append("intermediate_position")         # Zurück zur Zwischenposition
    methods.append("move_to_main_position")         # Zurück zur Hauptposition
    methods.append("delet_txt_file")                # Dateien löschen
    
    print(f"INFO: PRECISION WORKFLOW generated with {len(methods)} steps")
    print("SEQUENCE: Overview Detection → Approach → PRECISION Detection → Filter Object → PCA/Direction → Execute")
    
    return methods

def select_robot_methods_multi(info: dict, use_precision=False) -> list:
    """
    Enhanced robot method selection with multi-object support
    Now supports both LEGACY WORKFLOW and PRECISION WORKFLOW modes
    
    Args:
        info: Dictionary containing extracted information
        use_precision: If True, uses precision detection workflow; if False, uses legacy workflow
    
    Returns:
        List of method names for the selected workflow
    """
    
    if use_precision:
        return select_robot_methods_precision(info)
    
    # EXISTING LEGACY MODE (unchanged)
    print("ROBOT: Multi-Object Robot Method Selection (LEGACY MODE)")
    print("=" * 50)
    
    # Load and validate detection data
    detection_data = load_detection_data()
    if not detection_data:
        print("ERROR: No detection data available. Cannot proceed with robot commands.")
        return []
    
    print(f"INFO: Available objects: {detection_data.get('count', 0)} total")
    for obj_type in detection_data.get('available_objects', []):
        count = len([obj for obj in detection_data.get('objects', []) if obj['class_name'] == obj_type])
        print(f"   - {obj_type}: {count} object(s)")
    
    # Select target object
    selected_object = select_target_object(info)
    if not selected_object:
        return []
    
    # Prepare object data for robot system
    if not prepare_object_data_for_robot(selected_object):
        return []
    
    # Extract command information
    intent = info.get('intent', '').lower()
    target_location = info.get('target_location', '')
    action = info.get('action', '').lower()
    
    print(f"TARGET: Command Details:")
    print(f"   Intent: {intent}")
    print(f"   Target: {target_location}")
    print(f"   Object: {selected_object['class_name']} #{info.get('object_index', 0)}")
    
    # Basic safety checks
    if intent in ['danger', 'error']:
        print("WARNING: Dangerous command detected. Aborting.")
        return []
    
    if not target_location or target_location not in ['Zone_1', 'Zone_2', 'Zone_3']:
        print(f"ERROR: Invalid target location: {target_location}")
        return []
    
    # Return LEGACY WORKFLOW METHOD (replaces all individual methods)
    # This will call execute_legacy_robot_workflow() which replicates Application.py exactly
    legacy_method = f"execute_legacy_workflow_{target_location}"
    
    print(f"INFO: Using LEGACY WORKFLOW from Application.py")
    print(f"INFO: Single method: {legacy_method}")
    print(f"INFO: This replicates the exact Application.py workflow for {target_location}")
    
    return [legacy_method]

def get_object_selection_info():
    """
    Get current object selection information for display
    Returns formatted string with available objects
    """
    detection_data = load_detection_data()
    if not detection_data:
        return "No objects detected. Please run object detection first."
    
    info = "TARGET: Available Objects for Selection:\n"
    info += "=" * 40 + "\n"
    
    objects_by_class = {}
    for obj in detection_data.get('objects', []):
        class_name = obj['class_name']
        if class_name not in objects_by_class:
            objects_by_class[class_name] = []
        objects_by_class[class_name].append(obj)
    
    for class_name, objects in objects_by_class.items():
        if len(objects) == 1:
            obj = objects[0]
            info += f"• {class_name} (confidence: {obj['confidence']:.2f})\n"
            info += f"  Command: 'Bewege {class_name} zu Zone 1'\n\n"
        else:
            info += f"• {class_name} ({len(objects)} objects available):\n"
            for i, obj in enumerate(objects):
                info += f"  [{i}] Confidence: {obj['confidence']:.2f}\n"
            info += f"  Commands: 'Bewege ersten {class_name} zu Zone 1'\n"
            info += f"           'Bewege zweiten {class_name} zu Zone 2'\n\n"
    
    return info

# Backward compatibility function
def select_robot_methods(info: dict) -> list:
    """Backward compatibility wrapper for existing code"""
    return select_robot_methods_multi(info)

def main():
    """Test function for multi-object robot method selection"""
    
    # Display current object selection info
    print(get_object_selection_info())
    
    # Test with sample command
    test_info = {
        'intent': 'bewegen',
        'target_location': 'Zone_1', 
        'action': 'greife Zylinder, bewege zu Zone_1 und lasse los',
        'object': 'Zylinder',
        'object_index': 0,
        'command_type': 'new'
    }
    
    print("Testing with sample command:")
    print(f"Command Info: {json.dumps(test_info, indent=2, ensure_ascii=False)}")
    print()
    
    methods = select_robot_methods_multi(test_info)
    
    if methods:
        print(f"\nSUCCESS: Robot workflow generated with {len(methods)} steps")
    else:
        print(f"\nERROR: No robot workflow generated")

if __name__ == "__main__":
    main() 