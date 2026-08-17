import os
import sys
import shutil
from pathlib import Path

# Pfade: Vorgänger + src
ROOT = os.path.abspath(os.path.dirname(__file__))
PRE  = os.path.join(ROOT, "Code-YOLOv5-Windows_llm")
if PRE not in sys.path:
   sys.path.insert(0, PRE)
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
   sys.path.insert(0, SRC)

import json
import subprocess
import time
import tkinter as tk
from tkinter import ttk, messagebox
import speech_recognition as sr
from dotenv import load_dotenv
import threading

# Multi-Object System Imports
from src.speech.speech_to_text_local import SpeechToTextLocal
from src.speech.information_extraction_openai_api_multi import InformationExtractionOpenAIMulti
from src.robot_method_selector_multi import select_robot_methods_multi, select_target_object
from src.zone_coordinates import get_zone_coordinates

# Python executable for YOLOv5 scripts
def get_yolo_python():
   """Get correct Python executable for YOLOv5 detection"""
   candidates = [
       sys.executable,
       'python',
       'python3',
       os.path.join(ROOT, "venv_wrapper", "Scripts", "python.exe"),
       os.path.join(PRE, "venv", "Scripts", "python.exe"),
       os.path.join(PRE, "venv", "bin", "python"),
   ]
   
   for candidate in candidates:
       try:
           result = subprocess.run([candidate, '--version'],
                                 capture_output=True, text=True, timeout=5)
           if result.returncode == 0:
               print(f"INFO: Using Python executable: {candidate}")
               return candidate
       except (subprocess.TimeoutExpired, FileNotFoundError):
           continue
   print("WARNING: Falling back to system Python")
   return 'python'

def get_pre_venv_python():
   """Path to the Code-YOLOv5-Windows_llm venv interpreter for the current OS.

   Detection needs the pinned cv2/torch versions in that subproject's own
   venv, so prefer its Windows or Unix layout, whichever exists on disk.
   The venv there is currently Windows only, so on macOS/Linux it falls
   back to get_yolo_python(), which probes for a working interpreter.
   """
   windows_python = os.path.join(PRE, "venv", "Scripts", "python.exe")
   unix_python = os.path.join(PRE, "venv", "bin", "python")
   if os.name == "nt" and os.path.exists(windows_python):
       return windows_python
   if os.name != "nt" and os.path.exists(unix_python):
       return unix_python
   return get_yolo_python()

YOLO_PYTHON = get_yolo_python()

load_dotenv()

# Globale Zustände
recording         = False
stop_listening    = None
last_audio        = None
robot_methods     = []
ie_instance       = None          # für OpenAI IE
detected_objects  = []            # List of detected objects
selected_object   = None          # Currently selected object

# === WORKFLOW STATUS MANAGEMENT ===
class WorkflowStatus:
   READY_FOR_DETECTION = "BEREIT: Objekte erfassen"
   READY_FOR_COMMANDS  = "BEREIT: Sprachbefehle"
   READY_FOR_EXECUTION = "BEREIT: Ausführung"
   PROCESSING          = "VERARBEITUNG..."
   SIMULATING          = "SIMULATION AKTIV"

current_status = WorkflowStatus.READY_FOR_DETECTION

def update_workflow_status(status):
   global current_status
   current_status = status
   status_label.config(text=f"Status: {status}")
   update_button_states()

def update_button_states():
   """Update button states based on current workflow status"""
   # Detection button - always enabled
   btn_detect.config(state="normal")
   # Recording button - enabled only if objects detected
   btn_record.config(state="normal" if detected_objects else "disabled")
   # Execute button - enabled only if robot methods exist
   btn_execute.config(state="normal" if robot_methods else "disabled")

def save_command_history():
   if ie_instance:
       history_file = os.path.join("data", "command_history_multi.json")
       os.makedirs(os.path.dirname(history_file), exist_ok=True)
       with open(history_file, "w", encoding="utf-8") as f:
           json.dump(ie_instance.get_command_history(), f, ensure_ascii=False, indent=2)

def load_command_history():
   """Load command history from file and populate ie_instance"""
   try:
       if os.path.exists("data/command_history_multi.json"):
           with open("data/command_history_multi.json", "r", encoding="utf-8") as f:
               history = json.load(f)
               # Add commands to ie_instance history
               if ie_instance and hasattr(ie_instance, 'command_history') and ie_instance.command_history:
                   for cmd in history:
                       ie_instance.command_history.add_command(
                           cmd["command"], cmd["result"], cmd["executed"]
                       )
               update_command_history_display()
   except Exception as e:
       print(f"Error loading command history: {e}")

def update_command_history_display():
   if ie_instance:
       history_text.delete("1.0", tk.END)
       for i, cmd in enumerate(ie_instance.get_command_history()):
           status = "SUCCESS" if cmd["executed"] else "PENDING"
           history_text.insert(tk.END, f"{i+1}. [{status}] {cmd['command']}\n")
           history_text.insert(
               tk.END,
               f"   Result: {json.dumps(cmd['result'], ensure_ascii=False)}\n\n"
           )

def update_object_display():
   """Update the object selection display with current detection results"""
   object_text.delete("1.0", tk.END)
   if detected_objects:
       object_text.insert(tk.END, f"TARGET: Detected Objects ({len(detected_objects)} total):\n")
       object_text.insert(tk.END, "=" * 50 + "\n\n")
       # Group objects by class
       objects_by_class = {}
       for obj in detected_objects:
           class_name = obj['class_name']
           objects_by_class.setdefault(class_name, []).append(obj)
       for class_name, objects in objects_by_class.items():
           if len(objects) == 1:
               obj = objects[0]
               object_text.insert(tk.END, f"• {class_name} (confidence: {obj['confidence']:.2f})\n\n")
           else:
               object_text.insert(tk.END, f"• {class_name} ({len(objects)} objects):\n")
               for i, obj in enumerate(objects):
                   object_text.insert(tk.END, f"  [{i}] Confidence: {obj['confidence']:.2f}\n")
               object_text.insert(tk.END, "\n")
       # Show selected object
       if selected_object:
           object_text.insert(tk.END, "TARGET: Currently Selected:\n")
           object_text.insert(
               tk.END,
               f"   {selected_object['class_name']} (confidence: {selected_object['confidence']:.2f})\n"
           )
   else:
       object_text.insert(tk.END, "No objects detected.\nClick 'Capture & Detect Objects' to start.")

def handle_clarification_needed(info):
   clarification_text.delete("1.0", tk.END)
   if info.get("needs_clarification"):
       msg = "Bitte kläre folgende Infos:\n"
       for f in info.get("clarification_fields", []):
           msg += f"- {f}\n"
       if "error_message" in info:
           msg += f"\nFehler: {info['error_message']}\n"
       clarification_text.insert(tk.END, msg)
       return True
   return False

def capture_and_detect_objects():
   """Enhanced capture with simulation mode support"""
   global detected_objects, ie_instance

   update_workflow_status(WorkflowStatus.PROCESSING)
   output_text.delete("1.0", tk.END)
   output_text.insert(tk.END, "DETECTION: Starting object detection...\n")

   try:
       if exec_mode.get() == "simulate":
           # Set environment variable for simulation mode
           os.environ['DETECTION_MODE'] = 'simulation'
           output_text.insert(tk.END, "SIMULATION: Setting detection mode to simulation\n")
           
           # Simulationspfad: Testbild verwenden
           output_text.insert(tk.END, "SIMULATION: Using test image instead of camera\n")
           test_image_path = os.path.join(PRE, "test_photo", "test_photo.jpg")
           if not os.path.exists(test_image_path):
               output_text.insert(tk.END, f"ERROR: Test image not found: {test_image_path}\n")
               update_workflow_status(WorkflowStatus.READY_FOR_DETECTION)
               return
           photos_dir = os.path.join(PRE, "photos")
           os.makedirs(photos_dir, exist_ok=True)
           photo_path = os.path.join(photos_dir, "photo_1.jpg")
           shutil.copy2(test_image_path, photo_path)
           output_text.insert(tk.END, f"SIMULATION: Copied to: {photo_path}\n")
           detection_script = os.path.join(PRE, "detection_multi.py")
           # Use the PRE venv Python for correct cv2 environment
           PRE_PYTHON = get_pre_venv_python()
           output_text.insert(tk.END, f"SIMULATION: Running detection with Python: {PRE_PYTHON}\n")
           output_text.insert(tk.END, f"SIMULATION: Detection script: {detection_script}\n")
           result = subprocess.run(
               [PRE_PYTHON, detection_script],
               cwd=PRE, capture_output=True, text=True, timeout=180,
               encoding='utf-8', errors='ignore'
           )
       else:
           # Set environment variable for real mode
           os.environ['DETECTION_MODE'] = 'real'
           output_text.insert(tk.END, "REAL: Setting detection mode to real camera\n")
           
           # Realmodus: Roboter in Ausgangsposition fahren und echte Kamera verwenden
           output_text.insert(tk.END, "REAL: Moving robot to main position before detection\n")
           try:
               from src.robot_control import move_to_main_position
               move_to_main_position(robot_ip.get())
           except Exception as e:
               output_text.insert(tk.END, f"ERROR: move_to_main_position failed: {e}\n")

           # Clear photos directory
           photos_dir = os.path.join(PRE, "photos")
           if os.path.exists(photos_dir):
               for fn in os.listdir(photos_dir):
                   path_fn = os.path.join(photos_dir, fn)
                   if os.path.isfile(path_fn):
                       os.remove(path_fn)
               output_text.insert(tk.END, f"REAL: Cleared photos directory ({photos_dir})\n")

           output_text.insert(tk.END, "CAMERA: Using real camera for detection\n")
           detection_script = os.path.join(PRE, "detection_multi.py")
           # Use the PRE venv Python for correct cv2 environment
           PRE_PYTHON = get_pre_venv_python()
           output_text.insert(tk.END, f"REAL: Running detection with Python: {PRE_PYTHON}\n")
           output_text.insert(tk.END, f"REAL: Detection script: {detection_script}\n")
           result = subprocess.run(
               [PRE_PYTHON, detection_script],
               cwd=PRE, capture_output=True, text=True,
               encoding='utf-8', errors='ignore'
           )

       # Debug output from detection script
       if result.stdout:
           output_text.insert(tk.END, f"DETECTION OUTPUT:\n{result.stdout}\n")
       if result.stderr:
           output_text.insert(tk.END, f"DETECTION STDERR:\n{result.stderr}\n")

       # Ergebnis prüfen
       if result.returncode != 0:
           output_text.insert(tk.END, f"ERROR: Detection failed with return code {result.returncode}\n")
           update_workflow_status(WorkflowStatus.READY_FOR_DETECTION)
           return

       # Ergebnisse laden
       json_path = os.path.join(PRE, "txt_file", "detected_objects.json")
       if os.path.exists(json_path):
           with open(json_path, "r", encoding="utf-8") as f:
               detection_data = json.load(f)
           detected_objects = detection_data.get("objects", [])
           if detected_objects:
               output_text.insert(tk.END, f"SUCCESS: Detected {len(detected_objects)} objects:\n")
               for obj in detected_objects:
                   output_text.insert(
                       tk.END,
                       f"  - {obj['class_name']} (conf: {obj['confidence']:.2f})\n"
                   )
               update_object_display()
               update_workflow_status(WorkflowStatus.READY_FOR_COMMANDS)
               if ie_instance:
                   ie_instance.load_available_objects()
               output_text.insert(tk.END, "\nReady for voice commands!\n")
           else:
               output_text.insert(tk.END, "ERROR: No objects detected. Please adjust camera or lighting.\n")
               update_object_display()
               update_workflow_status(WorkflowStatus.READY_FOR_DETECTION)
       else:
           output_text.insert(tk.END, f"ERROR: Detection results file not found: {json_path}\n")
           update_workflow_status(WorkflowStatus.READY_FOR_DETECTION)

   except Exception as e:
       output_text.insert(tk.END, f"ERROR: Detection error: {e}\n")
       import traceback
       output_text.insert(tk.END, f"ERROR DETAILS:\n{traceback.format_exc()}\n")
       update_workflow_status(WorkflowStatus.READY_FOR_DETECTION)
   finally:
       # Clean up environment variable
       if 'DETECTION_MODE' in os.environ:
           del os.environ['DETECTION_MODE']

def callback(recognizer, audio):
   global last_audio
   last_audio = audio
   print("Audio captured.")

def toggle_recording():
   global recording, stop_listening, last_audio, robot_methods, ie_instance, selected_object

   if not recording:
       if not detected_objects:
           messagebox.showwarning("No Objects", "Please detect objects first using 'Capture & Detect Objects'")
           return
       recording = True
       btn_record.config(text="Stop Recording")
       update_workflow_status(WorkflowStatus.PROCESSING)
       output_text.delete("1.0", tk.END)
       output_text.insert(tk.END, "MICROPHONE: Recording...\n")
       idx = mic_mapping.get(mic_var.get(), 0)
       recognizer = sr.Recognizer()
       mic = sr.Microphone(device_index=idx)
       with mic as source:
           recognizer.adjust_for_ambient_noise(source, duration=1)
       stop_listening = recognizer.listen_in_background(
           mic, callback, phrase_time_limit=5
       )
   else:
       if stop_listening:
           stop_listening(wait_for_stop=False)
       recording = False
       btn_record.config(text="Start Recording")
       if not last_audio:
           output_text.insert(tk.END, "No audio captured.\n")
           update_workflow_status(WorkflowStatus.READY_FOR_COMMANDS)
           return
       temp_file = os.path.join("data", "audio", "live_audio.wav")
       os.makedirs(os.path.dirname(temp_file), exist_ok=True)
       with open(temp_file, "wb") as f:
           f.write(last_audio.get_wav_data())
       output_text.insert(tk.END, f"Audio saved to {temp_file}\n")
       try:
           stt  = SpeechToTextLocal()
           text = stt.transcribe(temp_file)
           output_text.insert(tk.END, "\nTranscribed Text:\n" + text + "\n")
       except Exception as e:
           output_text.insert(tk.END, f"Transcription error: {e}\n")
           update_workflow_status(WorkflowStatus.READY_FOR_COMMANDS)
           return
       try:
           if ie_instance is None:
               ie_instance = InformationExtractionOpenAIMulti()
           info = ie_instance.extract(text, command_type.get())
           if handle_clarification_needed(info):
               update_workflow_status(WorkflowStatus.READY_FOR_COMMANDS)
               return
           output_text.insert(
               tk.END,
               "\nExtracted Information:\n" +
               json.dumps(info, indent=2, ensure_ascii=False) + "\n"
           )
           update_command_history_display()
           save_command_history()
       except Exception as e:
           output_text.insert(tk.END, f"Extraction error: {e}\n")
           update_workflow_status(WorkflowStatus.READY_FOR_COMMANDS)
           return
       try:
           # **IMPORTANT: Change to precision workflow**
           robot_methods[:] = select_robot_methods_multi(info, use_precision=True)
           
           # DEBUG: PRECISION MODE ANALYSIS
           print(f"DEBUG PRECISION: robot_methods = {robot_methods}")
           print(f"DEBUG PRECISION: 'suction_on' in methods = {'suction_on' in robot_methods}")
           if 'suction_on' in robot_methods:
               print(f"DEBUG PRECISION: suction_on is step {robot_methods.index('suction_on') + 1}/{len(robot_methods)}")
           else:
               print("DEBUG PRECISION: *** WARNING: suction_on NOT FOUND in robot_methods ***")
           
           if robot_methods:
               print(f"SUCCESS: Robot workflow generated with {len(robot_methods)} steps")
           output_text.insert(
               tk.END,
               "\nSelected Robot Methods:\n" +
               "\n".join(robot_methods) + "\n"
           )
           if info.get('object') and 'object_index' in info:
               object_type = info['object']
               object_index = info.get('object_index', 0)
               objs = [o for o in detected_objects if o['class_name'].lower() == object_type.lower()]
               if objs and object_index < len(objs):
                   selected_object = objs[object_index]
                   update_object_display()
                   
                   # **NEW: Write selection_data.json immediately after audio processing**
                   txt_dir = os.path.join(PRE, 'txt_file')
                   os.makedirs(txt_dir, exist_ok=True)
                   selection_data = {
                       "selected_object_id": selected_object['id'],
                       "selected_object_class": selected_object['class_name'],
                       "selected_object_confidence": selected_object['confidence'],
                       "selection_timestamp": str(time.time()),
                       # NEUE FELDER für bessere Objekt-Verknüpfung:
                       "original_center_x": selected_object['center'][0],
                       "original_center_y": selected_object['center'][1],
                       "original_bbox": selected_object['bbox'],
                       "original_confidence": selected_object['confidence'],
                       "selection_phase": "overview"  # Unterscheidung zwischen Overview und Precision
                   }
                   
                   selection_data_path = os.path.join(txt_dir, "selection_data.json")
                   with open(selection_data_path, 'w') as f:
                       json.dump(selection_data, f, indent=2)
                   output_text.insert(tk.END, f" SELECTION: Object selection saved to selection_data.json\n")
                   print(f"DEBUG: Wrote selection_data.json for object ID {selected_object['id']}")
           update_workflow_status(WorkflowStatus.READY_FOR_EXECUTION)
       except Exception as e:
           output_text.insert(tk.END, f"Robot method selection error: {e}\n")
           update_workflow_status(WorkflowStatus.READY_FOR_COMMANDS)
           return

def execute_workflow_handler():
    global detected_objects, selected_object
    
    def simulation_output_callback(message):
        output_text.insert("end", message + "\n")
        output_text.see("end")
        app.update_idletasks()

    try:
        # Check if robot methods exist
        if not robot_methods:
            messagebox.showwarning("Warnung", "Keine Roboter-Methoden ausgewählt!")
            return
        
        # Clear output first
        output_text.delete(1.0, "end")
        
        # **NEW: Execute complete object processing pipeline before robot workflow**
        output_text.insert("end", "PIPELINE: Starting complete object processing pipeline...\n")
        
        # Get the last command info to determine target object
        current_command_info = None
        if ie_instance and hasattr(ie_instance, 'command_history'):
            try:
                current_command_info = ie_instance.get_command_history()[-1] if ie_instance.get_command_history() else None
            except:
                current_command_info = None
        
        if not current_command_info:
            output_text.insert("end", " ERROR: No command information found. Please give a voice command first.\n")
            return
        
        # **NEW: Re-detect objects before execution to ensure target is still present**
        output_text.insert("end", "VERIFICATION: Re-detecting objects to verify target is still present...\n")
        
        # Set detection mode and re-run detection
        detection_mode = "simulation" if exec_mode.get() == "simulate" else "real"
        os.environ['DETECTION_MODE'] = detection_mode
        
        try:
            # Re-run detection
            capture_and_detect_objects()
            
            if not detected_objects:
                output_text.insert("end", " ERROR: No objects detected during verification. Workflow aborted.\n")
                return
            
            output_text.insert("end", f" VERIFICATION: Found {len(detected_objects)} objects\n")
            
        except Exception as detection_error:
            output_text.insert("end", f" DETECTION ERROR: {str(detection_error)}\n")
            return
        finally:
            # Clean up environment variable
            if 'DETECTION_MODE' in os.environ:
                del os.environ['DETECTION_MODE']
        
        # **REMOVED: execute_complete_object_pipeline - new precision workflow handles this**
        # The precision workflow now handles all object processing internally
        
        if exec_mode.get() == "real":
            # Real robot execution with automatic re-detection
            update_workflow_status(WorkflowStatus.PROCESSING)
            output_text.insert("end", "ROBOT REAL: Executing on real robot...\n")
            
            try:
                # Import robot control here to avoid circular imports
                from src.robot_control import execute_robot_workflow, move_to_selected_object
                
                # **REMOVED: Doppelte Bewegung - wird bereits im Workflow ausgeführt**
                # move_to_selected_object(robot_ip.get()) 
                
                # Execute the robot workflow
                output_text.insert("end", "ROBOT REAL: Starting robot workflow...\n")
                execute_robot_workflow(robot_ip.get(), robot_methods)
                output_text.insert("end", " ROBOT REAL: Workflow successfully completed!\n")
                
                # Workflow erfolgreich beendet - keine finale Detection
                update_workflow_status(WorkflowStatus.READY_FOR_COMMANDS)
                
            except Exception as robot_error:
                output_text.insert("end", f" ERROR REAL: Error during robot execution: {str(robot_error)}\n")
                update_workflow_status(WorkflowStatus.READY_FOR_EXECUTION)
                messagebox.showerror("Robot Error", f"Robot execution failed: {str(robot_error)}")
        
        else:
            # Simulation Mode (unchanged)
            update_workflow_status(WorkflowStatus.SIMULATING)
            output_text.insert("end", "SIMULATION: Starting workflow simulation...\n")
            
            try:
                from src.robot_control import execute_robot_workflow_simulation, move_to_selected_object
                
                output_text.insert("end", "SIMULATION: Starting workflow simulation...\n")
                execute_robot_workflow_simulation(
                    robot_ip.get(), robot_methods, simulation_output_callback
                )
                output_text.insert("end", " SIMULATION: Workflow simulation completed successfully!\n")
                update_workflow_status(WorkflowStatus.READY_FOR_COMMANDS)
            except Exception as sim_error:
                output_text.insert("end", f" SIMULATION ERROR: {str(sim_error)}\n")
                update_workflow_status(WorkflowStatus.READY_FOR_EXECUTION)
                messagebox.showerror("Simulation Error", f"Simulation failed: {str(sim_error)}")
                
    except Exception as e:
        output_text.insert("end", f" SYSTEM ERROR: {str(e)}\n")
        update_workflow_status(WorkflowStatus.READY_FOR_EXECUTION)
        messagebox.showerror("System Error", f"System error: {str(e)}")
    
    finally:
        # Ensure buttons are re-enabled
        update_button_states()

def write_object_data_for_robot(selected_obj):
    """
    Write object data in the format expected by the robot control system
    Creates both new JSON format and legacy txt files for compatibility
    """
    try:
        # Use the TXT_FILE_DIR path from PRE
        txt_dir = os.path.join(PRE, 'txt_file')
        
        # Ensure directory exists
        os.makedirs(txt_dir, exist_ok=True)
        
        # Write center point data for the selected object
        center_x, center_y = selected_obj["center"]
        
        # Ensure coordinates are integers for pixel coordinates
        center_x = int(round(center_x))
        center_y = int(round(center_y))
        
        print(f"DEBUG: Writing center coordinates: ({center_x}, {center_y})")
        
        # Legacy single-object files (for pixel2robot.py compatibility)
        center_point_path = os.path.join(txt_dir, "center_point.txt")
        with open(center_point_path, 'w') as f:
            f.write(f"{center_x},{center_y}\n")
        print(f"DEBUG: Wrote legacy center_point.txt")
        
        # Write label data for the selected object
        label_path = os.path.join(txt_dir, "label.txt")
        bbox = selected_obj["bbox"]  # [x1, y1, x2, y2]
        confidence = selected_obj["confidence"]
        class_id = selected_obj["class"]
        
        with open(label_path, 'w') as f:
            # Format: class_id x1 y1 x2 y2 confidence
            f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {confidence}\n")
        print(f"DEBUG: Wrote legacy label.txt")
        
        # Write label path reference
        label_path_file = os.path.join(txt_dir, "label_path.txt")
        with open(label_path_file, 'w') as f:
            f.write(label_path)
        print(f"DEBUG: Wrote label_path.txt")
        
        # IMPORTANT: Write crop image path for the selected object (required for PCA)
        crop_img_path_obj = os.path.join(txt_dir, f"crop_img_path_obj_{selected_obj['id']}.txt")
        crop_img_path_main = os.path.join(txt_dir, "crop_img_path.txt")
        
        # Check if the specific object crop path exists
        if os.path.exists(crop_img_path_obj):
            with open(crop_img_path_obj, 'r') as f:
                crop_path = f.read().strip()
            
            # Copy to main crop path for legacy compatibility
            with open(crop_img_path_main, 'w') as f:
                f.write(crop_path)
            print(f"DEBUG: Wrote legacy crop_img_path.txt with path: {crop_path}")
        else:
            print(f"WARNING: Crop image path for object {selected_obj['id']} not found at {crop_img_path_obj}")
        
        # Multi-object specific files (keep existing functionality)
        center_point_obj_path = os.path.join(txt_dir, f"center_point_object_{selected_obj['id']}.txt")
        with open(center_point_obj_path, 'w') as f:
            f.write(f"{center_x},{center_y}\n")
        print(f"DEBUG: Wrote center_point_object_{selected_obj['id']}.txt")
        
        label_obj_path = os.path.join(txt_dir, f"label_object_{selected_obj['id']}.txt")
        with open(label_obj_path, 'w') as f:
            f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {confidence}\n")
        print(f"DEBUG: Wrote label_object_{selected_obj['id']}.txt")
        
        # Verify files were created and have content
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
        
        # NEW: Create selection data for multi-object scripts
        # Note: target_location and action will be set later in pipeline
        selection_data = {
            "selected_object_id": selected_obj['id'],
            "selected_object_class": selected_obj['class_name'],
            "selected_object_confidence": selected_obj['confidence'],
            "selection_timestamp": str(time.time())
        }
        
        selection_data_path = os.path.join(txt_dir, "selection_data.json")
        with open(selection_data_path, 'w') as f:
            json.dump(selection_data, f, indent=2)
        print(f"DEBUG: Wrote selection_data.json for object ID {selected_obj['id']}")
        
        print(f"SUCCESS: Object data written to txt files")
        print(f"   - Center: ({center_x}, {center_y})")
        print(f"   - Bbox: {bbox}")
        print(f"   - Class: {class_id} ({selected_obj['class_name']})")
        print(f"   - Confidence: {confidence:.3f}")
        print(f"   - Object ID: {selected_obj['id']}")
        print(f"   - Legacy files created for pixel2robot.py compatibility")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to write object data: {e}")
        import traceback
        traceback.print_exc()
        return False

def execute_complete_object_pipeline(command_info, output_text):
    """
    Execute the complete object processing pipeline using HYBRID APPROACH:
    - Direct imports for simple calculations (pixel2robot, pca, direction)
    - Subprocess for YOLOv5-specific scripts (detection.py) with correct venv
    
    Pipeline steps (1:1 from Application.py):
    1. Select target object from speech command
    2. Write object data to legacy txt files
    3. Run detection.py (legacy detection) - SUBPROCESS with work venv
    4. Convert pixel coordinates to robot coordinates - DIRECT IMPORT
    5. Execute PCA calculation - DIRECT IMPORT
    6. Execute direction calculation - DIRECT IMPORT
    7. Verify robot coordinates were generated
    8. Execute robot workflow
    """
    try:
        print("PIPELINE: Starting HYBRID object processing pipeline...")
        
        # STEP 1: Select target object from speech command
        print("STEP 1/6: Selecting target object from speech command...")
        
        # Select target object based on command (function loads available_objects internally)
        selected_obj = select_target_object(command_info)
        if not selected_obj:
            print(" STEP 1/6: Object selection failed")
            return False
            
        print(f" STEP 1/6: Selected object: {selected_obj['class_name']} #{selected_obj.get('object_id', 0)} (confidence: {selected_obj['confidence']:.2f})")
        
        # STEP 2: Write object data to legacy txt files
        print("STEP 2/6: Writing object data to legacy txt files (predecessor format)...")
        write_object_data_for_robot(selected_obj)
        print(" STEP 2/6: Object data written to legacy txt files")
        
        # STEP 3: Run legacy detection.py - SUBPROCESS with correct venv
        print("STEP 3/6: Running legacy detection.py (predecessor script) - SUBPROCESS with work venv...")
        success = run_legacy_detection_subprocess()
        if not success:
            print(" STEP 3/6: Legacy detection failed")
            return False
        print(" STEP 3/6: Legacy detection completed")
        
        # STEP 4: Convert pixel coordinates to robot coordinates - DIRECT IMPORT
        print("STEP 4/6: Converting pixel to robot coordinates - DIRECT IMPORT...")
        success = run_pixel2robot_direct()
        if not success:
            print(" STEP 4/6: Pixel2robot conversion failed")
            return False
        print(" STEP 4/6: Pixel2robot conversion completed")
        
        # STEP 5: Execute PCA calculation - DIRECT IMPORT
        print("STEP 5/6: Executing PCA calculation - DIRECT IMPORT...")
        success = run_pca_direct()
        if not success:
            print(" STEP 5/6: PCA calculation failed")
            return False
        print(" STEP 5/6: PCA calculation completed")
        
        # STEP 6: Execute direction calculation - DIRECT IMPORT
        print("STEP 6/6: Executing direction calculation - DIRECT IMPORT...")
        success = run_direction_direct()
        if not success:
            print(" STEP 6/6: Direction calculation failed")
            return False
        print(" STEP 6/6: Direction calculation completed")
        
        # Verify robot coordinates were generated
        robot_coords_file = os.path.join(PRE, 'txt_file', 'robot_coordinates.txt')
        if not os.path.exists(robot_coords_file):
            print(" ERROR: Robot coordinates not generated")
            return False
            
        print(" SUCCESS: Complete object processing pipeline completed")
        print(" Robot coordinates generated and ready for workflow execution")
        return True
        
    except Exception as e:
        print(f" ERROR: Object processing pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_legacy_detection_subprocess():
    """
    Run legacy detection.py using subprocess with the correct virtual environment (work)
    This avoids the cv2 import issue by using the YOLOv5 environment
    """
    try:
        # Path to the work venv python executable
        work_venv_python = os.path.join(PRE, 'work', 'bin', 'python')
        if os.name == 'nt':  # Windows
            work_venv_python = os.path.join(PRE, 'work', 'Scripts', 'python.exe')
        
        # Fallback: try to find python in work directory
        if not os.path.exists(work_venv_python):
            work_venv_python = 'python'  # Use system python as fallback
        
        # Legacy detection script path
        detection_script = os.path.join(PRE, 'detection.py')
        
        # Change to the predecessor directory for execution
        original_cwd = os.getcwd()
        os.chdir(PRE)
        
        try:
            # Run detection.py with proper encoding and timeout
            result = subprocess.run(
                [work_venv_python, 'detection.py'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=30  # 30 seconds timeout
            )
            
            if result.returncode == 0:
                print("DEBUG: Legacy detection.py completed successfully")
                return True
            else:
                print(f"DEBUG: Legacy detection.py failed with return code: {result.returncode}")
                print(f"STDERR: {result.stderr}")
                return False
                
        finally:
            # Always restore original directory
            os.chdir(original_cwd)
            
    except subprocess.TimeoutExpired:
        print("DEBUG: Legacy detection.py timed out")
        return False
    except Exception as e:
        print(f"DEBUG: Legacy detection.py error: {e}")
        return False


def run_pixel2robot_direct():
    """
    Run pixel2robot calculation directly by importing the module
    This avoids subprocess overhead for simple calculations
    """
    try:
        # Import and run pixel2robot directly
        sys.path.insert(0, PRE)
        
        # Try to import and execute the main functionality
        import pixel2robot
        
        # The pixel2robot.py script should process the txt files automatically
        # Just importing it should trigger the calculation
        print("DEBUG: pixel2robot calculation completed")
        return True
        
    except Exception as e:
        print(f"DEBUG: pixel2robot direct import failed: {e}")
        # Fallback to subprocess if direct import fails
        return run_script_subprocess('pixel2robot.py')


def run_pca_direct():
    """
    Run PCA calculation directly by importing the module
    """
    try:
        # Import and run pca directly
        sys.path.insert(0, PRE)
        
        import pca
        
        print("DEBUG: PCA calculation completed")
        return True
        
    except Exception as e:
        print(f"DEBUG: PCA direct import failed: {e}")
        # Fallback to subprocess if direct import fails
        return run_script_subprocess('pca.py')


def run_direction_direct():
    """
    Run direction calculation directly by importing the module
    """
    try:
        # Import and run direction directly
        sys.path.insert(0, PRE)
        
        import direction
        
        print("DEBUG: Direction calculation completed")
        return True
        
    except Exception as e:
        print(f"DEBUG: Direction direct import failed: {e}")
        # Fallback to subprocess if direct import fails
        return run_script_subprocess('direction.py')


def run_script_subprocess(script_name):
    """
    Fallback function to run a script via subprocess if direct import fails
    Uses the main venv (venv_wrapper) for non-cv2 scripts
    """
    try:
        script_path = os.path.join(PRE, script_name)
        
        # Change to the predecessor directory for execution
        original_cwd = os.getcwd()
        os.chdir(PRE)
        
        try:
            result = subprocess.run(
                ['python', script_name],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=15
            )
            
            if result.returncode == 0:
                print(f"DEBUG: {script_name} subprocess completed successfully")
                return True
            else:
                print(f"DEBUG: {script_name} subprocess failed: {result.stderr}")
                return False
                
        finally:
            os.chdir(original_cwd)
            
    except Exception as e:
        print(f"DEBUG: {script_name} subprocess error: {e}")
        return False

# --- GUI-Aufbau ---
app = tk.Tk()
app.title("Multi-Object Robot Command via Speech")

main_frame = ttk.Frame(app, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)

# Left frame - Controls
left_frame = ttk.Frame(main_frame)
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))

# === WORKFLOW STATUS DISPLAY ===
status_frame = ttk.LabelFrame(left_frame, text="Workflow Status", padding="5")
status_frame.pack(fill=tk.X, pady=(0,10))

status_label = ttk.Label(status_frame, text=f"Status: {current_status}", foreground="blue")
status_label.pack()

workflow_help = ttk.Label(
   status_frame,
   text="1. Detect Objects → 2. Voice Command → 3. Execute",
   font=("Arial", 8), foreground="gray"
)
workflow_help.pack()

ttk.Label(left_frame, text="Enter Robot IP:").pack(pady=(10,0))
robot_ip = tk.StringVar(app, value="192.168.2.180")
ttk.Entry(left_frame, textvariable=robot_ip, width=20).pack(pady=(0,10))

ttk.Label(left_frame, text="Befehlstyp:").pack(anchor="w", pady=(10,0), padx=10)
command_type = tk.StringVar(app, value="new")
ttk.Radiobutton(left_frame, text="Neuer Befehl", variable=command_type, value="new").pack(anchor="w", padx=20)
ttk.Radiobutton(left_frame, text="Korrektur", variable=command_type, value="correction").pack(anchor="w", padx=20)

ttk.Label(left_frame, text="Execution Mode:").pack(anchor="w", pady=(10,0), padx=10)
exec_mode = tk.StringVar(app, value="simulate")
ttk.Radiobutton(left_frame, text="Simulate", variable=exec_mode, value="simulate").pack(anchor="w", padx=20)
ttk.Radiobutton(left_frame, text="Real Robot", variable=exec_mode, value="real").pack(anchor="w", padx=20)

ttk.Label(left_frame, text="Select Microphone:").pack(pady=(10,0))
mic_names   = sr.Microphone.list_microphone_names()
mic_mapping = {name: idx for idx, name in enumerate(mic_names)}
mic_var     = tk.StringVar(app, value=mic_names[0] if mic_names else "")
ttk.OptionMenu(left_frame, mic_var, *mic_names).pack(pady=(0,10))

# === ENHANCED BUTTONS WITH STATE MANAGEMENT ===
button_frame = ttk.LabelFrame(left_frame, text="Controls", padding="5")
button_frame.pack(fill=tk.X, pady=(0,10))

btn_detect = ttk.Button(
   button_frame,
   text="Capture & Detect Objects",
   command=capture_and_detect_objects,
   width=25
)
btn_detect.pack(pady=2)

btn_record = ttk.Button(
   button_frame,
   text="Start Recording",
   command=toggle_recording,
   width=25,
   state="disabled"
)
btn_record.pack(pady=2)

btn_execute = ttk.Button(
   button_frame,
   text="Execute Robot Workflow",
   command=execute_workflow_handler,
   width=25,
   state="disabled"
)
btn_execute.pack(pady=2)

# Right frame - Displays
right_frame = ttk.Frame(main_frame)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

ttk.Label(right_frame, text="Detected Objects:").pack(anchor="w")
object_text = tk.Text(right_frame, height=8, width=60, bg="#E8F5E8")
object_text.pack(fill=tk.X, pady=(0,10))

ttk.Label(right_frame, text="Klärungshinweise:").pack(anchor="w")
clarification_text = tk.Text(right_frame, height=3, width=60, bg="#FFF3CD")
clarification_text.pack(fill=tk.X, pady=(0,10))

ttk.Label(right_frame, text="Command History:").pack(anchor="w")
history_text = tk.Text(right_frame, height=10, width=60)
history_text.pack(fill=tk.X, pady=(0,10))

ttk.Label(right_frame, text="Current Output:").pack(anchor="w")
output_text = tk.Text(right_frame, height=15, width=60)
output_text.pack(fill=tk.BOTH, expand=True)

# Initialize displays and button states
update_object_display()
update_button_states()

# Load command history after GUI is set up
app.after(100, load_command_history)

# Add usage instructions to object display initially
if not detected_objects:
   object_text.insert(tk.END, """
Multi-Object Robot Control System

Instructions:
1. WÄHLEN Sie den Execution Mode (Simulate/Real Robot)
2. KLICKEN Sie 'Capture & Detect Objects'
3. VERWENDEN Sie Sprachbefehle zur Objektauswahl
4. FÜHREN Sie den Robot Workflow aus

Examples:
• "Bewege Zylinder zu Zone 1"
• "Nimm ersten Marker zu Zone 2"
• "Bewege zweiten Quader zu Zone 3"

SIMULATION MODE: Verwendet test_photo.jpg statt Kamera
REAL MODE: Verwendet echte Kamera für Detection
""")

app.mainloop()
