# MO_Changes
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
import unicodedata


def _tk_safe_text(value: str) -> str:
   if not sys.platform.startswith("linux"):
       return value
   replacements = str.maketrans(
       {
           "ä": "ae",
           "ö": "oe",
           "ü": "ue",
           "Ä": "Ae",
           "Ö": "Oe",
           "Ü": "Ue",
           "ß": "ss",
           "→": " to ",
           "←": " from ",
           "↓": " then ",
           "↑": " up ",
           "•": "*",
       }
   )
   normalized = unicodedata.normalize("NFKD", value.translate(replacements))
   return normalized.encode("ascii", "ignore").decode("ascii")


class CrossPlatformText(tk.Text):
   def insert(self, index: str, chars: str, *args: str) -> None:
       super().insert(index, _tk_safe_text(chars), *args)

# Multi-Object System Imports
from src.speech.speech_to_text_local import SpeechToTextLocal
from src.speech.information_extraction_openai_api_multi import InformationExtractionOpenAIMulti
from src.speech.microphone_devices import discover_input_microphones
from src.detection_preparation import get_default_robot_ip, prepare_robot_for_detection
from src.robot_method_selector_multi import select_robot_methods_multi, select_target_object
from src.zone_coordinates import get_zone_coordinates
from src.multimodal import (
   GestureProcessClient,
   OperatorFeedback,
   extract_zone,
   is_affirmative,
   is_drop_here,
   is_negative,
   resolve_multimodal_selection,
)

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

def get_pre_venv_python(platform_choice="windows"):
   """Path to the Code-YOLOv5-Windows_llm venv interpreter for the chosen platform.

   Detection needs the pinned cv2/torch versions in that subproject's own
   venv. platform_choice comes from the Path Select control so the user
   picks Windows, macOS or Linux explicitly instead of relying on
   auto-detection. Windows keeps its original hardcoded path unchanged.
   Falls back to get_yolo_python() when that venv layout is not present,
   which is always true for macOS/Linux right now since the venv there
   was created on Windows.
   """
   platform_python_paths = {
       "windows": os.path.join(PRE, "venv", "Scripts", "python.exe"),
       "mac": os.path.join(PRE, "venv", "bin", "python"),
       "linux": os.path.join(PRE, "venv", "bin", "python"),
   }
   candidate = platform_python_paths.get(platform_choice)
   if candidate and os.path.exists(candidate):
       return candidate
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
gesture_client    = GestureProcessClient(ROOT, display=True)
gesture_start_error = None
pending_command_info = None
gesture_poll_job = None

# === WORKFLOW STATUS MANAGEMENT ===
class WorkflowStatus:
   READY_FOR_DETECTION = "BEREIT: Objekte erfassen"
   READY_FOR_COMMANDS  = "BEREIT: Sprachbefehle"
   READY_FOR_EXECUTION = _tk_safe_text("BEREIT: Ausführung")
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

def update_robot_ip():
   robot_ip.set(get_default_robot_ip(robot_type.get()))

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

def load_overview_detection_data():
   detection_path = os.path.join(PRE, "txt_file", "detected_objects.json")
   with open(detection_path, "r", encoding="utf8") as detection_file:
       return json.load(detection_file)

def publish_multimodal_rejection(reason):
   feedback = OperatorFeedback(
       lambda message: (
           output_text.insert(tk.END, message),
           output_text.see(tk.END),
       )
   )
   feedback.rejection(reason)

def capture_and_detect_objects():
   """Enhanced capture with simulation mode support"""
   global detected_objects, ie_instance

   update_workflow_status(WorkflowStatus.PROCESSING)
   output_text.delete("1.0", tk.END)
   output_text.insert(tk.END, "DETECTION: Starting object detection...\n")

   previous_robot_type = os.environ.get("ROBOT_TYPE")
   os.environ["ROBOT_TYPE"] = robot_type.get()

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
           PRE_PYTHON = get_pre_venv_python(platform_select.get())
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
           output_text.insert(tk.END, "REAL: Preparing selected robot for detection\n")
           try:
               prepare_robot_for_detection(robot_type.get(), robot_ip.get())
           except Exception as e:
               output_text.insert(tk.END, f"ERROR: Robot detection preparation failed: {e}\n")
               update_workflow_status(WorkflowStatus.READY_FOR_DETECTION)
               return

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
           PRE_PYTHON = get_pre_venv_python(platform_select.get())
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
       if previous_robot_type is None:
           os.environ.pop("ROBOT_TYPE", None)
       else:
           os.environ["ROBOT_TYPE"] = previous_robot_type

def callback(recognizer, audio):
   global last_audio
   last_audio = audio
   print("Audio captured.")

def _operator_feedback():
   return OperatorFeedback(
       lambda message: (
           output_text.insert(tk.END, message),
           output_text.see(tk.END),
           app.update_idletasks(),
       )
   )


def _transcribe_audio(audio, filename):
   audio_path = os.path.join("data", "audio", filename)
   os.makedirs(os.path.dirname(audio_path), exist_ok=True)
   with open(audio_path, "wb") as audio_file:
       audio_file.write(audio.get_wav_data())
   return SpeechToTextLocal().transcribe(audio_path)


def _request_click_confirmation(prompt, reason):
   feedback = _operator_feedback()
   feedback.publish("Speech confirmation was not recognized. Use the confirmation window.")
   approved = messagebox.askyesno(
       "Robot Confirmation",
       _tk_safe_text(
           f"{prompt}\n\n"
           f"Speech confirmation status: {reason}\n\n"
           "Click Yes to approve this robot action or No to cancel."
       ),
       parent=app,
   )
   if approved:
       print("MULTIMODAL CONFIRMATION: Accepted with interface Yes button")
       feedback.publish("Command confirmed with the interface Yes button.")
       return True
   print("MULTIMODAL CONFIRMATION: Rejected with interface No button")
   feedback.publish("Command cancelled with the interface No button.")
   return False


def _listen_for_confirmation(prompt):
   feedback = _operator_feedback()
   idx = mic_mapping.get(mic_var.get())
   if idx is None:
       return _request_click_confirmation(prompt, "microphone unavailable")
   attempts = ((prompt, 10.0), ("Please answer yes or no.", 5.0))
   for attempt_prompt, timeout_seconds in attempts:
       feedback.publish(attempt_prompt, wait_for_speech=True)
       recognizer = sr.Recognizer()
       try:
           with sr.Microphone(device_index=idx) as source:
               audio = recognizer.listen(
                   source,
                   timeout=timeout_seconds,
                   phrase_time_limit=3.0,
               )
           answer = _transcribe_audio(audio, "confirmation.wav")
           print(f"MULTIMODAL CONFIRMATION: {answer}")
       except sr.WaitTimeoutError:
           continue
       except Exception as error:
           feedback.publish(f"Confirmation could not be captured. {error}")
           return _request_click_confirmation(prompt, "microphone capture failed")
       if is_affirmative(answer):
           feedback.publish("Command confirmed.")
           return True
       if is_negative(answer):
           feedback.publish("Command cancelled.")
           return False
   return _request_click_confirmation(prompt, "spoken answer not recognized")


def _write_selection_data(info, gesture_result):
   global selected_object
   object_type = str(info.get("object", ""))
   object_index = int(info.get("object_index", 0))
   matching = [
       item
       for item in detected_objects
       if item["class_name"].lower() == object_type.lower()
   ]
   if info.get("selection_mode") != "gesture":
       if object_index < 0 or object_index >= len(matching):
           raise ValueError("Selected object index is not available")
       selected_object = matching[object_index]
   if selected_object is None:
       raise ValueError("No object was selected")
   txt_dir = os.path.join(PRE, "txt_file")
   os.makedirs(txt_dir, exist_ok=True)
   selection_data = {
       "selected_object_id": selected_object["id"],
       "selected_object_class": selected_object["class_name"],
       "selected_object_confidence": selected_object["confidence"],
       "selection_timestamp": str(time.time()),
       "original_center_x": selected_object["center"][0],
       "original_center_y": selected_object["center"][1],
       "original_bbox": selected_object["bbox"],
       "original_confidence": selected_object["confidence"],
       "selection_phase": "overview",
       "selection_source": info.get("selection_mode", "speech"),
       "gesture_session_id": info.get("gesture_session_id"),
       "fingertip_pixel": gesture_result.get("fingertip_pixel"),
   }
   with open(os.path.join(txt_dir, "selection_data.json"), "w") as selection_file:
       json.dump(selection_data, selection_file, indent=2)
   update_object_display()


def _confirmation_prompt(info):
   name = selected_object["class_name"]
   number = int(info.get("object_index", 0)) + 1
   target = str(info.get("target_location") or "")
   object_name = f"{name} {number}"
   if target:
       spoken_target = target.replace("_", " ")
       return f"Do you want me to pick up {object_name} and place it in {spoken_target}?"
   return f"Do you want me to pick up {object_name}?"


def _gesture_result_is_fresh(gesture_result, maximum_age_seconds=1.5):
   observed_at = gesture_result.get("last_seen_at_unix_s")
   return isinstance(observed_at, (int, float)) and time.time() - observed_at <= maximum_age_seconds


def _prepare_command(text, gesture_result):
   global ie_instance, selected_object, pending_command_info
   if ie_instance is None:
       ie_instance = InformationExtractionOpenAIMulti()
   gesture_only = not text.strip()
   if gesture_only:
       info = {
           "intent": "nehmen",
           "target_location": "",
           "action": "pick",
           "object": None,
           "object_index": 0,
           "selection_mode": "gesture",
           "needs_clarification": False,
           "clarification_fields": [],
       }
       resolution_text = "pick up this object"
   else:
       info = ie_instance.extract(text, command_type.get())
       resolution_text = text
   detection_data = load_overview_detection_data()
   multimodal = resolve_multimodal_selection(
       resolution_text,
       info,
       gesture_result,
       detection_data,
   )
   if multimodal.required and not multimodal.accepted:
       publish_multimodal_rejection(multimodal.reason)
       return False
   if multimodal.required and not _gesture_result_is_fresh(gesture_result):
       publish_multimodal_rejection("fingertip_not_detected")
       return False
   if multimodal.required and multimodal.selected_object is not None:
       selected_object = multimodal.selected_object
       info["object"] = selected_object["class_name"]
       info["object_index"] = multimodal.object_index
       info["selection_mode"] = "gesture"
       info["gesture_session_id"] = gesture_result.get("session_id")
   resolved_fields = set()
   if multimodal.required and multimodal.selected_object is not None:
       resolved_fields.add("object")
   clarification_fields = [
       field
       for field in (info.get("clarification_fields") or [])
       if field not in resolved_fields
   ]
   info["clarification_fields"] = clarification_fields
   info["needs_clarification"] = bool(clarification_fields)
   if handle_clarification_needed(info):
       return False
   robot_methods[:] = select_robot_methods_multi(info, use_precision=True)
   if not robot_methods:
       return False
   _write_selection_data(info, gesture_result)
   output_text.insert(
       tk.END,
       "\nExtracted Information:\n"
       + json.dumps(info, indent=2, ensure_ascii=False)
       + "\n\nSelected Robot Methods:\n"
       + "\n".join(robot_methods)
       + "\n",
   )
   pending_command_info = info
   if not _listen_for_confirmation(_confirmation_prompt(info)):
       robot_methods.clear()
       pending_command_info = None
       selected_object = None
       update_object_display()
       return False
   update_command_history_display()
   save_command_history()
   return True


def _finish_recording():
   global recording, stop_listening, last_audio, gesture_poll_job
   if not recording:
       return
   recording = False
   if gesture_poll_job is not None:
       app.after_cancel(gesture_poll_job)
       gesture_poll_job = None
   if stop_listening:
       stop_listening(wait_for_stop=True)
       stop_listening = None
   btn_record.config(text="Start Recording")
   gesture_result = gesture_client.finish() if gesture_start_error is None else {
       "status": "error",
       "reason": "gesture_process_not_started",
       "safe_to_use": False,
   }
   print("MULTIMODAL: Gesture result " + json.dumps(gesture_result, ensure_ascii=False))
   text = ""
   if last_audio is not None:
       try:
           text = _transcribe_audio(last_audio, "live_audio.wav")
           output_text.insert(tk.END, "\nTranscribed Text:\n" + text + "\n")
       except Exception as error:
           output_text.insert(tk.END, f"Transcription error: {error}\n")
           update_workflow_status(WorkflowStatus.READY_FOR_COMMANDS)
           return
   elif float(gesture_result.get("hold_seconds", 0.0)) < 10.0:
       output_text.insert(tk.END, "No speech or stable gesture was captured.\n")
       update_workflow_status(WorkflowStatus.READY_FOR_COMMANDS)
       return
   try:
       prepared = _prepare_command(text, gesture_result)
   except Exception as error:
       output_text.insert(tk.END, f"Command preparation error: {error}\n")
       prepared = False
   update_workflow_status(
       WorkflowStatus.READY_FOR_EXECUTION
       if prepared
       else WorkflowStatus.READY_FOR_COMMANDS
   )


def _poll_gesture_session():
   global gesture_poll_job
   if not recording:
       return
   result = gesture_client.latest_result()
   if result is not None and _gesture_result_is_fresh(result):
       held_seconds = float(result.get("hold_seconds", 0.0))
       if held_seconds >= 10.0:
           _finish_recording()
           return
   gesture_poll_job = app.after(200, _poll_gesture_session)


def toggle_recording():
   global recording, stop_listening, last_audio, gesture_start_error
   global selected_object, pending_command_info, gesture_poll_job
   if recording:
       _finish_recording()
       return
   if not detected_objects:
       messagebox.showwarning("No Objects", "Please detect objects first")
       return
   idx = mic_mapping.get(mic_var.get())
   if idx is None:
       messagebox.showerror("Microphone unavailable", "No valid input microphone was found")
       return
   robot_methods.clear()
   selected_object = None
   pending_command_info = None
   last_audio = None
   gesture_start_error = None
   update_object_display()
   gesture_outcome = {}

   def start_gesture_capture():
       try:
           session = gesture_client.start(selection_kind="object", hold_seconds=5.0)
           gesture_outcome["session"] = session
       except Exception as error:
           gesture_outcome["error"] = str(error)

   gesture_thread = threading.Thread(target=start_gesture_capture, daemon=True)
   gesture_thread.start()
   gesture_thread.join(timeout=25.0)
   if gesture_thread.is_alive():
       gesture_start_error = "gesture process startup timed out"
       gesture_client.cancel()
       output_text.insert(
           tk.END,
           "MULTIMODAL: Gesture camera startup timed out.\n",
       )
       update_workflow_status(WorkflowStatus.READY_FOR_COMMANDS)
       return
   gesture_start_error = gesture_outcome.get("error")
   if gesture_start_error:
       print(f"MULTIMODAL: Gesture process could not start. {gesture_start_error}")
       gesture_client.cancel()
       output_text.insert(
           tk.END,
           f"MULTIMODAL: Gesture process could not start. {gesture_start_error}\n",
       )
       update_workflow_status(WorkflowStatus.READY_FOR_COMMANDS)
       return
   session = gesture_outcome["session"]
   print(f"MULTIMODAL: Gesture capture is running for session {session.session_id}")

   try:
       print(f"MICROPHONE: Starting selected input index {idx}")
       recognizer = sr.Recognizer()
       microphone = sr.Microphone(device_index=idx)
       with microphone as source:
           recognizer.adjust_for_ambient_noise(source, duration=1)
       stop_listening = recognizer.listen_in_background(
           microphone,
           callback,
           phrase_time_limit=5,
       )
       print("MICROPHONE: Background listener started")
   except Exception as error:
       gesture_client.cancel()
       output_text.insert(tk.END, f"MICROPHONE: Could not start the selected input. {error}\n")
       update_workflow_status(WorkflowStatus.READY_FOR_COMMANDS)
       return
   recording = True
   btn_record.config(text="Stop Recording")
   update_workflow_status(WorkflowStatus.PROCESSING)
   output_text.delete("1.0", tk.END)
   output_text.insert(tk.END, "MULTIMODAL: Camera and microphone are active.\n")
   gesture_poll_job = app.after(200, _poll_gesture_session)


def _transform_destination_point(gesture_result):
   fingertip = gesture_result.get("fingertip_pixel")
   frame_width = int(gesture_result.get("frame_width") or 0)
   frame_height = int(gesture_result.get("frame_height") or 0)
   if not isinstance(fingertip, list) or len(fingertip) != 2:
       raise ValueError("Destination fingertip pixel is missing")
   if frame_width <= 0 or frame_height <= 0:
       raise ValueError("Destination camera size is missing")
   pixel_x, pixel_y = float(fingertip[0]), float(fingertip[1])
   if robot_type.get() == "franka":
       from src.FR_franka import transform_franka_pixel_to_robot

       point = transform_franka_pixel_to_robot(
           pixel_x,
           pixel_y,
           frame_width,
           frame_height,
       )
       print(
           "FRANKA DESTINATION TRANSFORM: "
           f"u={pixel_x:.1f}, v={pixel_y:.1f}, x={point.x:.5f}, y={point.y:.5f}"
       )
       return point.x, point.y
   detection_data = load_overview_detection_data()
   metadata = detection_data.get("metadata", {})
   target_width = int(metadata.get("image_width") or frame_width)
   target_height = int(metadata.get("image_height") or frame_height)
   calibrated_u = pixel_x * target_width / frame_width
   calibrated_v = pixel_y * target_height / frame_height
   from src.robot_control import transform_ur_pixel_to_robot

   x_robot, y_robot = transform_ur_pixel_to_robot(calibrated_u, calibrated_v)
   print(
       "UR DESTINATION TRANSFORM: "
       f"u={calibrated_u:.1f}, v={calibrated_v:.1f}, "
       f"x={x_robot:.5f}, y={y_robot:.5f}"
   )
   return x_robot, y_robot


def _collect_destination_methods():
   from src.robot_method_selector_multi import (
       build_precision_place_methods,
       build_precision_point_place_methods,
   )

   feedback = _operator_feedback()
   idx = mic_mapping.get(mic_var.get())
   if idx is None:
       raise RuntimeError("A microphone is required to select a destination")
   destination_audio = []

   def destination_callback(recognizer, audio):
       destination_audio.append(audio)
       print("MULTIMODAL DESTINATION: Audio captured")

   while True:
       destination_audio.clear()
       session = gesture_client.start(selection_kind="location", hold_seconds=3.0)
       print(f"MULTIMODAL DESTINATION: Gesture session {session.session_id}")
       recognizer = sr.Recognizer()
       microphone = sr.Microphone(device_index=idx)
       with microphone as source:
           recognizer.adjust_for_ambient_noise(source, duration=0.5)
       stop_destination_audio = recognizer.listen_in_background(
           microphone,
           destination_callback,
           phrase_time_limit=5,
       )
       reminder_at = time.monotonic() + 120.0
       chosen_zone = None
       chosen_point = None
       try:
           while chosen_zone is None and chosen_point is None:
               app.update()
               gesture_result = gesture_client.latest_result()
               if gesture_result is not None and not _gesture_result_is_fresh(gesture_result):
                   gesture_result = None
               if destination_audio:
                   audio = destination_audio.pop(0)
                   try:
                       text = _transcribe_audio(audio, "destination.wav")
                   except Exception as error:
                       print(f"MULTIMODAL DESTINATION: Transcription failed. {error}")
                       text = ""
                   print(f"MULTIMODAL DESTINATION SPEECH: {text}")
                   chosen_zone = extract_zone(text)
                   if chosen_zone is None and is_drop_here(text):
                       if gesture_result is None:
                           feedback.publish("Point at the destination and say drop here again.")
                       else:
                           chosen_point = gesture_result
               if (
                   chosen_zone is None
                   and chosen_point is None
                   and gesture_result is not None
                   and float(gesture_result.get("hold_seconds", 0.0)) >= 10.0
               ):
                   chosen_point = gesture_result
               if time.monotonic() >= reminder_at:
                   feedback.publish(
                       "Would you like to drop the object? Point at a place or say which zone."
                   )
                   reminder_at = time.monotonic() + 120.0
               time.sleep(0.1)
       finally:
           stop_destination_audio(wait_for_stop=True)
           gesture_client.finish()
       if chosen_zone is not None:
           prompt = f"Do you want me to place the object in {chosen_zone.replace('_', ' ')}?"
           if _listen_for_confirmation(prompt):
               return build_precision_place_methods(chosen_zone)
           feedback.publish("Destination cancelled. Please choose again.")
           continue
       if chosen_point is not None:
           if _listen_for_confirmation("Do you want me to drop the object there?"):
               x_robot, y_robot = _transform_destination_point(chosen_point)
               return build_precision_point_place_methods(x_robot, y_robot)
           feedback.publish("Destination cancelled. Please point or speak again.")

def execute_workflow_handler():
    global detected_objects, selected_object, pending_command_info
    
    def simulation_output_callback(message):
        output_text.insert("end", message + "\n")
        output_text.see("end")
        app.update_idletasks()

    try:
        # Check if robot methods exist
        if not robot_methods:
            messagebox.showwarning(
                "Warnung",
                _tk_safe_text("Keine Roboter-Methoden ausgewählt!"),
            )
            return
        
        # Clear output first
        output_text.delete(1.0, "end")
        
        # **NEW: Execute complete object processing pipeline before robot workflow**
        output_text.insert("end", "PIPELINE: Starting complete object processing pipeline...\n")
        
        current_command_info = pending_command_info
        
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
        
        destination_known = any(
            method.startswith("move_to_target") or method.startswith("move_to_point")
            for method in robot_methods
        )

        if exec_mode.get() == "real":
            # Real robot execution with automatic re-detection
            update_workflow_status(WorkflowStatus.PROCESSING)
            output_text.insert("end", "ROBOT REAL: Executing on real robot...\n")
            
            try:
                output_text.insert("end", "ROBOT REAL: Starting robot workflow...\n")
                if robot_type.get() == "franka":
                    from src.FR_franka import (
                        create_franka_workflow_session,
                        execute_franka_workflow,
                    )

                    if destination_known:
                        execute_franka_workflow(
                            robot_methods,
                            robot_ip=robot_ip.get(),
                            simulation=False,
                        )
                    else:
                        with create_franka_workflow_session(
                            robot_ip=robot_ip.get(),
                            simulation=False,
                        ) as franka_session:
                            franka_session.execute(robot_methods)
                            output_text.insert(
                                "end",
                                "MULTIMODAL: Object is held at the intermediate position.\n",
                            )
                            place_methods = _collect_destination_methods()
                            franka_session.execute(place_methods)
                else:
                    from src.robot_control import execute_robot_workflow

                    execute_robot_workflow(
                        robot_ip.get(),
                        robot_methods,
                        return_home=destination_known,
                    )
                if not destination_known and robot_type.get() != "franka":
                    output_text.insert(
                        "end",
                        "MULTIMODAL: Object is held at the intermediate position.\n",
                    )
                    place_methods = _collect_destination_methods()
                    execute_robot_workflow(
                        robot_ip.get(),
                        place_methods,
                        return_home=True,
                    )
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
                output_text.insert("end", "SIMULATION: Starting workflow simulation...\n")
                if robot_type.get() == "franka":
                    from src.FR_franka import (
                        create_franka_workflow_session,
                        execute_franka_workflow,
                    )

                    if destination_known:
                        execute_franka_workflow(
                            robot_methods,
                            robot_ip=robot_ip.get(),
                            simulation=True,
                            output_callback=simulation_output_callback,
                        )
                    else:
                        with create_franka_workflow_session(
                            robot_ip=robot_ip.get(),
                            simulation=True,
                            output_callback=simulation_output_callback,
                        ) as franka_session:
                            franka_session.execute(robot_methods)
                            output_text.insert(
                                "end",
                                "MULTIMODAL: Simulated object is at the intermediate position.\n",
                            )
                            place_methods = _collect_destination_methods()
                            franka_session.execute(place_methods)
                else:
                    from src.robot_control import execute_robot_workflow_simulation

                    execute_robot_workflow_simulation(
                        robot_ip.get(),
                        robot_methods,
                        simulation_output_callback,
                        return_home=destination_known,
                    )
                if not destination_known and robot_type.get() != "franka":
                    output_text.insert(
                        "end",
                        "MULTIMODAL: Simulated object is at the intermediate position.\n",
                    )
                    place_methods = _collect_destination_methods()
                    execute_robot_workflow_simulation(
                        robot_ip.get(),
                        place_methods,
                        simulation_output_callback,
                        return_home=True,
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
        
        # Legacy single-object files (for UR_pixel2robot.py compatibility)
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
        print(f"   - Legacy files created for UR_pixel2robot.py compatibility")
        
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
    - Subprocess for YOLOv5-specific scripts (UR_detection.py) with correct venv
    
    Pipeline steps (1:1 from NU_Application.py):
    1. Select target object from speech command
    2. Write object data to legacy txt files
    3. Run UR_detection.py (legacy detection) - SUBPROCESS with work venv
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
        
        # STEP 3: Run legacy UR_detection.py - SUBPROCESS with correct venv
        print("STEP 3/6: Running legacy UR_detection.py (predecessor script) - SUBPROCESS with work venv...")
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
    Run legacy UR_detection.py using subprocess with the correct virtual environment (work)
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
        detection_script = os.path.join(PRE, 'UR_detection.py')
        
        # Change to the predecessor directory for execution
        original_cwd = os.getcwd()
        os.chdir(PRE)
        
        try:
            # Run UR_detection.py with proper encoding and timeout
            result = subprocess.run(
                [work_venv_python, 'UR_detection.py'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=30  # 30 seconds timeout
            )
            
            if result.returncode == 0:
                print("DEBUG: Legacy UR_detection.py completed successfully")
                return True
            else:
                print(f"DEBUG: Legacy UR_detection.py failed with return code: {result.returncode}")
                print(f"STDERR: {result.stderr}")
                return False
                
        finally:
            # Always restore original directory
            os.chdir(original_cwd)
            
    except subprocess.TimeoutExpired:
        print("DEBUG: Legacy UR_detection.py timed out")
        return False
    except Exception as e:
        print(f"DEBUG: Legacy UR_detection.py error: {e}")
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
        import UR_pixel2robot as pixel2robot
        
        # The UR_pixel2robot.py script should process the txt files automatically
        # Just importing it should trigger the calculation
        print("DEBUG: pixel2robot calculation completed")
        return True
        
    except Exception as e:
        print(f"DEBUG: pixel2robot direct import failed: {e}")
        # Fallback to subprocess if direct import fails
        return run_script_subprocess('UR_pixel2robot.py')


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
        return run_script_subprocess('UR_pca.py')


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
        return run_script_subprocess('UR_direction.py')


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

def close_application():
   gesture_client.cancel()
   app.destroy()

app.protocol("WM_DELETE_WINDOW", close_application)

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
   text=_tk_safe_text("1. Detect Objects → 2. Voice Command → 3. Execute"),
   font=("Arial", 8), foreground="gray"
)
workflow_help.pack()

ttk.Label(left_frame, text="Robot Type:").pack(anchor="w", pady=(10,0), padx=10)
robot_type = tk.StringVar(app, value="franka")
ttk.Radiobutton(
   left_frame,
   text="Franka Emika",
   variable=robot_type,
   value="franka",
   command=update_robot_ip,
).pack(anchor="w", padx=20)
ttk.Radiobutton(
   left_frame,
   text="Universal Robot",
   variable=robot_type,
   value="universal",
   command=update_robot_ip,
).pack(anchor="w", padx=20)

ttk.Label(left_frame, text="Enter Robot IP:").pack(pady=(10,0))
robot_ip = tk.StringVar(app, value=get_default_robot_ip(robot_type.get()))
ttk.Entry(left_frame, textvariable=robot_ip, width=20).pack(pady=(0,10))

ttk.Label(left_frame, text="Befehlstyp:").pack(anchor="w", pady=(10,0), padx=10)
command_type = tk.StringVar(app, value="new")
ttk.Radiobutton(left_frame, text="Neuer Befehl", variable=command_type, value="new").pack(anchor="w", padx=20)
ttk.Radiobutton(left_frame, text="Korrektur", variable=command_type, value="correction").pack(anchor="w", padx=20)

ttk.Label(left_frame, text="Execution Mode:").pack(anchor="w", pady=(10,0), padx=10)
exec_mode = tk.StringVar(app, value="simulate")
ttk.Radiobutton(left_frame, text="Simulate", variable=exec_mode, value="simulate").pack(anchor="w", padx=20)
ttk.Radiobutton(left_frame, text="Real Robot", variable=exec_mode, value="real").pack(anchor="w", padx=20)

ttk.Label(left_frame, text="Path Select:").pack(anchor="w", pady=(10,0), padx=10)
default_platform = "windows" if os.name == "nt" else ("mac" if sys.platform == "darwin" else "linux")
platform_select = tk.StringVar(app, value=default_platform)
ttk.Radiobutton(left_frame, text="Windows", variable=platform_select, value="windows").pack(anchor="w", padx=20)
ttk.Radiobutton(left_frame, text="macOS", variable=platform_select, value="mac").pack(anchor="w", padx=20)
ttk.Radiobutton(left_frame, text="Linux", variable=platform_select, value="linux").pack(anchor="w", padx=20)

ttk.Label(left_frame, text="Select Microphone:").pack(pady=(10,0))
try:
   microphone_options, default_microphone_index = discover_input_microphones()
except Exception as error:
   print(f"MICROPHONE: Device discovery failed. {error}")
   microphone_options = []
   default_microphone_index = None

mic_mapping = {
   option.display_name: option.device_index for option in microphone_options
}
default_microphone_name = next(
   (
       option.display_name
       for option in microphone_options
       if option.device_index == default_microphone_index
   ),
   "No input microphone found",
)
mic_var = tk.StringVar(app, value=default_microphone_name)
ttk.OptionMenu(
   left_frame,
   mic_var,
   default_microphone_name,
   *mic_mapping.keys(),
).pack(pady=(0,10))

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
object_text = CrossPlatformText(right_frame, height=8, width=60, bg="#E8F5E8")
object_text.pack(fill=tk.X, pady=(0,10))

ttk.Label(
   right_frame,
   text=_tk_safe_text("Klärungshinweise:"),
).pack(anchor="w")
clarification_text = CrossPlatformText(
   right_frame,
   height=3,
   width=60,
   bg="#FFF3CD",
)
clarification_text.pack(fill=tk.X, pady=(0,10))

ttk.Label(right_frame, text="Command History:").pack(anchor="w")
history_text = CrossPlatformText(right_frame, height=10, width=60)
history_text.pack(fill=tk.X, pady=(0,10))

ttk.Label(right_frame, text="Current Output:").pack(anchor="w")
output_text = CrossPlatformText(right_frame, height=15, width=60)
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
