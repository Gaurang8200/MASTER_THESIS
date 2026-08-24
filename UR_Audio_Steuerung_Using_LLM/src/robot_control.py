# MO_Changes

import sys
import time
import os
import subprocess
import socket
import struct
import json
import shutil

from src.robot_output import emit_method_execution

# Projekt‐Root und Vorgänger‐Ordner
PROJECT_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PREDECESSOR_DIR = os.path.join(PROJECT_ROOT, "Code-YOLOv5-Windows_llm")
if PREDECESSOR_DIR not in sys.path:
    sys.path.insert(0, PREDECESSOR_DIR)

# Use the active project environment for every perception subprocess
PRE_PYTHON = sys.executable

# Global simulation mode flag
simulation_mode = False
simulation_output_callback = None

def set_simulation_mode(enabled=True, output_callback=None):
    """Enable/disable simulation mode with optional output callback"""
    global simulation_mode, simulation_output_callback
    simulation_mode = enabled
    simulation_output_callback = output_callback


def transform_ur_pixel_to_robot(pixel_x, pixel_y):
    previous_directory = os.getcwd()
    os.chdir(PREDECESSOR_DIR)
    try:
        import pixel2robot_multi

        x_robot, y_robot, _ = pixel2robot_multi.pixel2robot(pixel_x, pixel_y)
        return float(x_robot), float(y_robot)
    finally:
        os.chdir(previous_directory)

def sim_print(message: str) -> None:
    """Print simulation output to the terminal and optional interface."""
    print(message)
    if simulation_output_callback and simulation_output_callback is not print:
        simulation_output_callback(message + "\n")

# URScript‐Sende‐Funktion
def send_urscript(script, host):
    port = 30002
    if simulation_mode:
        sim_print(f"ROBOT SIMULATION: URScript would be sent to {host}:{port}")
        sim_print(f"SCRIPT: Script content:")
        sim_print(f"{'='*50}")
        sim_print(script.strip())
        sim_print(f"{'='*50}")
        return
    
    print(f"[DEBUG] send_urscript → Host={host}, Port={port}, Script={script.strip()}")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.send(script.encode('utf-8'))
    s.close()

# URScript‐Generatoren (1:1 aus Vorgänger)
def generate_urscript_movej(x, y, z, rx, ry, rz, a=0.1, v=0.1):
    return f"""
def move_to_position():
    movej(p[{x}, {y}, {z}, {rx}, {ry}, {rz}], a={a}, v={v})
    textmsg("Movement complete!")
end

move_to_position()
"""

def generate_urscript_movel(x, y, z, rx, ry, rz, a=0.1, v=0.1):
    return f"""
def move_to_position():
    movel(p[{x}, {y}, {z}, {rx}, {ry}, {rz}], a={a}, v={v})
    textmsg("Movement complete!")
end

move_to_position()
"""

def generate_urscript_movej_forward(x, y, z, rx, ry, rz, a=1.0, v=1.0):
    return f"""
def move_to_position():
    movej([{x}, {y}, {z}, {rx}, {ry}, {rz}], a={a}, v={v})
    textmsg("Movement complete!")
end

move_to_position()
"""

def generate_urscript_movel_forward(x, y, z, rx, ry, rz, a=1, v=1):
    return f"""
def move_to_position():
    movel([{x}, {y}, {z}, {rx}, {ry}, {rz}], a={a}, v={v})
    textmsg("Movement complete!")
end

move_to_position()
"""

# Greifer‐Funktionen
def suction_on(robot_ip):
    """
    Activates the suction gripper with enhanced diagnostics
    
    Standard UR configuration:
    - Digital output 0: Vacuum valve (False = open, True = closed) 
    - Digital output 1: Vacuum pump (True = on, False = off)
    
    Alternative configurations available via suction_on_alt()
    """
    # DEBUG: SUCTION FUNCTION ANALYSIS
    print(f"DEBUG SUCTION: *** suction_on called with robot_ip={robot_ip} ***")
    print(f"DEBUG SUCTION: simulation_mode={simulation_mode}")
    print(f"DEBUG SUCTION: About to generate and send URScript...")
    urscript = f"""
def start_suction():
    set_tool_digital_out(0, False)  # Turn off the vacuum pump
    sleep(2)  # Wait for 2 seconds
    set_tool_digital_out(1, True)  # Turn on the vacuum pump
    textmsg("Vacuum pump started!")
end

start_suction()
"""
    if simulation_mode:
        sim_print("ROBOT SIMULATION: Activating suction gripper")
        sim_print("SIMULATION: Digital Output 0 (valve) = False")
        sim_print("SIMULATION: Digital Output 1 (pump) = True")
    else:
        print(f"[DEBUG] suction_on → sending to {robot_ip}")
        print("[DEBUG] suction_on → Output 0 (valve): False, Output 1 (pump): True")
    
    send_urscript(urscript, robot_ip)

def suction_off(robot_ip):
    """
    Deactivates the suction gripper with enhanced diagnostics
    """
    urscript = f"""
def start_release():
    set_tool_digital_out(1, False)  # Turn off the vacuum pump
    sleep(2)  # Wait for 2 seconds
    set_tool_digital_out(0, True)  # Turn on the vacuum pump
    textmsg("Vacuum pump started!")
end

start_release()
"""
    if simulation_mode:
        sim_print("ROBOT SIMULATION: Deactivating suction gripper")
        sim_print("SIMULATION: Digital Output 1 (pump) = False")
        sim_print("SIMULATION: Digital Output 0 (valve) = True")
    else:
        print(f"[DEBUG] suction_off → sending to {robot_ip}")
        print("[DEBUG] suction_off → Output 1 (pump): False, Output 0 (valve): True")
    
    send_urscript(urscript, robot_ip)

def suction_on_alt(robot_ip, config="alt1"):
    """
    Alternative suction configurations for different hardware setups
    
    config options:
    - "alt1": Inverted logic (pump on output 0, valve on output 1)
    - "alt2": Single output control (only pump on output 0)
    - "alt3": Extended timing for slow valves
    """
    if config == "alt1":
        # Alternative 1: Inverted output assignment
        urscript = """
def start_suction_alt1():
    textmsg("DEBUG: ALT1 - Starting suction (inverted outputs)...")
    set_tool_digital_out(1, False)  # Valve on output 1
    sleep(0.5)
    set_tool_digital_out(0, True)   # Pump on output 0
    textmsg("ALT1: Suction activated (pump=0, valve=1)")
end
start_suction_alt1()
"""
    elif config == "alt2":
        # Alternative 2: Single output (pump only)
        urscript = """
def start_suction_alt2():
    textmsg("DEBUG: ALT2 - Starting suction (pump only)...")
    set_tool_digital_out(0, True)   # Only pump control
    textmsg("ALT2: Pump activated on output 0")
end
start_suction_alt2()
"""
    elif config == "alt3":
        # Alternative 3: Extended timing
        urscript = """
def start_suction_alt3():
    textmsg("DEBUG: ALT3 - Starting suction (extended timing)...")
    set_tool_digital_out(0, False)
    sleep(2.0)  # Extended wait
    set_tool_digital_out(1, True)
    sleep(3.0)  # Extended activation time
    textmsg("ALT3: Suction with extended timing")
end
start_suction_alt3()
"""
    
    if simulation_mode:
        sim_print(f"ROBOT SIMULATION: Activating suction with config {config}")
    else:
        print(f"[DEBUG] suction_on_alt → config: {config}, robot: {robot_ip}")
    
    send_urscript(urscript, robot_ip)

def diagnose_suction_system(robot_ip):
    """
    Comprehensive diagnostics for the suction system
    """
    urscript = """
def diagnose_suction():
    textmsg("=== SUCTION SYSTEM DIAGNOSTICS ===")
    
    # Check current digital output states
    out0_state = get_tool_digital_out(0)
    out1_state = get_tool_digital_out(1)
    textmsg("Current Output 0: " + str(out0_state))
    textmsg("Current Output 1: " + str(out1_state))
    
    # Test sequence: cycle through all combinations
    textmsg("Testing output combinations...")
    
    # Test 1: Both off
    set_tool_digital_out(0, False)
    set_tool_digital_out(1, False)
    sleep(1.0)
    textmsg("Test 1: Both OFF (0=False, 1=False)")
    
    # Test 2: Only output 0
    set_tool_digital_out(0, True)
    set_tool_digital_out(1, False)
    sleep(1.0)
    textmsg("Test 2: Output 0 ON (0=True, 1=False)")
    
    # Test 3: Only output 1
    set_tool_digital_out(0, False)
    set_tool_digital_out(1, True)
    sleep(1.0)
    textmsg("Test 3: Output 1 ON (0=False, 1=True)")
    
    # Test 4: Both on
    set_tool_digital_out(0, True)
    set_tool_digital_out(1, True)
    sleep(1.0)
    textmsg("Test 4: Both ON (0=True, 1=True)")
    
    # Reset to safe state
    set_tool_digital_out(0, False)
    set_tool_digital_out(1, False)
    textmsg("=== DIAGNOSTICS COMPLETE - Reset to safe state ===")
end

diagnose_suction()
"""
    
    if simulation_mode:
        sim_print("ROBOT SIMULATION: Running suction system diagnostics")
        sim_print("SIMULATION: Testing all digital output combinations")
        sim_print("SIMULATION: Check robot messages for detailed output")
    else:
        print(f"[DEBUG] diagnose_suction_system → running on {robot_ip}")
        print("[DEBUG] Check robot interface messages for detailed diagnostics")
    
    send_urscript(urscript, robot_ip)

# Roboter‐Status
def get_current_position(robot_ip):
    if simulation_mode:
        sim_print(f"ROBOT SIMULATION: Getting current position from {robot_ip}")
        # Return simulated position
        return [0.0, 0.0, 0.5, 3.057, -0.812, 0.028]
    
    port = 30003
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((robot_ip, port))
    data = s.recv(1108)
    s.close()
    return list(struct.unpack('!6d', data[444:492]))

def get_joint_angles(robot_ip):
    if simulation_mode:
        sim_print(f"ROBOT SIMULATION: Getting joint angles from {robot_ip}")
        # Return simulated joint angles
        return [-0.2553, -1.6563, 0.9641, -0.8604, -1.5900, 1.8328]
    
    port = 30003
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((robot_ip, port))
    data = s.recv(2048)
    s.close()
    return list(struct.unpack('!6d', data[252:300]))

def has_reached_position(current_pos, target_pos, threshold=0.005):
    if simulation_mode:
        sim_print(f"ROBOT SIMULATION: Checking if position reached (always True in simulation)")
        return True
    return all(abs(c - t) <= threshold for c, t in zip(current_pos[:3], target_pos[:3]))

def move_to_main_position(robot_ip):
    main_position = [-0.2553, -1.6563, 0.9641, -0.8604, -1.5900, 1.8328]
    if simulation_mode:
        sim_print(f"ROBOT SIMULATION: Moving to main position: {main_position}")
    else:
        print(f"[DEBUG] move_to_main_position → {main_position}")
    
    send_urscript(generate_urscript_movej_forward(*main_position), robot_ip)
    
    if simulation_mode:
        sim_print("SUCCESS SIMULATION: Reached main position")
        time.sleep(0.5)  # Short simulation delay
    else:
        while True:
            if has_reached_position(get_joint_angles(robot_ip), main_position):
                print("[DEBUG] move_to_main_position → reached")
                break
            time.sleep(0.1)

# globale Variablen für Pick & Place
x_robot = None
y_robot = None
z_pick   = None
x_place  = None
y_place  = None
z_place  = None

def move_to_object(robot_ip):
    global x_robot, y_robot
    base    = PREDECESSOR_DIR
    path_rc = os.path.join(base, 'txt_file', 'robot_coordinates.txt')
    
    if simulation_mode:
        sim_print(f"ROBOT SIMULATION: Reading robot coordinates from {path_rc}")
    else:
        print(f"[DEBUG] move_to_object → reading {path_rc}")
    
    if not os.path.exists(path_rc):
        raise FileNotFoundError(f"Datei nicht gefunden: {path_rc}")

    with open(path_rc, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    if simulation_mode:
        sim_print(f"INFO SIMULATION: Raw coordinates: {lines}")
    else:
        print(f"[DEBUG] move_to_object → raw lines = {lines}")

    if len(lines) < 2:
        raise RuntimeError(f"{path_rc} enthält <2 Zeilen: {len(lines)} gefunden")

    x_raw = float(lines[0])
    y_raw = float(lines[1])
    
    # Globale Variablen OHNE Offset setzen (für präzises Greifen)
    x_robot = x_raw
    y_robot = y_raw
    
    # Temporärer Kamera-Offset nur für Bewegung
    x_robot_offset = x_robot + 0.084
    
    if simulation_mode:
        sim_print(f"INFO SIMULATION: Raw coordinates: x={x_robot:.6f}, y={y_robot:.6f}")
        sim_print(f"INFO SIMULATION: Camera position with offset: x={x_robot_offset:.6f}, y={y_robot:.6f}")
    else:
        print(f"[DEBUG] move_to_object → x_robot={x_robot:.6f}, y_robot={y_robot:.6f}")
        print(f"[DEBUG] move_to_object → camera position with offset: x={x_robot_offset:.6f}")

    new_pos = [x_robot_offset, y_robot, 0.5, 3.057, -0.812, 0.028]
    
    if simulation_mode:
        sim_print(f"ROBOT SIMULATION: Moving to object position: {new_pos}")
    else:
        print(f"[DEBUG] move_to_object → moving to {new_pos}")
    
    send_urscript(generate_urscript_movel(*new_pos), robot_ip)

    if simulation_mode:
        sim_print("SUCCESS SIMULATION: Reached object position")
        time.sleep(0.5)
    else:
        while True:
            if has_reached_position(get_current_position(robot_ip), new_pos):
                print("[DEBUG] move_to_object → reached")
                break
            time.sleep(0.1)

def pick_the_object(robot_ip):
    global x_robot, y_robot, z_pick
    base    = PREDECESSOR_DIR
    
    # SCHRITT 1: Objektklassifizierung aus FINALER LABEL-DATEI (mit Fallback)
    final_label_file = os.path.join(base, 'txt_file', 'final_object_label.txt')
    label_file = os.path.join(base, 'txt_file', 'label.txt')
    
    if simulation_mode:
        sim_print(f"OBJECT CLASSIFICATION: Reading object class from final or standard label file")
    else:
        print(f"[DEBUG] pick_the_object → reading class from final or standard label")
    
    # Try to read from final label file first, then fallback to standard label.txt
    try:
        if os.path.exists(final_label_file):
            with open(final_label_file, 'r') as f:
                content = f.read().strip()
                if content:
                    cls = int(content.split()[0])
                    if simulation_mode:
                        sim_print(f"FINAL CLASSIFICATION: Using final_object_label.txt - class: {cls}")
                    else:
                        print(f"[DEBUG] pick_the_object → using final label, class_id={cls}")
                else:
                    raise ValueError("Final label file is empty")
        else:
            # Fallback to standard label.txt
            with open(label_file, 'r') as f:
                cls = int(f.read().split()[0])
                if simulation_mode:
                    sim_print(f"STANDARD CLASSIFICATION: Using standard label.txt - class: {cls}")
                else:
                    print(f"[DEBUG] pick_the_object → using standard label, class_id={cls}")
    except (FileNotFoundError, ValueError, IndexError):
        # Final fallback if both files fail
        cls = 0  # Default to class 0 (Zylinder)
        if simulation_mode:
            sim_print(f"FALLBACK CLASSIFICATION: Using default class: {cls}")
        else:
            print(f"[DEBUG] pick_the_object → using fallback class_id={cls}")
    
    # Set z_pick based on object class (same logic as before)
    z_pick = {0: 0.05, 1: 0.03, 2: 0.02}.get(cls)
    
    if simulation_mode:
        sim_print(f"INFO SIMULATION: Object class: {cls}, z_pick: {z_pick}")
    else:
        print(f"[DEBUG] pick_the_object → class_id={cls}, z_pick={z_pick}")

    rpypath = os.path.join(base, 'txt_file', 'robot_RPY.txt')
    
    if simulation_mode:
        sim_print(f"INFO SIMULATION: Reading RPY orientation from {rpypath}")
    else:
        print(f"[DEBUG] pick_the_object → reading RPY from {rpypath}")
    
    # Handle empty or missing RPY file
    try:
        raw = open(rpypath).read().strip()
        if not raw:
            # File is empty, use default values
            rx, ry = 3.057, -0.812
            if simulation_mode:
                sim_print("INFO SIMULATION: RPY file empty, using defaults: rx=3.057, ry=-0.812")
        else:
            clean = raw.strip("[]").replace(",", " ")
            parts = clean.split()
            if len(parts) >= 2:
                rx, ry = float(parts[0]), float(parts[1])
            else:
                rx, ry = 3.057, -0.812
                if simulation_mode:
                    sim_print("INFO SIMULATION: Insufficient RPY data, using defaults")
    except (FileNotFoundError, ValueError, IndexError):
        rx, ry = 3.057, -0.812
        if simulation_mode:
            sim_print("INFO SIMULATION: RPY file error, using defaults: rx=3.057, ry=-0.812")
    
    if simulation_mode:
        sim_print(f"INFO SIMULATION: RPY orientation: rx={rx:.3f}, ry={ry:.3f}")
    else:
        print(f"[DEBUG] pick_the_object → parsed rx={rx:.3f}, ry={ry:.3f}")

    pick_pos = [x_robot, y_robot, z_pick, rx, ry, 0.0]
    
    if simulation_mode:
        sim_print(f"ROBOT SIMULATION: Moving to pick position: {pick_pos}")
    else:
        print(f"[DEBUG] pick_the_object → moving to {pick_pos}")
    
    send_urscript(generate_urscript_movel(*pick_pos), robot_ip)

    if simulation_mode:
        sim_print("SUCCESS SIMULATION: Reached pick position")
        time.sleep(0.5)
    else:
        while True:
            if has_reached_position(get_current_position(robot_ip), pick_pos):
                print("[DEBUG] pick_the_object → reached")
                break
            time.sleep(0.1)
        time.sleep(5)  # INCREASED: Fix timing issue between workflow steps (was 1)

def pick_up_object(robot_ip):
    global x_robot, y_robot, z_pick
    base    = PREDECESSOR_DIR
    rpypath = os.path.join(base, 'txt_file', 'robot_RPY.txt')
    
    if simulation_mode:
        sim_print(f"ROBOT SIMULATION: Reading RPY for lift movement")
    else:
        print(f"[DEBUG] pick_up_object → reading RPY from {rpypath}")
    
    # Handle empty or missing RPY file
    try:
        raw = open(rpypath).read().strip()
        if not raw:
            # File is empty, use default values
            rx, ry = 3.057, -0.812
            if simulation_mode:
                sim_print("INFO SIMULATION: RPY file empty, using defaults: rx=3.057, ry=-0.812")
        else:
            clean = raw.strip("[]").replace(",", " ")
            parts = clean.split()
            if len(parts) >= 2:
                rx, ry = float(parts[0]), float(parts[1])
            else:
                rx, ry = 3.057, -0.812
                if simulation_mode:
                    sim_print("INFO SIMULATION: Insufficient RPY data, using defaults")
    except (FileNotFoundError, ValueError, IndexError):
        rx, ry = 3.057, -0.812
        if simulation_mode:
            sim_print("INFO SIMULATION: RPY file error, using defaults: rx=3.057, ry=-0.812")

    up_pos = [x_robot, y_robot, 0.2, rx, ry, 0.0]
    
    if simulation_mode:
        sim_print(f"ROBOT SIMULATION: Lifting object to: {up_pos}")
    else:
        print(f"[DEBUG] pick_up_object → moving to {up_pos}")
    
    send_urscript(generate_urscript_movel(*up_pos), robot_ip)

    if simulation_mode:
        sim_print("SUCCESS SIMULATION: Object lifted successfully")
        time.sleep(0.5)
    else:
        while True:
            if has_reached_position(get_current_position(robot_ip), up_pos):
                print("[DEBUG] pick_up_object → reached")
                break
            time.sleep(0.1)
        time.sleep(5)  # INCREASED: Fix timing issue between workflow steps (was 1)

def intermediate_position(robot_ip):
    inter = [-1.8497, -1.8064, 1.9345, -1.6989, -1.5687, -1.8500]
    
    if simulation_mode:
        sim_print(f"ROBOT SIMULATION: Moving to intermediate position: {inter}")
    else:
        print(f"[DEBUG] intermediate_position → moving to {inter}")
    
    send_urscript(generate_urscript_movej_forward(*inter), robot_ip)
    
    if simulation_mode:
        sim_print("SUCCESS SIMULATION: Reached intermediate position")
        time.sleep(0.5)
    else:
        while True:
            if has_reached_position(get_joint_angles(robot_ip), inter):
                print("[DEBUG] intermediate_position → reached")
                break
            time.sleep(0.1)

def final_position(robot_ip):
    global x_place, y_place, z_place
    
    if simulation_mode:
        sim_print(f"ROBOT SIMULATION: Moving to final position: x={x_place}, y={y_place}, z={z_place}")
    else:
        print(f"[DEBUG] final_position → using x_place={x_place}, y_place={y_place}, z_place={z_place}")
    
    final = [x_place, y_place, z_place, 2.221, 2.221, 0.0]
    
    if simulation_mode:
        sim_print(f"ROBOT SIMULATION: Final coordinates: {final}")
    else:
        print(f"[DEBUG] final_position → moving to {final}")
    
    send_urscript(generate_urscript_movel(*final), robot_ip)
    
    if simulation_mode:
        sim_print("SUCCESS SIMULATION: Reached final position")
        time.sleep(0.5)
    else:
        while True:
            if has_reached_position(get_current_position(robot_ip), final):
                print("[DEBUG] final_position → reached")
                break
            time.sleep(0.1)

def move_to_selected_object(robot_ip):
    """
    Bewegt Roboter zu dem in selection_data.json ausgewählten Objekt
    Verwendet pixel2robot_multi.py (nicht das alte pixel2robot.py)
    LOGIK: Identisch mit Vorgänger move_to_object(), aber verwendet spezifische Objekt-Koordinaten
    
    Args:
        robot_ip: IP-Adresse des Roboters ('192.168.2.180')
    """
    global x_robot, y_robot
    
    try:
        # 1. Lese selected_object_id aus selection_data.json
        selection_file = os.path.join(PREDECESSOR_DIR, 'txt_file', 'selection_data.json')
        with open(selection_file, 'r') as file:
            selection_data = json.load(file)
            selected_object_id = selection_data['selected_object_id']
            selected_object_class = selection_data['selected_object_class']
            
        if simulation_mode:
            sim_print(f"SELECTED OBJECT: {selected_object_class} (ID: {selected_object_id})")
        else:
            print(f"[DEBUG] move_to_selected_object → {selected_object_class} (ID: {selected_object_id})")
        
        # 2. Lese center_point_object_X.txt für das spezifische Objekt
        center_point_file = os.path.join(PREDECESSOR_DIR, 'txt_file', f'center_point_object_{selected_object_id}.txt')
        
        if not os.path.exists(center_point_file):
            raise FileNotFoundError(f"Center point file nicht gefunden: {center_point_file}")
        
        with open(center_point_file, 'r') as file:
            content = file.read().strip()
            if not content:
                raise ValueError(f"Center point file ist leer: {center_point_file}")
            
            # Parse Koordinaten (Format: "1530 696" oder "1530,696")
            if ',' in content:
                pixel_x, pixel_y = content.split(',')
            else:
                coords = content.split()
                pixel_x, pixel_y = coords[0], coords[1]
            
            pixel_x = int(float(pixel_x.strip()))
            pixel_y = int(float(pixel_y.strip()))
        
        if simulation_mode:
            sim_print(f"PIXEL COORDS: Objekt {selected_object_id} → ({pixel_x}, {pixel_y})")
        else:
            print(f"[DEBUG] move_to_selected_object → pixel coords: ({pixel_x}, {pixel_y})")
        
        # 3. Konvertiere mit pixel2robot_multi.py (GLEICHE Logik wie Vorgänger convert_pixel_to_robot)
        success = convert_pixel_to_robot_multi(pixel_x, pixel_y)
        if not success:
            raise RuntimeError("Pixel-zu-Robot-Koordinatenkonvertierung fehlgeschlagen")
        
        # 4. Lese robot_coordinates.txt (GLEICHE Logik wie Vorgänger move_to_object)
        robot_coords_file = os.path.join(PREDECESSOR_DIR, 'txt_file', 'robot_coordinates.txt')
        with open(robot_coords_file, 'r') as file:
            lines = [l.strip() for l in file.readlines() if l.strip()]
        
        if len(lines) < 2:
            raise RuntimeError(f"robot_coordinates.txt enthält <2 Zeilen: {len(lines)} gefunden")
        
        x_raw = float(lines[0])
        y_raw = float(lines[1])
        
        # 5. Globale Variablen OHNE Offset setzen (für präzises Greifen)
        x_robot = x_raw
        y_robot = y_raw
        
        # Temporärer Kamera-Offset für Precision-Detection
        x_robot_offset = x_robot + 0.084  # Offset for Camera and Tool
        
        if simulation_mode:
            sim_print(f"COORDINATES: Raw=({x_raw:.6f}, {y_raw:.6f}), Camera offset=({x_robot_offset:.6f}, {y_robot:.6f})")
        else:
            print(f"[DEBUG] move_to_selected_object → x_robot={x_robot:.6f}, y_robot={y_robot:.6f}")
            print(f"[DEBUG] move_to_selected_object → camera position with offset: x={x_robot_offset:.6f}")

        # 6. Bewege Roboter mit Kamera-Offset
        new_position = [x_robot_offset, y_robot, 0.5, 3.057, -0.812, 0.028]
        
        if simulation_mode:
            sim_print(f"MOVEMENT: Bewege zu Position: {new_position}")
        else:
            print(f"[DEBUG] move_to_selected_object → moving to {new_position}")
        
        send_urscript(generate_urscript_movel(*new_position), robot_ip)

        # 7. Warte bis Position erreicht (IDENTISCH mit Vorgänger)
        if simulation_mode:
            sim_print("SUCCESS: Roboter hat das ausgewählte Objekt erreicht")
            time.sleep(0.5)
        else:
            while True:
                current_position = get_current_position(robot_ip)
                if has_reached_position(current_position, new_position):
                    print("[DEBUG] move_to_selected_object → reached selected object")
                    break
                time.sleep(0.1)
            
        return True
        
    except Exception as e:
        error_msg = f"move_to_selected_object Fehler: {e}"
        if simulation_mode:
            sim_print(f"ERROR: {error_msg}")
        else:
            print(f"[ERROR] {error_msg}")
        return False

def convert_pixel_to_robot_multi(pixel_x, pixel_y):
    """
    Konvertiert Pixel-Koordinaten mit pixel2robot_multi.py 
    NEUE LÖSUNG: Kombiniert direkten Import + Kommandozeilen-Parameter (keine temporären Dateien)
    LOGIK: Identisch mit Vorgänger convert_pixel_to_robot(), aber eleganter implementiert
    """
    try:
        if simulation_mode:
            sim_print(f"CONVERSION: Verwende pixel2robot_multi.py für ({pixel_x}, {pixel_y})")
            # Mock-Konvertierung für Simulation
            robot_coords_file = os.path.join(PREDECESSOR_DIR, 'txt_file', 'robot_coordinates.txt')

            # Create realistic mock robot coordinates
            mock_x = 0.300 + (pixel_x - 1280) * 0.0003  # Scale factor für X
            mock_y = -0.050 + (pixel_y - 720) * 0.0002   # Scale factor für Y

            # Write mock coordinates to robot_coordinates.txt
            with open(robot_coords_file, 'w') as f:
                f.write(f"{mock_x:.6f}\n")
                f.write(f"{mock_y:.6f}\n")

            sim_print(f"MOCK CONVERSION: Created robot coordinates: x={mock_x:.3f}, y={mock_y:.3f}")
            sim_print(
                "SIMULATION COORDINATE RESULT:\n"
                f"[[{mock_x:.5f}]\n"
                f" [{mock_y:.5f}]\n"
                " [0.50000]\n"
                " [1.00000]]"
            )
            return True
        else:
            print(f"[DEBUG] convert_pixel_to_robot_multi → converting ({pixel_x}, {pixel_y})")
            
            # Wechsel zum PREDECESSOR_DIR für korrekte Pfade
            original_cwd = os.getcwd()
            os.chdir(PREDECESSOR_DIR)
            
            try:
                # OPTION 1: Direkter Import (schnell, keine temporären Dateien)
                try:
                    sys.path.insert(0, PREDECESSOR_DIR)
                    import pixel2robot_multi
                    
                    # Koordinaten direkt konvertieren
                    x_robot, y_robot = pixel2robot_multi.convert_coordinates(pixel_x, pixel_y)
                    
                    if x_robot is not None and y_robot is not None:
                        print(f"[DEBUG] convert_pixel_to_robot_multi → direct import success: ({x_robot:.5f}, {y_robot:.5f})")
                        return True
                    else:
                        print("[DEBUG] convert_pixel_to_robot_multi → direct import failed, trying subprocess")
                        raise ImportError("Direct import failed")
                        
                except (ImportError, Exception) as e:
                    print(f"[DEBUG] convert_pixel_to_robot_multi → direct import error: {e}")
                    
                    # OPTION 2: Subprocess mit Kommandozeilen-Parametern (keine temporären Dateien)
                    print("[DEBUG] convert_pixel_to_robot_multi → using subprocess with parameters")
                    
                    result = subprocess.run(
                        [PRE_PYTHON, 'pixel2robot_multi.py', str(pixel_x), str(pixel_y)],
                        capture_output=True, text=True, timeout=30,
                        encoding='utf-8', errors='ignore'
                    )
                    
                    if result.returncode == 0:
                        print("[DEBUG] convert_pixel_to_robot_multi → subprocess success")
                        return True
                    else:
                        print(f"[ERROR] pixel2robot_multi.py subprocess failed: {result.stderr}")
                        return False
                        
            finally:
                os.chdir(original_cwd)
                
    except Exception as e:
        if simulation_mode:
            sim_print(f"ERROR: Konvertierung fehlgeschlagen: {e}")
        else:
            print(f"[ERROR] convert_pixel_to_robot_multi → {e}")
        return False

# Detection- und Mapping-Routinen mit Simulation Support
def detect_object():
    """
    Run object detection based on current mode - USES LEGACY detection.py
    This is equivalent to detect_object() in Application.py
    """
    
    if simulation_mode:
        sim_print("DETECTION SIMULATION: Running LEGACY detection.py on test image...")
        try:
            # Use subprocess to call LEGACY detection.py with proper environment
            base = PREDECESSOR_DIR
            script = os.path.join(base, 'detection.py')  # LEGACY VERSION
            
            # Run detection in simulation mode using subprocess
            env = os.environ.copy()
            
            result = subprocess.run([PRE_PYTHON, script], 
                                  cwd=base,
                                  env=env,
                                  capture_output=True,
                                  text=True,
                                  timeout=60,
                                  encoding='utf-8', errors='ignore')
            
            if result.returncode == 0:
                sim_print("SUCCESS SIMULATION: LEGACY object detection completed")
                
                # Try to read detection results
                try:
                    label_file = os.path.join(base, 'txt_file', 'label.txt')
                    if os.path.exists(label_file):
                        with open(label_file, 'r') as f:
                            label_content = f.read().strip()
                        
                        if label_content:
                            sim_print(f"SUCCESS SIMULATION: Object detected with label: {label_content}")
                        else:
                            sim_print("WARNING SIMULATION: No object detected (empty label.txt)")
                    else:
                        sim_print("WARNING SIMULATION: Detection results file not found")
                except Exception as e:
                    sim_print(f"WARNING SIMULATION: Could not read detection results: {e}")
            else:
                sim_print(f"ERROR SIMULATION: Detection script failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            sim_print("ERROR SIMULATION: Detection timeout")
        except Exception as e:
            sim_print(f"ERROR SIMULATION: Detection failed: {e}")
    else:
        print("[DEBUG] detect_object → real mode (LEGACY detection.py)")
        try:
            # Use subprocess to call LEGACY detection.py with proper environment  
            base = PREDECESSOR_DIR
            script = os.path.join(base, 'detection.py')  # LEGACY VERSION
            
            result = subprocess.run([PRE_PYTHON, script], 
                                  cwd=base,
                                  capture_output=True,
                                  text=True,
                                  timeout=120,
                                  encoding='utf-8', errors='ignore')
            
            if result.returncode == 0:
                print("SUCCESS REAL: LEGACY object detection completed")
                
                # Try to read detection results
                try:
                    label_file = os.path.join(base, 'txt_file', 'label.txt')
                    if os.path.exists(label_file):
                        with open(label_file, 'r') as f:
                            label_content = f.read().strip()
                        
                        if label_content:
                            print(f"SUCCESS REAL: Object detected with label: {label_content}")
                        else:
                            print("WARNING REAL: No object detected (empty label.txt)")
                    else:
                        print("WARNING REAL: Detection results file not found")
                except Exception as e:
                    print(f"WARNING REAL: Could not read detection results: {e}")
            else:
                print(f"ERROR REAL: Detection script failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("ERROR REAL: Detection timeout")
        except Exception as e:
            print(f"ERROR REAL: Detection failed: {e}")

def convert_pixel_to_robot():
    """
    Run pixel to robot coordinate conversion - USES LEGACY pixel2robot.py
    This is equivalent to convert_pixel_to_robot() in Application.py
    """
    base   = PREDECESSOR_DIR
    script = os.path.join(base, 'pixel2robot.py')  # LEGACY VERSION
    
    if simulation_mode:
        sim_print(f"INFO SIMULATION: Converting pixel coordinates to robot coordinates using LEGACY {script}")
        
        # In simulation mode, create mock robot coordinates based on center_point.txt
        try:
            center_point_file = os.path.join(base, 'txt_file', 'center_point.txt')
            robot_coords_file = os.path.join(base, 'txt_file', 'robot_coordinates.txt')
            
            if os.path.exists(center_point_file):
                with open(center_point_file, 'r') as f:
                    center_data = f.read().strip()
                
                if center_data:
                    # Parse pixel coordinates
                    if ',' in center_data:
                        pixel_x, pixel_y = center_data.split(',')
                    else:
                        coords = center_data.split()
                        pixel_x, pixel_y = coords[0], coords[1]
                    pixel_x, pixel_y = float(pixel_x), float(pixel_y)

                    # Create realistic mock robot coordinates based on pixel position
                    # Typical conversion: pixel [1199, 865] → robot [0.450, -0.120]
                    mock_x = 0.300 + (pixel_x - 640) * 0.0003  # Scale factor for X
                    mock_y = -0.050 + (pixel_y - 480) * 0.0002  # Scale factor for Y

                    # Write mock coordinates to robot_coordinates.txt
                    with open(robot_coords_file, 'w') as f:
                        f.write(f"{mock_x:.6f}\n")
                        f.write(f"{mock_y:.6f}\n")

                    sim_print(f"MOCK SIMULATION: Created robot coordinates: x={mock_x:.3f}, y={mock_y:.3f}")
                    sim_print(
                        "SIMULATION COORDINATE RESULT:\n"
                        f"[[{mock_x:.5f}]\n"
                        f" [{mock_y:.5f}]\n"
                        " [0.50000]\n"
                        " [1.00000]]"
                    )
                else:
                    sim_print("WARNING SIMULATION: center_point.txt is empty, running LEGACY script")
                    # Run the actual legacy script even in simulation
                    subprocess.run([PRE_PYTHON, script], cwd=base,
                                 stderr=subprocess.DEVNULL)
            else:
                sim_print("WARNING SIMULATION: center_point.txt not found, running LEGACY script")
                # Run the actual legacy script even in simulation
                subprocess.run([PRE_PYTHON, script], cwd=base,
                             stderr=subprocess.DEVNULL)
        except Exception as e:
            sim_print(f"ERROR SIMULATION: Failed to process coordinates: {e}")
            # Fallback: run the actual legacy script
            subprocess.run([PRE_PYTHON, script], cwd=base,
                         stderr=subprocess.DEVNULL)
    else:
        print(f"[DEBUG] convert_pixel_to_robot → running LEGACY {script}")
        subprocess.run([PRE_PYTHON, script], cwd=base,
                       stderr=subprocess.DEVNULL)
    
    if simulation_mode:
        sim_print("SUCCESS SIMULATION: LEGACY pixel to robot coordinate conversion completed")

def pca_calculation():
    """
    Run PCA calculation - USES LEGACY pca.py 
    This is equivalent to pca_calculation() in Application.py
    """
    base   = PREDECESSOR_DIR
    script = os.path.join(base, 'pca.py')  # LEGACY VERSION (not pca_multi.py)
    
    if simulation_mode:
        sim_print(f"INFO SIMULATION: Running LEGACY PCA analysis: {script}")
    else:
        print(f"[DEBUG] pca_calculation → running LEGACY version: {script}")
    
    subprocess.run([PRE_PYTHON, script], cwd=base,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if simulation_mode:
        sim_print("SUCCESS SIMULATION: LEGACY PCA analysis completed - object orientation calculated")

def direction_object():
    """
    Run direction calculation - USES LEGACY direction.py
    This is equivalent to direction_object() in Application.py  
    """
    base   = PREDECESSOR_DIR
    script = os.path.join(base, 'direction.py')  # LEGACY VERSION (not direction_multi.py)
    
    if simulation_mode:
        sim_print(f"DIRECTION SIMULATION: Calculating approach direction with LEGACY: {script}")
    else:
        print(f"[DEBUG] direction_object → running LEGACY version: {script}")
    
    subprocess.run([PRE_PYTHON, script], cwd=base,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if simulation_mode:
        sim_print("SUCCESS SIMULATION: LEGACY direction calculation completed - approach vector determined")

def delet_txt_file():
    """Clear temporary txt files for next cycle"""
    base = PREDECESSOR_DIR
    files = [
        'label.txt', 'center_point.txt', 'robot_coordinates.txt',
        'robot_RPY.txt', 'vectors.txt', 'direction.txt'
    ]
    
    for fn in files:
        path = os.path.join(base, 'txt_file', fn)
        if os.path.exists(path):
            if simulation_mode:
                sim_print(f"CLEAR SIMULATION: Clearing {fn}")
            else:
                print(f"[DEBUG] delet_txt_file → wiping {path}")
            open(path, 'w').close()
    
    if simulation_mode:
        sim_print("SUCCESS SIMULATION: Txt files cleared - system reset for next command")

def filter_and_prepare_selected_object_after_precision_detection(robot_ip):
    """
    Nach precision_detection: Erstellt finale Objektdateien für das ausgewählte Objekt
    ERSTELLT NEUE DATEIEN (überschreibt keine bestehenden):
    - final_object_center_point.txt
    - final_object_label.txt  
    - final_object_crop_path.txt
    - final_object_info.txt
    """
    if simulation_mode:
        sim_print("PRECISION FILTER: Creating final object files for selected object")
    else:
        print("[DEBUG] filter_and_prepare_selected_object_after_precision_detection")
    
    try:
        base = PREDECESSOR_DIR
        
        # 1. Lese selection_data.json um zu wissen, welches Objekt wir wollen
        selection_file = os.path.join(base, 'txt_file', 'selection_data.json')
        with open(selection_file, 'r') as f:
            selection_data = json.load(f)
            target_object_id = selection_data['selected_object_id']
            target_object_class = selection_data['selected_object_class']
        
        if simulation_mode:
            sim_print(f"TARGET OBJECT: {target_object_class} (ID: {target_object_id})")
        
        # 2. Lese die NEUEN Detection-Ergebnisse aus detected_objects.json 
        detection_file = os.path.join(base, 'txt_file', 'detected_objects.json')
        with open(detection_file, 'r') as f:
            detection_data = json.load(f)
            detected_objects = detection_data.get('objects', [])
        
        # 3. Finde das gewünschte Objekt in den neuen Ergebnissen - VERBESSERTE OBJEKT-SUCHE
        target_object = None
        for obj in detected_objects:
            # VERBESSERTE OBJEKT-SUCHE: Mehrere Kriterien
            if obj['class_name'] == target_object_class:
                # Kriterium 1: Positionsabweichung prüfen
                center_diff_x = abs(obj['center'][0] - selection_data.get('original_center_x', obj['center'][0]))
                center_diff_y = abs(obj['center'][1] - selection_data.get('original_center_y', obj['center'][1]))
                position_match = center_diff_x < 150 and center_diff_y < 150  # Erweiterte Toleranz
                
                # Kriterium 2: Confidence-Ähnlichkeit prüfen
                confidence_diff = abs(obj['confidence'] - selection_data.get('original_confidence', obj['confidence']))
                confidence_match = confidence_diff < 0.3  # Confidence-Toleranz
                
                # Kriterium 3: Bounding-Box-Ähnlichkeit prüfen
                original_bbox = selection_data.get('original_bbox', obj['bbox'])
                bbox_similarity = (
                    abs(obj['bbox'][2] - obj['bbox'][0] - (original_bbox[2] - original_bbox[0])) < 50 and  # Breite
                    abs(obj['bbox'][3] - obj['bbox'][1] - (original_bbox[3] - original_bbox[1])) < 50     # Höhe
                )
                
                # Objekt auswählen wenn mindestens 2 von 3 Kriterien erfüllt sind
                matches = sum([position_match, confidence_match, bbox_similarity])
                if matches >= 2:
                    target_object = obj
                    if simulation_mode:
                        sim_print(f"OBJECT MATCH: Found with {matches}/3 criteria (pos={position_match}, conf={confidence_match}, bbox={bbox_similarity})")
                    break
        
        if target_object is None:
            # FALLBACK: Finde Objekt mit geringster Entfernung zur ursprünglichen Position
            print("WARNING: No exact object match found, using closest object of target class by position")
            best_distance = float('inf')
            best_object = None
            for obj in detected_objects:
                if obj['class_name'] == target_object_class:
                    # Berechne Entfernung zur ursprünglichen Position
                    orig_x = selection_data.get('original_center_x', obj['center'][0])
                    orig_y = selection_data.get('original_center_y', obj['center'][1])
                    distance = ((obj['center'][0] - orig_x)**2 + (obj['center'][1] - orig_y)**2)**0.5
                    if distance < best_distance:
                        best_distance = distance
                        best_object = obj
            
            if best_object is not None:
                target_object = best_object
                if simulation_mode:
                    sim_print(f"FALLBACK: Using closest {target_object_class} object (distance: {best_distance:.1f} pixels)")
                else:
                    print(f"[DEBUG] FALLBACK: Using closest {target_object_class} object (distance: {best_distance:.1f} pixels)")
            
            if target_object is None:
                # EXTENDED FALLBACK: Try with highest confidence of same class
                same_class_objects = [obj for obj in detected_objects if obj['class_name'] == target_object_class]
                if same_class_objects:
                    target_object = max(same_class_objects, key=lambda x: x['confidence'])
                    if simulation_mode:
                        sim_print(f"EXTENDED FALLBACK: Using highest confidence {target_object_class}")
        
        if target_object is None:
            error_msg = f"ERROR: No {target_object_class} objects found in precision detection"
            if simulation_mode:
                sim_print(error_msg)
            else:
                print(error_msg)
            return False
        
        # 4. ERSTELLE NEUE FINALE DATEIEN (keine Überschreibung!)
        txt_dir = os.path.join(base, 'txt_file')
        
        # FINAL CENTER POINT - für PCA/Direction/Border Verarbeitung
        final_center_file = os.path.join(txt_dir, 'final_object_center_point.txt')
        center_x, center_y = target_object['center']
        with open(final_center_file, 'w') as f:
            f.write(f"{int(center_x)} {int(center_y)}")  # Space-separated, kein newline
        
        # FINAL LABEL - für Objektklassifizierung und PCA/Direction
        final_label_file = os.path.join(txt_dir, 'final_object_label.txt')
        bbox = target_object['bbox']
        confidence = target_object['confidence'] 
        class_id = target_object['class']
        with open(final_label_file, 'w') as f:
            f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {confidence}\n")
        
        # 5. FINAL CROP PATH - für PCA-Berechnung
        yolo_runs_dir = os.path.join(base, "yolov5", "runs", "detect")
        if os.path.exists(yolo_runs_dir):
            multi_exp_dirs = [d for d in os.listdir(yolo_runs_dir) if d.startswith('multi_exp')]
            if multi_exp_dirs:
                latest_exp = sorted(multi_exp_dirs)[-1]
                latest_exp_path = os.path.join(yolo_runs_dir, latest_exp)
                
                # Suche nach dem Crop-Bild für unsere Objektklasse
                crop_class_dir = os.path.join(latest_exp_path, "crops", target_object_class)
                if os.path.exists(crop_class_dir):
                    crop_files = [f for f in os.listdir(crop_class_dir) if f.endswith('.jpg')]
                    if crop_files:
                        # Nehme das erste verfügbare Crop
                        crop_path = os.path.join(crop_class_dir, crop_files[0])
                        
                        final_crop_file = os.path.join(txt_dir, 'final_object_crop_path.txt')
                        with open(final_crop_file, 'w') as f:
                            f.write(crop_path)
                        
                        if simulation_mode:
                            sim_print(f"FINAL CROP PATH: {crop_path}")
        
        # 6. FINAL OBJECT INFO - für Debugging und Verifikation
        final_info_file = os.path.join(txt_dir, 'final_object_info.txt')
        with open(final_info_file, 'w') as f:
            f.write(f"Selected Object Information (After Precision Detection)\n")
            f.write(f"===============================================\n")
            f.write(f"Object Class: {target_object_class}\n")
            f.write(f"Object ID: {target_object['id']}\n")
            f.write(f"Confidence: {confidence:.3f}\n")
            f.write(f"Center: ({int(center_x)}, {int(center_y)})\n")
            f.write(f"Bbox: {bbox}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if simulation_mode:
            sim_print(f"SUCCESS: Created final object files for {target_object_class}")
            sim_print(f"  Files: final_object_center_point.txt, final_object_label.txt, final_object_crop_path.txt")
            sim_print(f"  Center: ({int(center_x)}, {int(center_y)})")
            sim_print(f"  Confidence: {confidence:.3f}")
        else:
            print(f"[DEBUG] Created final object files for {target_object_class}")
        
        return True
        
    except Exception as e:
        error_msg = f"Failed to create final object files: {e}"
        if simulation_mode:
            sim_print(f"ERROR: {error_msg}")
        else:
            print(f"[ERROR] {error_msg}")
        return False

def precision_detection(robot_ip):
    """Zweite Detection nach Heranfahren für höhere Präzision - ÄQUIVALENT zu Vorgänger-Step-11"""
    if simulation_mode:
        sim_print("PRECISION DETECTION: Running second detection for higher precision")
    else:
        print("[DEBUG] precision_detection → running second object detection")
    
    # WICHTIG: Setze explizit real mode für neues Kamerabild nach Heranfahren
    original_env_mode = os.environ.get('DETECTION_MODE', None)
    if not simulation_mode:
        os.environ['DETECTION_MODE'] = 'real'
        print("[DEBUG] precision_detection → forced DETECTION_MODE=real for new camera capture")
    
    try:
        # Führe detection_multi_precision_run.py aus
        detection_script = os.path.join(PREDECESSOR_DIR, 'detection_multi_precision_run.py')
        result = subprocess.run(
            [PRE_PYTHON, detection_script],
            cwd=PREDECESSOR_DIR,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                "Precision detection failed with return code "
                f"{result.returncode}. {details}"
            )
    finally:
        # Stelle ursprüngliche Environment Variable wieder her
        if original_env_mode is None:
            if 'DETECTION_MODE' in os.environ:
                del os.environ['DETECTION_MODE']
        else:
            os.environ['DETECTION_MODE'] = original_env_mode
    
    if simulation_mode:
        sim_print("SUCCESS: Precision detection completed")

def precision_pca_calculation(robot_ip):
    """PCA-Berechnung nach Precision Detection - ÄQUIVALENT zu Vorgänger-Step-13"""
    if simulation_mode:
        sim_print("PRECISION PCA: Running PCA analysis for object orientation")
    else:
        print("[DEBUG] precision_pca_calculation → running PCA analysis")
    
    _run_precision_stage(
        "pca_multi.py",
        os.path.join(PREDECESSOR_DIR, "txt_file", "vectors.txt"),
        "PCA",
    )
    
    if simulation_mode:
        sim_print("SUCCESS: Precision PCA analysis completed")

def precision_direction_object(robot_ip):
    """Richtungsberechnung nach Precision PCA - ÄQUIVALENT zu Vorgänger-Step-15"""
    if simulation_mode:
        sim_print("PRECISION DIRECTION: Calculating object orientation")
    else:
        print("[DEBUG] precision_direction_object → calculating direction")
    
    _run_precision_stage(
        "direction_multi.py",
        os.path.join(PREDECESSOR_DIR, "txt_file", "robot_RPY.txt"),
        "direction",
    )
    
    if simulation_mode:
        sim_print("SUCCESS: Precision direction calculation completed")


def _run_precision_stage(script_name: str, output_path: str, stage_name: str) -> None:
    if os.path.exists(output_path):
        os.remove(output_path)
    script_path = os.path.join(PREDECESSOR_DIR, script_name)
    result = subprocess.run(
        [PRE_PYTHON, script_path],
        cwd=PREDECESSOR_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"Precision {stage_name} failed with return code "
            f"{result.returncode}. {details[-4000:]}"
        )
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(
            f"Precision {stage_name} did not create required output {output_path}"
        )

# Standard-Roboter-IP
robot_ip = '192.168.2.180'

def determine_current_object_class():
    """
    Einfache Objektklassen-Bestimmung aus final_object_label.txt
    """
    base = PREDECESSOR_DIR
    
    # SCHRITT 1: final_object_label.txt lesen (bevorzugt)
    final_label_file = os.path.join(base, 'txt_file', 'final_object_label.txt')
    if os.path.exists(final_label_file):
        try:
            with open(final_label_file, 'r') as f:
                content = f.read().strip()
                if content:
                    object_class = int(content.split()[0])  # Erste Zahl = class
                    print(f"[DEBUG] determine_current_object_class → aus final_object_label.txt: {object_class}")
                    return object_class
        except (ValueError, IndexError):
            pass
    
    # FALLBACK 1: label.txt (Legacy Workflow)
    label_file = os.path.join(base, 'txt_file', 'label.txt')
    if os.path.exists(label_file):
        try:
            with open(label_file, 'r') as f:
                content = f.read().strip()
                if content:
                    object_class = int(content.split()[0])
                    print(f"[DEBUG] determine_current_object_class → aus label.txt: {object_class}")
                    return object_class
        except (ValueError, IndexError):
            pass
    
    # FALLBACK 2: Default
    print(f"[DEBUG] determine_current_object_class → fallback zu Class 0 (Cylinder)")
    return 0

def execute_robot_method(method_name, robot_ip):
    from src.zone_coordinates import get_zone_coordinates, get_object_place_height
    global x_place, y_place, z_place

    emit_method_execution(method_name, simulation_output_callback)
    if method_name == "suction_on":
        print("DEBUG METHOD: *** SUCTION_ON DETECTED - this is the suction step! ***")

    if simulation_mode:
        sim_print(f"EXECUTE SIMULATION: Executing method: {method_name}")
    else:
        print(f"[DEBUG] execute_robot_method → {method_name}")

    # NEW: Handle LEGACY WORKFLOW methods (replaces individual method calls)
    if method_name.startswith("execute_legacy_workflow_"):
        target_zone = method_name.replace("execute_legacy_workflow_", "")
        if simulation_mode:
            sim_print(f"LEGACY WORKFLOW: Starting complete Application.py workflow for {target_zone}")
        else:
            print(f"[DEBUG] execute_legacy_workflow → target_zone={target_zone}")
        
        # Execute the complete legacy workflow (equivalent to entire Application.py main loop)
        return execute_legacy_robot_workflow(robot_ip, target_zone)
    
    # NEW: PRECISION METHODS WITH OBJECT FILTERING
    elif method_name == "precision_detection":
        # Zweite Detection (NUR Detection, OHNE automatisches Processing)
        return precision_detection(robot_ip)
        
    elif method_name == "filter_and_prepare_selected_object_after_precision_detection":
        # Objektfilterung nach Precision Detection
        return filter_and_prepare_selected_object_after_precision_detection(robot_ip)
        
    elif method_name == "precision_pca_calculation":
        # PCA für das bereits gefilterte Objekt
        return precision_pca_calculation(robot_ip)
        
    elif method_name == "precision_direction_object":
        # Direction für das bereits gefilterte Objekt  
        return precision_direction_object(robot_ip)
    
    # LEGACY: Individual method support (kept for compatibility, but not used in new system)
    elif method_name == "move_to_main_position":
        move_to_main_position(robot_ip)
    elif method_name == "detect_object":
        detect_object()
    elif method_name == "convert_pixel_to_robot":
        convert_pixel_to_robot()
    elif method_name == "move_to_object":
        move_to_object(robot_ip)
    elif method_name == "pick_the_object":
        pick_the_object(robot_ip)
    elif method_name == "suction_on":
        suction_on(robot_ip)
    elif method_name == "pick_up_object":
        pick_up_object(robot_ip)
    elif method_name == "intermediate_position":
        intermediate_position(robot_ip)
    elif method_name.startswith("move_to_target"):
        # Support both formats: move_to_target(Zone_1) and move_to_target_Zone_1
        if "(" in method_name and ")" in method_name:
            # Old format: move_to_target(Zone_1)
            target = method_name.split("(", 1)[1].rstrip(")")
        else:
            # New format: move_to_target_Zone_1
            target = method_name.replace("move_to_target_", "")
        
        zone_coords = get_zone_coordinates(target)
        
        if zone_coords is None:
            if simulation_mode:
                sim_print(f"ERROR SIMULATION: Zone '{target}' not found in zone coordinates")
                return
            else:
                print(f"[ERROR] Zone '{target}' not found in zone coordinates")
                return
        
        # NEUE LOGIK: Objektspezifische Z-Höhe bestimmen
        object_class = determine_current_object_class()
        z_place_height = get_object_place_height(object_class)
        
        x_place, y_place, z_place = (zone_coords["x"],
                                     zone_coords["y"],
                                     z_place_height)
        if simulation_mode:
            sim_print(f"TARGET SIMULATION: Zone coordinates set for {target}: ({x_place}, {y_place}, {z_place})")
            sim_print(f"TARGET SIMULATION: Object class {object_class} → Place height: {z_place_height}")
        else:
            print(f"[DEBUG] move_to_target → x_place={x_place}, y_place={y_place}, z_place={z_place}")
            print(f"[DEBUG] move_to_target → object_class={object_class}, place_height={z_place_height}")
        
        # AUSKOMMENTIERT: Bewegung entfernt, da final_position() diese übernimmt (verhindert doppelte Bewegung/Bremsen)
        # script = generate_urscript_movel(x_place, y_place, z_place,
        #                                 2.221, 2.221, 0.0)
        # send_urscript(script, robot_ip)
        
        if simulation_mode:
            sim_print(f"SUCCESS SIMULATION: Zone coordinates prepared for {target}")
            # time.sleep(0.5)  # Auskommentiert da keine Bewegung
    elif method_name.startswith("move_to_point(") and method_name.endswith(")"):
        coordinates = method_name[len("move_to_point("):-1].split(",")
        if len(coordinates) != 2:
            raise ValueError(f"Invalid point target method {method_name}")
        object_class = determine_current_object_class()
        x_place = float(coordinates[0])
        y_place = float(coordinates[1])
        z_place = get_object_place_height(object_class)
        if simulation_mode:
            sim_print(
                "TARGET SIMULATION: Calibrated point coordinates set: "
                f"({x_place}, {y_place}, {z_place})"
            )
        else:
            print(
                "[DEBUG] move_to_point → "
                f"x_place={x_place}, y_place={y_place}, z_place={z_place}"
            )
    elif method_name == "final_position":
        final_position(robot_ip)
    elif method_name in ("suction_off", "release_object"):
        suction_off(robot_ip)
    elif method_name == "pca_calculation":
        pca_calculation()
    elif method_name == "pca":  # Alias für Kompatibilität
        pca_calculation()
    elif method_name == "direction_object":
        direction_object()
    elif method_name == "direction":  # Alias für Kompatibilität
        direction_object()
    elif method_name == "move_to_main":  # Alias für Kompatibilität
        move_to_main_position(robot_ip)
    elif method_name == "intermediate":  # Alias für Kompatibilität
        intermediate_position(robot_ip)
    elif method_name == "final":  # Alias für Kompatibilität
        final_position(robot_ip)
    elif method_name == "delet_txt_file":
        delet_txt_file()
    elif method_name == "delete_txt":  # Alias für Kompatibilität
        delet_txt_file()
    elif method_name == "move_to_selected_object":
        return move_to_selected_object(robot_ip)
    elif method_name == "convert_pixel_to_robot_multi":
        # Extract coordinates from method call if provided, otherwise use default
        if "(" in method_name and ")" in method_name:
            # Format: convert_pixel_to_robot_multi(1530,696)
            coords_str = method_name.split("(", 1)[1].rstrip(")")
            pixel_x, pixel_y = map(int, coords_str.split(","))
            return convert_pixel_to_robot_multi(pixel_x, pixel_y)
        else:
            # Use default center_point.txt if no coordinates provided
            center_point_file = os.path.join(PREDECESSOR_DIR, 'txt_file', 'center_point.txt')
            if os.path.exists(center_point_file):
                with open(center_point_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        if ',' in content:
                            pixel_x, pixel_y = content.split(',')
                        else:
                            coords = content.split()
                            pixel_x, pixel_y = coords[0], coords[1]
                        pixel_x = int(float(pixel_x.strip()))
                        pixel_y = int(float(pixel_y.strip()))
                        return convert_pixel_to_robot_multi(pixel_x, pixel_y)
            return False
    elif method_name == "alert_operator":
        if simulation_mode:
            sim_print("WARNING SIMULATION: Alert: Unsafe command detected. Operator would be notified!")
        else:
            print("Alert: Unsicherer Befehl erkannt. Operator wird benachrichtigt!")
    else:
        if simulation_mode:
            sim_print(f"ERROR SIMULATION: Undefined method: {method_name}")
        else:
            print(f"Undefined method: {method_name}")

def execute_robot_workflow_simulation(
    robot_ip,
    method_list,
    output_callback=None,
    return_home=True,
):
    """
    Execute robot workflow in simulation mode with detailed output
    UPDATED: Now supports LEGACY WORKFLOW system
    """
    global simulation_mode, simulation_output_callback
    simulation_mode = True
    simulation_output_callback = output_callback
    
    sim_print("SIMULATION: Starting robot workflow simulation")
    sim_print(f"ROBOT SIMULATION: Target robot IP: {robot_ip}")
    sim_print(f"SIMULATION: Methods to execute: {len(method_list)}")
    sim_print("="*60)
    
    # Check if we're using the new legacy workflow system
    if len(method_list) == 1 and method_list[0].startswith("execute_legacy_workflow_"):
        sim_print("WORKFLOW: Using NEW LEGACY WORKFLOW SYSTEM (Application.py equivalent)")
        target_zone = method_list[0].replace("execute_legacy_workflow_", "")
        sim_print(f"WORKFLOW: Target zone: {target_zone}")
        
        # Prepare object data for robot before workflow execution
        try:
            # Load detection data directly
            import json
            detection_json = os.path.join(PREDECESSOR_DIR, "txt_file", "detected_objects.json")
            
            detection_data = None
            if os.path.exists(detection_json):
                with open(detection_json, 'r') as f:
                    detection_data = json.load(f)
            
            if detection_data and detection_data.get('objects'):
                # Smart object selection: Use highest confidence object 
                sorted_objects = sorted(detection_data['objects'], key=lambda x: x['confidence'], reverse=True)
                selected_object = sorted_objects[0]
                
                # Strategy 2: Try to find Cylinder objects first (common target)
                for obj in sorted_objects:
                    if obj['class_name'].lower() in ['cylinder', 'zylinder']:
                        selected_object = obj
                        break
                
                sim_print(f"OBJECT SIMULATION: Preparing object data for {selected_object['class_name']}")
                
                # Prepare object data directly (inline implementation)
                txt_dir = os.path.join(PREDECESSOR_DIR, "txt_file")
                
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
                
                sim_print("SUCCESS SIMULATION: Object data prepared for LEGACY workflow")
                sim_print(f"SUCCESS SIMULATION: Selected {selected_object['class_name']} (confidence: {selected_object['confidence']:.2f})")
            else:
                sim_print("WARNING SIMULATION: No detection data available, proceeding with limited simulation")
        except Exception as e:
            sim_print(f"WARNING SIMULATION: Object preparation error: {e}, continuing simulation")
        
        # Execute the legacy workflow
        try:
            result = execute_robot_method(method_list[0], robot_ip)
            if result:
                sim_print("SUCCESS SIMULATION: LEGACY workflow completed successfully!")
            else:
                sim_print("ERROR SIMULATION: LEGACY workflow failed!")
        except Exception as e:
            sim_print(f"ERROR SIMULATION: LEGACY workflow error: {e}")
    else:
        # OLD SYSTEM: Individual method execution (kept for compatibility)
        sim_print("WORKFLOW: Using OLD individual method system")
        
        # Prepare object data for robot before workflow execution (old method)
        try:
            # Load detection data directly
            import json
            detection_json = os.path.join(PREDECESSOR_DIR, "txt_file", "detected_objects.json")
            
            detection_data = None
            if os.path.exists(detection_json):
                with open(detection_json, 'r') as f:
                    detection_data = json.load(f)
            
            if detection_data and detection_data.get('objects'):
                # Smart object selection: Use highest confidence object 
                sorted_objects = sorted(detection_data['objects'], key=lambda x: x['confidence'], reverse=True)
                selected_object = sorted_objects[0]
                
                # Strategy 2: Try to find Cylinder objects first (common target)
                for obj in sorted_objects:
                    if obj['class_name'].lower() in ['cylinder', 'zylinder']:
                        selected_object = obj
                        break
                
                sim_print(f"OBJECT SIMULATION: Preparing object data for {selected_object['class_name']}")
                
                # Prepare object data directly (inline implementation)
                txt_dir = os.path.join(PREDECESSOR_DIR, "txt_file")
                
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
                
                sim_print("SUCCESS SIMULATION: Object data prepared for robot workflow")
                sim_print(f"SUCCESS SIMULATION: Selected {selected_object['class_name']} (confidence: {selected_object['confidence']:.2f})")
            else:
                sim_print("WARNING SIMULATION: No detection data available, proceeding with limited simulation")
        except Exception as e:
            sim_print(f"WARNING SIMULATION: Object preparation error: {e}, continuing simulation")
        
        for i, method in enumerate(method_list, 1):
            sim_print(f"\nSTEP SIMULATION: Step {i}/{len(method_list)}: {method}")
            
            try:
                execute_robot_method(method, robot_ip)
                sim_print(f"SUCCESS SIMULATION: Step {i} completed successfully")
            except Exception as e:
                sim_print(f"ERROR SIMULATION: Step {i} failed: {e}")
                continue
            
            # Small delay between steps for better visualization
            time.sleep(0.3)
    
    sim_print("\n" + "="*60)
    sim_print("COMPLETE SIMULATION: Workflow simulation completed!")
    if return_home:
        sim_print("READY SIMULATION: Robot returned to main position and is ready")
    else:
        sim_print("READY SIMULATION: Robot remains at the intermediate position")
    
    # Reset simulation mode
    simulation_mode = False
    simulation_output_callback = None

def execute_robot_workflow(robot_ip, method_list, return_home=True):
    """
    Execute real robot workflow with return to main position
    UPDATED: Now supports LEGACY WORKFLOW system
    """
    # DEBUG: WORKFLOW ANALYSIS
    print(f"DEBUG WORKFLOW: Starting workflow with {len(method_list)} methods")
    print(f"DEBUG WORKFLOW: Methods: {method_list}")
    print(f"DEBUG WORKFLOW: 'suction_on' in method_list = {'suction_on' in method_list}")
    if 'suction_on' in method_list:
        print(f"DEBUG WORKFLOW: suction_on position = {method_list.index('suction_on') + 1}/{len(method_list)}")
    
    print(f"[DEBUG] execute_robot_workflow → methods={method_list}")
    
    # Check if we're using the new legacy workflow system
    if len(method_list) == 1 and method_list[0].startswith("execute_legacy_workflow_"):
        print("[DEBUG] execute_robot_workflow → using LEGACY WORKFLOW system")
        target_zone = method_list[0].replace("execute_legacy_workflow_", "")
        print(f"[DEBUG] execute_robot_workflow → target_zone={target_zone}")
        
        # Execute the complete legacy workflow (no need for individual steps)
        result = execute_robot_method(method_list[0], robot_ip)
        if result:
            print("SUCCESS REAL: LEGACY workflow completed - robot ready for next command")
        else:
            print("ERROR REAL: LEGACY workflow failed")
    else:
        # OLD SYSTEM: Individual method execution (kept for compatibility)
        print("[DEBUG] execute_robot_workflow → using individual method system")
        for i, m in enumerate(method_list, 1):
            print(f"DEBUG WORKFLOW: Step {i}/{len(method_list)}: {m}")
            try:
                execute_robot_method(m, robot_ip)
                print(f"DEBUG WORKFLOW: Step {i} completed successfully")
            except Exception as e:
                print(f"DEBUG WORKFLOW: Step {i} FAILED: {e}")
                if m == "suction_on":
                    print("DEBUG WORKFLOW: *** SUCTION_ON STEP FAILED! ***")
                continue
            
            # INTELLIGENTE WARTEZEIT:
            if m in ["suction_on", "suction_off"]:
                print(f"DEBUG WORKFLOW: Extended wait after {m} (5 seconds)")
                time.sleep(5)  # Längere Wartezeit für Saugfunktionen
            elif i < len(method_list) and method_list[i] in ["suction_on", "suction_off"]:
                print(f"DEBUG WORKFLOW: Extended wait before {method_list[i]} (5 seconds)")  
                time.sleep(5)  # Längere Wartezeit vor Saugfunktionen
            else:
                time.sleep(1)   # Standard-Wartezeit für andere Schritte
        
        if return_home:
            print("[DEBUG] execute_robot_workflow → returning to main position")
            move_to_main_position(robot_ip)
            print("SUCCESS REAL: Workflow completed and robot is ready")
        else:
            print("SUCCESS REAL: Pick phase completed at intermediate position")

def execute_legacy_robot_workflow(robot_ip, target_zone="Zone_1"):
    """
    Execute the EXACT robot workflow from Application.py
    This replicates the while loop from Application.py step by step
    
    Args:
        robot_ip: IP address of the robot
        target_zone: Target zone (Zone_1, Zone_2, Zone_3)
    """
    from src.zone_coordinates import get_zone_coordinates, get_object_place_height
    global x_place, y_place, z_place
    
    if simulation_mode:
        sim_print("LEGACY WORKFLOW: Starting EXACT Application.py workflow...")
        sim_print(f"LEGACY: Target robot IP: {robot_ip}")
        sim_print(f"LEGACY: Target zone: {target_zone}")
    else:
        print(f"[DEBUG] execute_legacy_robot_workflow → robot_ip={robot_ip}, target_zone={target_zone}")
    
    try:
        # Step 1: Move to main position (same as Application.py)
        if simulation_mode:
            sim_print("LEGACY STEP 1: move_to_main_position")
        move_to_main_position(robot_ip)
        
        # Step 2: Detect object (same as Application.py)
        if simulation_mode:
            sim_print("LEGACY STEP 2: detect_object")
        detect_object()
        
        # Step 3: Check if object found (same as Application.py)
        if simulation_mode:
            sim_print("LEGACY STEP 3: Checking if object found in label.txt")
        file_path = os.path.join(PREDECESSOR_DIR, 'txt_file', 'label.txt')
        with open(file_path, 'r') as file:
            contents = file.read().strip()
            if len(contents) == 0:
                if simulation_mode:
                    sim_print("ERROR LEGACY: Object has not been found - workflow aborted")
                else:
                    print("Object has not been found")
                return False
        
        # Step 4: Convert pixel to robot coordinates (same as Application.py)
        if simulation_mode:
            sim_print("LEGACY STEP 4: convert_pixel_to_robot")
        convert_pixel_to_robot()
        
        # Step 5: Move to object (same as Application.py)
        if simulation_mode:
            sim_print("LEGACY STEP 5: move_to_object")
        move_to_object(robot_ip)
        
        # Step 6: Detect object again (same as Application.py)
        if simulation_mode:
            sim_print("LEGACY STEP 6: detect_object (second time)")
        detect_object()
        
        # Step 7: Read object type and set coordinates (same as Application.py)
        if simulation_mode:
            sim_print("LEGACY STEP 7: Reading object type and setting pick/place coordinates")
        with open(file_path, 'r') as file:
            content = file.read().strip()
        object_type = int(content.split()[0])
        
        # Set coordinates based on object type (ANGEPASST: objektspezifische Z-Höhe)
        z_pick = {0: 0.05, 1: 0.030, 2: 0.020}.get(object_type)
        
        # NEUE LOGIK: Zone-Koordinaten (X,Y) + objektspezifische Z-Höhe
        zone_coords = get_zone_coordinates(target_zone)
        if zone_coords:
            x_place, y_place = zone_coords["x"], zone_coords["y"]  # X,Y aus Zone
            z_place = get_object_place_height(object_type)        # Z objektspezifisch
        else:
            # Fallback (ANGEPASST: nur Z objektspezifisch)
            if object_type == 0:
                x_place, y_place = 0.366, -0.146
            elif object_type == 1:
                x_place, y_place = 0.366, -0.008
            elif object_type == 2:
                x_place, y_place = 0.366, 0.121
            else:
                raise ValueError("Invalid object_type. It must be 0, 1, or 2.")
            z_place = get_object_place_height(object_type)  # Z objektspezifisch
        
        if simulation_mode:
            sim_print(f"LEGACY: Object type: {object_type}, z_pick: {z_pick}, z_place: {z_place}")
            sim_print(f"LEGACY: Target coordinates: ({x_place}, {y_place}, {z_place})")
        else:
            print(f"[DEBUG] Legacy workflow → object_type={object_type}, z_pick={z_pick}, z_place={z_place}")
        
        # Step 8: PCA calculation (same as Application.py)
        if simulation_mode:
            sim_print("LEGACY STEP 8: pca_calculation")
        pca_calculation()
        
        # Step 9: Direction calculation (same as Application.py)
        if simulation_mode:
            sim_print("LEGACY STEP 9: direction_object")
        direction_object()
        
        # Step 10: Pick the object (same as Application.py)
        if simulation_mode:
            sim_print("LEGACY STEP 10: pick_the_object")
        pick_the_object(robot_ip)
        
        # Step 11: Turn on suction (same as Application.py)
        if simulation_mode:
            sim_print("LEGACY STEP 11: suction_on")
        suction_on(robot_ip)
        
        # Step 12: Wait (same as Application.py)
        if simulation_mode:
            sim_print("LEGACY STEP 12: time.sleep(3)")
        time.sleep(3)
        
        # Step 13: Pick up object (same as Application.py)
        if simulation_mode:
            sim_print("LEGACY STEP 13: pick_up_object")
        pick_up_object(robot_ip)
        
        # Step 14: Move to intermediate position (same as Application.py)
        if simulation_mode:
            sim_print("LEGACY STEP 14: intermediate_position")
        intermediate_position(robot_ip)
        
        # Step 15: Move to final position (same as Application.py)
        if simulation_mode:
            sim_print("LEGACY STEP 15: final_position")
        final_position(robot_ip)
        
        # Step 16: Turn off suction (same as Application.py)
        if simulation_mode:
            sim_print("LEGACY STEP 16: suction_off")
        suction_off(robot_ip)
        
        # Step 17: Wait (same as Application.py)
        if simulation_mode:
            sim_print("LEGACY STEP 17: time.sleep(3)")
        time.sleep(3)
        
        # Step 18: Move to intermediate position (same as Application.py)
        if simulation_mode:
            sim_print("LEGACY STEP 18: intermediate_position")
        intermediate_position(robot_ip)
        
        # Step 19: Move to main position (same as Application.py)
        if simulation_mode:
            sim_print("LEGACY STEP 19: move_to_main_position")
        move_to_main_position(robot_ip)
        
        # Step 20: Delete txt files (same as Application.py)
        if simulation_mode:
            sim_print("LEGACY STEP 20: delet_txt_file")
        delet_txt_file()
        
        if simulation_mode:
            sim_print("SUCCESS LEGACY: EXACT Application.py workflow completed successfully!")
        else:
            print("SUCCESS: Legacy robot workflow completed successfully!")
        
        return True
        
    except Exception as e:
        if simulation_mode:
            sim_print(f"ERROR LEGACY: Workflow failed: {e}")
        else:
            print(f"ERROR: Legacy robot workflow failed: {e}")
        return False

if __name__ == "__main__":
    # kurzer CLI-Test
    print("=== Testlauf: move_to_main_position + detect_object ===")
    move_to_main_position(robot_ip)
    detect_object()
