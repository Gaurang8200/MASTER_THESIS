import socket
import time
import os
import sys

def send_tcp_script(script_path, ip_address="192.168.1.1", port=30002):
    """
    Send URScript commands to the robot via TCP socket
    """
    try:
        with open(script_path, 'r') as file:
            script_content = file.read()
        
        # Clean script content
        cleaned_script = script_content.strip()
        if not cleaned_script.endswith('\n'):
            cleaned_script += '\n'
        
        print(f"ROBOT: Connecting to robot at {ip_address}:{port}")
        
        # Create TCP socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(10)  # 10 second timeout
            
            # Connect to robot
            s.connect((ip_address, port))
            print("SUCCESS: Connected to robot")
            
            # Send script
            s.send(cleaned_script.encode())
            print("FILE: Script sent successfully")
            
            # Wait for acknowledgment (optional)
            time.sleep(0.5)
            
        return True
        
    except FileNotFoundError:
        print(f"ERROR: Script file not found: {script_path}")
        return False
    except ConnectionError as e:
        print(f"ERROR: Connection failed: {e}")
        return False
    except socket.timeout:
        print("ERROR: Connection timeout")
        return False
    except Exception as e:
        print(f"ERROR: Failed to send script: {e}")
        return False

def read_pose(ip_address="192.168.1.1", port=30002):
    """
    Read current robot pose from the robot
    """
    try:
        print(f"ROBOT: Connecting to robot for pose reading...")
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5.0)
            s.connect((ip_address, port))
            
            # Send pose query command
            get_pose_script = "get_actual_tcp_pose()\\n"
            s.send(get_pose_script.encode())
            
            # Wait for response
            time.sleep(0.1)
            
            print("SUCCESS: Pose request sent")
            return True
            
    except Exception as e:
        print(f"ERROR: Failed to read pose: {e}")
        return False

def test_script_sending():
    """
    Test function to verify script sending functionality
    """
    print("TEST: Testing script sending functionality...")
    
    # Test script content
    test_script = """
    # Test URScript
    def test_program():
        popup("Hello from Python", "Test Message", False, False, blocking=True)
        textmsg("Script executed successfully")
    end
    
    test_program()
    """
    
    # Save test script
    test_path = "test_script.script"
    try:
        with open(test_path, 'w') as f:
            f.write(test_script)
        print(f"FILE: Test script saved: {test_path}")
        
        # Try to send (will fail if no robot connected)
        result = send_tcp_script(test_path)
        if result:
            print("SUCCESS: Test script sent successfully")
        else:
            print("WARNING: Test script could not be sent (robot not connected)")
        
        # Clean up
        os.remove(test_path)
        print("FILE: Test script cleaned up")
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")

def main():
    """
    Main function for testing
    """
    print("SCRIPT: URScript TCP Sender")
    print("=" * 40)
    
    # Check if we have a script file argument
    if len(sys.argv) > 1:
        script_file = sys.argv[1]
        if os.path.exists(script_file):
            print(f"FILE: Sending script: {script_file}")
            result = send_tcp_script(script_file)
            if result:
                print("SUCCESS: Script sent successfully")
            else:
                print("ERROR: Failed to send script")
        else:
            print(f"ERROR: Script file not found: {script_file}")
    else:
        # Run test
        test_script_sending()

# URScript file content
urscript = """
def move_to_position():
    movel(p[-0.37221, -0.01232, 0.559, 2.944, -1.163, 0.023], a=0.1, v=0.10)
    textmsg("Movement complete!")
end

move_to_position()
"""

# Robot IP address
robot_ip = '192.168.0.118'

# Send URScript to the robot
send_urscript(urscript, robot_ip)

# Call main function
main()
