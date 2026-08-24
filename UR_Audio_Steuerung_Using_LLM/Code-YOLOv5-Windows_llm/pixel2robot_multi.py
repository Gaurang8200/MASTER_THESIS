import numpy as np
import json
import function_pool as fp
import sys
import os


def pixel2robot(cordx, cordy):
    """
    Core conversion function - unchanged from original
    """
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    input_name = "output_wp2camera.json"
    input_name2 = "output_c2f.json"
    input_name3 = "robot_poses.json"
    output_name = "output_b2p.json" #not used
    x = cordx
    y = cordy
    nr = 15
        
    pixel_coords = np.array([[x], [y], [1]])
    tvec, rvec, camera_matrix, dist = fp.read_wp2c(input_name)
    fTc = fp.read_c2f(input_name2)
    bTf_i, _ = fp.read_bTf(input_name3)
    #print("fTc: \n", fTc)
    #print("bTf_i: \n", bTf_i[nr])

    result, bTc, rot_c2p, trans_c2p, bTp, Spitze_mat = fp.calc_pixel2robot(tvec, rvec, camera_matrix, bTf_i, fTc, pixel_coords, nr, output_name, False)#printout results False
    #print(result.shape)
    print("results:")
    result[:3] = result[:3] / 1000
    print(result)

    #XML
    x_robot = result[0, 0]
    y_robot = result[1, 0]
    return x_robot, y_robot, result


def convert_coordinates(pixel_x, pixel_y):
    """
    NEW: Direct function call interface for robot_control.py
    Converts pixel coordinates to robot coordinates and writes result to robot_coordinates.txt
    
    Args:
        pixel_x: X-coordinate in pixels
        pixel_y: Y-coordinate in pixels
    
    Returns:
        tuple: (x_robot, y_robot) or (None, None) on error
    """
    try:
        print(f"DEBUG: Converting pixel coordinates: ({pixel_x}, {pixel_y})")
        
        # Core conversion using existing pixel2robot function
        x_robot, y_robot, result = pixel2robot(pixel_x, pixel_y)
        
        # Write result to robot_coordinates.txt (same as original)
        results = np.array(result)
        file_path = 'txt_file/robot_coordinates.txt'
        
        with open(file_path, 'w') as file:
            for result in results[:2]:  # Select only the first two elements
                file.write(f"{result[0]:.5f}\n")
        
        print(f"DEBUG: Robot coordinates written to {file_path}")
        print(f"DEBUG: X: {x_robot:.5f}, Y: {y_robot:.5f}")
        
        return x_robot, y_robot
        
    except Exception as e:
        print(f"ERROR in convert_coordinates: {e}")
        return None, None


def write_robot_coordinates(result):
    """
    Helper function to write robot coordinates to file
    """
    results = np.array(result)
    file_path = 'txt_file/robot_coordinates.txt'
    
    with open(file_path, 'w') as file:
        for result in results[:2]:  # Select only the first two elements
            file.write(f"{result[0]:.5f}\n")


# Script execution mode - supports multiple input methods
if __name__ == "__main__":
    # NEW: Option 1 - Command line parameters (no temporary files)
    if len(sys.argv) == 3:
        # Direct parameter input: python pixel2robot_multi.py 1530 696
        try:
            cordx = int(sys.argv[1])
            cordy = int(sys.argv[2])
            print(f"DEBUG: Using command line parameters: ({cordx}, {cordy})")
            
            # Execute conversion
            x, y, result = pixel2robot(cordx, cordy)
            
            # Write results to file
            write_robot_coordinates(result)
            
            print(f"SUCCESS: Conversion completed - X: {x:.5f}, Y: {y:.5f}")
            
        except (ValueError, IndexError) as e:
            print(f"ERROR: Invalid command line parameters: {e}")
            print("Usage: python pixel2robot_multi.py <pixel_x> <pixel_y>")
            sys.exit(1)
    
    # LEGACY: Option 2 - File-based input (maintains predecessor compatibility)
    else:
        # Original behavior: read from center_point.txt
        file_path = 'txt_file/center_point.txt'
        
        try:
            with open(file_path, 'r') as file:
                content = file.read()
                values = content.strip().split()  # Split by whitespace
                
                # Convert the values to integers and store them in x and y
                cordx = int(values[0].strip())
                cordy = int(values[1].strip())
                
                print(f"DEBUG: Using file-based input from {file_path}: ({cordx}, {cordy})")
        
        except FileNotFoundError:
            print(f"ERROR: File not found: {file_path}")
            sys.exit(1)
        except (ValueError, IndexError) as e:
            print(f"ERROR: Invalid file format in {file_path}: {e}")
            sys.exit(1)
        
        # Execute conversion (original behavior)
        x, y, result = pixel2robot(cordx, cordy)
        
        # Write results to file (original behavior)
        write_robot_coordinates(result)
        
        print(f"SUCCESS: File-based conversion completed - X: {x:.5f}, Y: {y:.5f}")
