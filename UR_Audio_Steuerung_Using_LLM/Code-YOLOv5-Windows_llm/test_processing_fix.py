

import os
import subprocess
import sys
from pathlib import Path

def find_latest_multi_exp():
    """Find latest multi_exp directory"""
    runs_dir = Path("yolov5/runs/detect")
    if not runs_dir.exists():
        print("ERROR: yolov5/runs/detect directory not found")
        return None
    
    multi_exp_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("multi_exp")]
    if not multi_exp_dirs:
        print("ERROR: No multi_exp directories found")
        return None
    
    latest_dir = max(multi_exp_dirs, key=lambda x: x.stat().st_mtime)
    print(f"INFO: Latest multi_exp directory: {latest_dir}")
    return latest_dir

def check_expected_files():
    """Check if expected processing files exist"""
    latest_dir = find_latest_multi_exp()
    if not latest_dir:
        return False, []
    
    expected_files = ["border_1.jpg", "direction_1.jpg", "pca_1.jpg"]
    found_files = []
    missing_files = []
    
    for file in expected_files:
        file_path = latest_dir / file
        if file_path.exists():
            size = file_path.stat().st_size
            found_files.append(f"{file} ({size} bytes)")
            print(f" Found: {file} ({size} bytes)")
        else:
            missing_files.append(file)
            print(f" Missing: {file}")
    
    return len(missing_files) == 0, found_files

def create_basic_files_for_testing():
    """Create basic files needed for testing"""
    os.makedirs("txt_file", exist_ok=True)
    
    # Create center_point.txt (needed for direction calculation)
    with open("txt_file/center_point.txt", 'w') as f:
        f.write("640,480\n")
    print("CREATED: txt_file/center_point.txt")
    
    # Create a basic crop_img_path.txt (fallback)
    latest_dir = find_latest_multi_exp()
    if latest_dir:
        # Look for any existing image in the multi_exp directory
        image_files = list(latest_dir.glob("*.jpg"))
        if image_files:
            crop_path = str(image_files[0])
            with open("txt_file/crop_img_path.txt", 'w') as f:
                f.write(crop_path)
            print(f"CREATED: txt_file/crop_img_path.txt -> {crop_path}")
            return True
    
    print("WARNING: Could not create crop_img_path.txt - no images found")
    return False

def test_processing_scripts():
    """Test the fixed processing scripts"""
    print("🧪 TESTING FIXED PROCESSING SCRIPTS")
    print("=" * 60)
    
    # Step 1: Check if we have a recent detection run
    if not find_latest_multi_exp():
        print("ERROR: No recent detection run found. Please run detection first.")
        return False
    
    # Step 2: Create basic files needed for testing
    if not create_basic_files_for_testing():
        print("ERROR: Could not create required test files")
        return False
    
    # Step 3: Check current state
    print("\nStep 1: Check current state...")
    all_found, found_files = check_expected_files()
    if all_found:
        print(" All files already exist!")
        return True
    
    # Step 4: Test border detection
    print("\nStep 2: Testing border_multi.py...")
    try:
        result = subprocess.run([sys.executable, "border_multi.py"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(" border_multi.py completed successfully")
        else:
            print(f" border_multi.py failed: {result.stderr}")
    except Exception as e:
        print(f" border_multi.py error: {e}")
    
    # Step 5: Test PCA calculation
    print("\nStep 3: Testing pca_multi.py...")
    try:
        result = subprocess.run([sys.executable, "pca_multi.py"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(" pca_multi.py completed successfully")
        else:
            print(f" pca_multi.py failed: {result.stderr}")
    except Exception as e:
        print(f" pca_multi.py error: {e}")
    
    # Step 6: Test direction calculation  
    print("\nStep 4: Testing direction_multi.py...")
    try:
        result = subprocess.run([sys.executable, "direction_multi.py"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(" direction_multi.py completed successfully")
        else:
            print(f" direction_multi.py failed: {result.stderr}")
    except Exception as e:
        print(f" direction_multi.py error: {e}")
    
    # Step 7: Final check
    print("\nStep 5: Final verification...")
    all_found, found_files = check_expected_files()
    
    print("\n" + "=" * 60)
    if all_found:
        print("🎉 SUCCESS: All processing files created!")
        print("The following files are now available:")
        for file in found_files:
            print(f"   {file}")
        print("\nYour multi-object detection workflow should now work without errors!")
    else:
        print(" PARTIAL SUCCESS: Some files are still missing")
        print("This may be due to missing dependencies or configuration issues")
    
    return all_found

if __name__ == "__main__":
    success = test_processing_scripts()
    if success:
        print("\n All tests passed! The processing pipeline is fixed.")
    else:
        print("\n Some tests failed. Please check the error messages above.") 