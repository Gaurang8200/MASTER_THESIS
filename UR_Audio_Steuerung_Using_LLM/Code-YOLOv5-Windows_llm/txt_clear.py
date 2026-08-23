import os

def clear_txt_files():
    """Clear all text files in txt_file directory"""
    txt_dir = "txt_file"
    
    if not os.path.exists(txt_dir):
        print(f"WARNING: Directory {txt_dir} does not exist")
        return False
    
    files_cleared = 0
    
    try:
        for filename in os.listdir(txt_dir):
            if filename.endswith('.txt') or filename.endswith('.json'):
                file_path = os.path.join(txt_dir, filename)
                
                # Check if it's a file (not directory)
                if os.path.isfile(file_path):
                    with open(file_path, 'w') as f:
                        f.write("")  # Clear the file
                    files_cleared += 1
                    print(f"FILE: Cleared {filename}")
        
        print(f"SUCCESS: Cleared {files_cleared} files in {txt_dir}")
        return True
        
    except Exception as e:
        print(f"ERROR: Error clearing files: {e}")
        return False

def main():
    """Main function to clear all txt files"""
    print("CLEANUP: Starting text file cleanup...")
    
    result = clear_txt_files()
    
    if result:
        print("SUCCESS: Text file cleanup completed")
    else:
        print("ERROR: Text file cleanup failed")
