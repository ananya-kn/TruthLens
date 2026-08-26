import os
import shutil

# --- Configuration ---
OUTPUTS_DIR = "outputs"   # Folder with your txt files
DEST_DIR = "."  # Where grouped dirs will be created
# -------------------

# Check if the source directory exists before starting
if not os.path.isdir(OUTPUTS_DIR):
    print(f"Error: Source directory '{OUTPUTS_DIR}' not found. Exiting.")
    exit() # Stop the script if the source folder is missing

os.makedirs(DEST_DIR, exist_ok=True)

print(f"Scanning '{OUTPUTS_DIR}' for .txt files...")

for filename in os.listdir(OUTPUTS_DIR):
    if filename.endswith(".txt") and "_" in filename:
        prefix = filename.split("_")[0]
        prefix_dir = os.path.join(DEST_DIR, prefix)
        os.makedirs(prefix_dir, exist_ok=True)

        src = os.path.join(OUTPUTS_DIR, filename)
        dst = os.path.join(prefix_dir, filename)

        shutil.copy2(src, dst)
        
        print(f"Copied {filename} → {prefix_dir}")
    else:
        print(f"Skipping '{filename}' (not a valid format or not a .txt file).")

print("\nScript finished.")