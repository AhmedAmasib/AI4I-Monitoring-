import tarfile
import os

current_folder = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_folder)
zip_path = os.path.join(base_dir, "models", "model.tar.gz")
extract_path = os.path.join(base_dir, "models")

print(f"Searching for: {zip_path}")

if os.path.exists(zip_path):
    with tarfile.open(zip_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print("SUCCESS! Your model is unpacked in the 'models' folder.")
else:
    print("STILL NOT FOUND.")
    print(f"Please move 'model.tar.gz' into: {extract_path}")