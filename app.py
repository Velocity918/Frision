from PIL import Image
import shutil
folder_path = "object"
try:
    shutil.rmtree(folder_path)
    print("Folder and all its contents deleted successfully")
except OSError as e:
    print(f"Error: {e}")
import os

folder_name = "object"
try:
    os.mkdir(folder_name)
    print(f"Directory '{folder_name}' created successfully.")
except FileExistsError:
    print(f"Directory '{folder_name}' already exists.")
except PermissionError:
    print("Permission denied.")
import handle2
image = Image.open("frid.jpg").convert("RGB")
length = handle2.object_detector(image)

