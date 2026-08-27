from PIL import Image
import Identifier
import shutil
import os
import Classifier
import torch
folder_path = "object"
try:
    shutil.rmtree(folder_path)
    print("Folder and all its contents deleted successfully")
except OSError as e:
    print(f"Error: {e}")
predictor,device = Identifier.load_objectifier()
model, processor, text_features, texts = Classifier.load_classifier()
try:
    os.mkdir(folder_path)
    print(f"Directory '{folder_path}' created successfully.")
except FileExistsError:
    print(f"Directory '{folder_path}' already exists.")
except PermissionError:
    print("Permission denied.")
similarities = []
best_indices = []
image = Image.open("frid.jpg").convert("RGB")
length = Identifier.object_detector(image,device,predictor)
for start in range(0, length, 32):
    end = min(start + 32, length)
    similarity,best_index = Classifier.classy(start, end,model, processor, text_features, texts, device)
    similarities.append(similarity)
    best_indices.append(best_index)
similarities = torch.cat(similarities)
best_indices = torch.cat(best_indices)
for i, index in enumerate(best_indices):
    print(
        i,
        "→",
        texts[index.item()],
        similarities[i, index].item()
    )