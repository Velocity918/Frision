import torch
from transformers import AutoProcessor, AutoModel
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")

import json
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
with open("food_names.json") as file:
    data = json.load(file)
texts = data
inputs = processor(text=texts, padding="max_length", return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
   text_output = model.get_text_features(**inputs)
   text_features = text_output.pooler_output

text_features = text_features / text_features.norm(dim=-1, keepdim=True)
torch.save(text_features.cpu(), "food_embeddings.pt")

print("Saved:", text_features.shape)