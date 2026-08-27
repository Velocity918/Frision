import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import json

MODEL_NAME = "google/siglip-so400m-patch14-384"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Load saved embeddings
text_features = torch.load(
    "food_embeddings.pt",
    map_location=device
)

with open("food_names.json") as file:
    texts = json.load(file)

image = Image.open("object/15.jpg")

inputs = processor(
    images=image,
    return_tensors="pt"
)

inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    image_output = model.get_image_features(**inputs)
    image_features = image_output.pooler_output

# Normalize
image_features = image_features / image_features.norm(
    dim=-1,
    keepdim=True
)

# Cosine similarity
similarities = image_features @ text_features.T

best_index = similarities[0].argmax().item()

print(texts[best_index])
print(similarities[0][best_index].item())