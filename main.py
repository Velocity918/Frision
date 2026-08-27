import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import food_constant
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection
)
import json
def formatting(start, stop):
    adat = ""
    with open('food_names.json',mode="r",) as file:
        data = json.load(file)
    for n in range(start,stop):
        adat = adat +data[n] +". "
    return adat
MODEL_ID = "IDEA-Research/grounding-dino-tiny"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)
print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    MODEL_ID
).to(device)
print("Model loaded!")
image = Image.open("fridge.jpg").convert("RGB")
with open("food_names.json") as file:
    data = json.load(file)
tally = 0
batches = 50
for n in range(0,(len(data)//batches)+1):
    form = formatting(tally,min(len(data),(n+1)*batches))
    tally = (n+1)*batches

    text = form

    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.30,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )

    result = results[0]
    plt.imshow(image)
    ax = plt.gca()
    for box, score, label in zip(
        result["boxes"],
        result["scores"],
        result["text_labels"]
    ):
        print(
            f"{label}: {score.item():.3f} "
            f"box={box.tolist()}"
        )
        box =box.cpu().tolist()
        rect1 = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect1)
    plt.savefig(f"detections{n}.png")
    plt.close()